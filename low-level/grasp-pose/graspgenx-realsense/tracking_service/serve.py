"""Loopback Tracking SAM service; official GraspGenX scene-depth RPC worker."""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import sys
import time
import uuid
import msgpack
import msgpack_numpy
import numpy as np
import zmq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_client'))
from backend import TrackingBackend
from grasp_client import GraspClient,load_sweep_params,validate_result
from pointcloud import FilterConfig
from tracking_protocol import decode_frame,filter_tracked_depth,png_mask,MaskGuard


def append_csv(path,row):
    exists=path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if not exists:w.writeheader()
        w.writerow(row)


class TrackingService:
    def __init__(self,grasp_port=5556,max_age_s=2.0):
        self.backend=TrackingBackend()
        self.guard=MaskGuard()
        self.snapshots=OrderedDict()
        self.session=None
        self.last_activity=0.
        self.seq=-1
        self.stream_id=None
        self.last_shape=None
        self.generation=0
        self.latest_usable=False
        self.last_frame_id=None
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='grasp-rpc')
        self.job=None
        self.grasp_port=grasp_port
        self.max_age_s=max_age_s
        self.out=ROOT/'outputs/tracking'
        self.out.mkdir(exist_ok=True)

    def reset(self):
        self.backend.reset()
        self.guard.reset()
        self.snapshots.clear()
        self.generation+=1
        self.latest_usable=False
        self.last_frame_id=None
        self.stream_id=None
        self.last_shape=None

    def authorize(self,session):
        if not isinstance(session,str) or not session or len(session)>100:
            raise ValueError('A valid session ID is required')
        now=time.monotonic()
        if session!=self.session:
            if self.session is not None and now-self.last_activity<30:
                raise ValueError('Tracker is owned by another active client; close it or wait 30s')
            self.reset()
            self.session=session
            self.seq=-1
        self.last_activity=now

    def frame(self,p):
        start=time.perf_counter()
        seq=int(p['seq'])
        if seq<=self.seq:raise ValueError('Out-of-order/replayed frame sequence')
        self.seq=seq
        bgr,depth,K=decode_frame(p)
        decoded=time.perf_counter()
        frame_id=str(p['frame_id'])
        if not frame_id or len(frame_id)>250:raise ValueError('Invalid frame ID')
        initial='point' in p or bool(p.get('text'))
        if initial:
            self.reset()
        elif self.stream_id is not None and (p.get('stream_id')!=self.stream_id or bgr.shape!=self.last_shape):
            self.guard.lost=True
            self.guard.reason='camera_reconnected_or_resolution_changed'
        self.stream_id=p.get('stream_id')
        self.last_shape=bgr.shape
        mask=np.zeros(depth.shape,np.uint8)
        metrics=dict(sam_init_ms=None,vos_ms=None,sam_score=None,peak_allocated_mib=None,
                     allocated_mib=None,reserved_mib=None)
        if self.guard.lost:
            ok,status=False,self.guard.reason
        elif not initial and not self.backend.tracker.is_tracking():
            ok,status=False,'click_to_initialize'
        else:
            try:
                mask,metrics=self.backend.process(bgr[:,:,::-1].copy(),point=p.get('point'),text=p.get('text'))
                ok,status=self.guard.check(mask,initial=initial)
            except Exception:
                self.latest_usable=False
                self.guard.lost=True
                self.guard.reason='model_error_reinitialize'
                raise
        model_done=time.perf_counter()
        cfg=FilterConfig(**p.get('filters',{}))
        filtered=filter_tracked_depth(depth,mask,cfg) if ok else np.zeros_like(mask)
        pixels=int(mask.sum())
        valid_pixels=int(filtered.sum())
        self.latest_usable=ok and valid_pixels>=100
        if ok and not self.latest_usable:status='tracking_no_valid_depth'
        self.last_frame_id=frame_id
        if self.latest_usable:
            # Deep snapshot retained for G; no subsequent tracking update mutates it.
            self.snapshots[frame_id]=dict(frame_id=frame_id,depth=depth.copy(),K=K.copy(),
                                          mask=filtered.copy(),raw_mask=mask.copy(),
                                          ready=time.monotonic(),generation=self.generation,
                                          intrinsics=p.get('intrinsics',{}),
                                          filters=p.get('filters',{}))
            while len(self.snapshots)>8:self.snapshots.popitem(last=False)
        processing_ms=(time.perf_counter()-model_done)*1000
        response=dict(frame_id=frame_id,session=self.session,generation=self.generation,seq=seq,
                      status=status,tracking_valid=ok,grasp_allowed=self.latest_usable,
                      instance_id=1,mask_png=png_mask(mask),filtered_mask_png=png_mask(filtered),
                      mask_pixels=pixels,filtered_pixels=valid_pixels,
                      valid_depth_ratio=valid_pixels/max(pixels,1),metrics=metrics,
                      decode_ms=(decoded-start)*1000,mask_depth_ms=processing_ms)
        append_csv(self.out/'service_frames.csv',dict(session=self.session,frame_id=frame_id,seq=seq,
                    status=status,mask_pixels=pixels,filtered_pixels=valid_pixels,
                    decode_ms=response['decode_ms'],mask_depth_ms=processing_ms,**metrics))
        return response

    def submit_grasp(self,p):
        if self.job is not None and not self.job['future'].done():
            raise ValueError('One grasp job is already running; no request queue')
        frame_id=p['frame_id']
        if not self.latest_usable or self.guard.lost:
            raise ValueError('Tracking/depth is invalid; reinitialize before grasping')
        if frame_id!=self.last_frame_id or frame_id not in self.snapshots:
            raise ValueError('Grasp requires latest matched tracked frame; wait for a new mask')
        snap=self.snapshots[frame_id]
        if time.monotonic()-snap['ready']>self.max_age_s:
            raise ValueError('Tracked frame is stale; wait for a fresh frame')
        if np.any(snap['intrinsics'].get('coeffs',[])):
            raise ValueError('Official scene-depth requires rectified pinhole depth/K; nonzero distortion is unsupported')
        planner=p.get('planner','graspmoe')
        count=int(p.get('num_grasps',200))
        topk=int(p.get('topk',100))
        threshold=float(p.get('threshold',-1))
        if planner not in ('graspmoe','diffusion') or not 1<=count<=1000 or not (topk==-1 or 1<=topk<=1000):
            raise ValueError('Invalid planner/count/topk')
        job_id=uuid.uuid4().hex
        future=self.pool.submit(self.run_grasp,snap,planner,count,topk,threshold)
        self.job=dict(id=job_id,future=future,generation=self.generation,frame_id=frame_id)
        return dict(job_id=job_id,frame_id=frame_id,generation=self.generation,status='running')

    def run_grasp(self,snap,planner,count,topk,threshold):
        start=time.perf_counter()
        client=GraspClient(port=self.grasp_port,timeout_ms=60000)
        try:
            wire,rtt=client.infer_scene_depth(snap['depth'],snap['K'],snap['mask'].astype(np.int32),
                                             load_sweep_params(),planner=planner,num_grasps=count)
        finally:client.close()
        ids=list(wire['instance_ids'])
        if 1 not in ids:raise ValueError('GraspGenX returned no valid grasps for instance 1')
        idx=ids.index(1)
        g=np.asarray(wire['grasps'][idx],np.float32)
        s=np.asarray(wire['confidences'][idx],np.float32)
        validate_result(g,s)
        order=np.argsort(-s)
        if threshold>0:order=order[s[order]>=threshold]
        if topk!=-1:order=order[:topk]
        tags=wire.get('branch_tags',[[] for _ in ids])[idx]
        elapsed=(time.perf_counter()-start)*1000
        append_csv(self.out/'service_grasps.csv',dict(frame_id=snap['frame_id'],planner=planner,
                    grasps=len(order),grasp_rpc_rtt_ms=rtt,
                    grasp_infer_ms=wire['timing']['infer_ms'],
                    grasp_handler_sync_ms=wire['timing'].get('handler_sync_ms'),grasp_job_ms=elapsed))
        return dict(frame_id=snap['frame_id'],generation=snap['generation'],
                    candidate_count_generated=len(g),
                    candidate_count_after_threshold=int(np.count_nonzero(s>=threshold)) if threshold>0 else len(s),
                    candidate_count_after_filter=len(order),
                    grasps=g[order],confidences=s[order],branch_tags=[tags[i] for i in order] if tags else ['diff']*len(order),
                    server_timing=wire['timing'],gpu=wire.get('gpu',{}),grasp_rpc_rtt_ms=rtt,
                    grasp_job_ms=elapsed,filters=snap['filters'])

    def dispatch(self,p):
        action=p.get('action')
        if action=='health':return dict(status='ok',service='Tracking_SAM+AOTT',grasp_port=self.grasp_port)
        self.authorize(p.get('session'))
        if action=='reset':
            self.reset()
            return dict(status='reset',generation=self.generation)
        if action=='close':
            self.reset()
            self.session=None
            self.seq=-1
            return dict(status='closed')
        if action=='frame':return self.frame(p)
        if action=='grasp_submit':return self.submit_grasp(p)
        if action=='grasp_poll':
            if self.job is None or p['job_id']!=self.job['id']:raise ValueError('Unknown grasp job')
            if self.job['generation']!=self.generation:raise ValueError('Selection changed; old grasp result discarded')
            if not self.job['future'].done():return dict(status='running',job_id=self.job['id'])
            return dict(status='done',job_id=self.job['id'],result=self.job['future'].result())
        raise ValueError(f'Unknown action: {action}')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--port',type=int,default=5557)
    p.add_argument('--grasp-port',type=int,default=5556)
    args=p.parse_args()
    service=TrackingService(grasp_port=args.grasp_port)
    socket=zmq.Context.instance().socket(zmq.REP)
    socket.setsockopt(zmq.MAXMSGSIZE,32*1024*1024)
    socket.setsockopt(zmq.LINGER,0)
    socket.bind(f'tcp://127.0.0.1:{args.port}')
    print(f'Tracking SAM listening on 127.0.0.1:{args.port}',flush=True)
    try:
        while True:
            raw=socket.recv()
            start=time.perf_counter()
            try:
                packet=msgpack.unpackb(raw,raw=False,object_hook=msgpack_numpy.decode)
                result=service.dispatch(packet)
                result['service_ms']=(time.perf_counter()-start)*1000
            except Exception as exc:
                import traceback
                traceback.print_exc()
                result=dict(error=f'{type(exc).__name__}: {exc}')
            socket.send(msgpack.packb(result,use_bin_type=True,default=msgpack_numpy.encode))
    except KeyboardInterrupt:pass
    finally:
        socket.close()
        service.pool.shutdown(wait=False,cancel_futures=True)
