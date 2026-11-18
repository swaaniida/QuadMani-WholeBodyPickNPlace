"""Bounded latest-frame networking worker; never loads SAM/torch on Windows."""
import csv
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from grasp_client import GraspClient
from tracking_protocol import frame_packet,read_mask


def csv_row(path,row):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    exists=path.exists()
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if not exists:w.writeheader()
        w.writerow(row)


def summary(path):
    if not Path(path).exists():return {}
    with Path(path).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    result={}
    if not rows:return result
    for key in rows[0]:
        if key.endswith('_ms'):
            values=[float(r[key]) for r in rows if r.get(key) not in ('',None)]
            if values:result[key]=dict(n=len(values),mean=float(np.mean(values)),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)))
    return result


class TrackingWorker:
    def __init__(self,capture,config,host='127.0.0.1',port=15557,fps=10,output=Path('outputs/tracking'),
                 planner='graspmoe',topk=100,threshold=-1,num_grasps=200):
        self.capture=capture
        self.config=config
        self.host,self.port=host,port
        self.fps=fps
        self.output=Path(output)
        self.planner,self.topk,self.threshold,self.num_grasps=planner,topk,threshold,num_grasps
        self.session=uuid.uuid4().hex
        self.lock=threading.Lock()
        self.stop=threading.Event()
        self.latest=None
        self.completed=None
        self.grasp_error=None
        self.frame_observer=None  # Optional nonblocking recording hook; inference protocol unchanged.
        self.prompt=None
        self.reset_pending=False
        self.grasp_pending=None
        self.grasp_busy=False
        self.revision=0
        self.status='Connecting to Tracking SAM...'
        self.thread=threading.Thread(target=self.run,name='tracking-network',daemon=True)

    def start(self):self.thread.start();return self

    def select(self,frame,point=None,text=None):
        with self.lock:
            self.revision+=1
            self.prompt=(frame,point,text,self.revision)
            self.latest=None
            self.completed=None
            self.grasp_error=None
            self.grasp_pending=None

    def reset(self):
        with self.lock:
            self.revision+=1
            self.reset_pending=True
            self.prompt=None
            self.latest=None
            self.completed=None
            self.grasp_error=None
            self.grasp_pending=None

    def request_grasp(self):
        with self.lock:
            if self.grasp_busy or self.grasp_pending is not None or self.completed is not None:
                raise ValueError('A grasp request is already running; no queue')
            self.grasp_pending=time.perf_counter()
            self.grasp_busy=True  # Reserve immediately, including the submit/ACK interval.

    def take_grasp_error(self):
        with self.lock:
            error=self.grasp_error
            self.grasp_error=None
            return error if error is not None and error['revision']==self.revision else None

    def _reject_grasp(self,message,revision):
        # Grasp rejection is not evidence that SAM/AOT tracking is invalid.
        with self.lock:
            self.grasp_busy=False
            self.grasp_error=dict(message=str(message),revision=revision)
        print(f'Grasp skipped; tracking continues: {message}',flush=True)

    def state(self):
        with self.lock:return self.latest,self.status,self.grasp_busy

    def take_result(self):
        with self.lock:
            result=self.completed
            self.completed=None
            return result

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)

    def run(self):
        client=GraspClient(self.host,self.port,timeout_ms=60000)
        seq=0
        last_id=None
        next_frame=0.
        last_done=None
        actual_fps=0.
        job=None
        try:
            while not self.stop.is_set():
                try:
                    with self.lock:
                        reset=self.reset_pending
                        self.reset_pending=False
                        prompt=self.prompt
                        self.prompt=None
                        revision=self.revision
                        cfg=self.config
                    if reset:
                        client.request(dict(action='reset',session=self.session))
                        job=None
                        self.grasp_busy=False
                        last_id=None
                    if prompt is not None:
                        frame,point,text,revision=prompt
                        job=None
                        self.grasp_busy=False
                    else:
                        frame=self.capture.latest()
                        point=text=None
                    with self.lock:
                        grasp_time=self.grasp_pending
                        self.grasp_pending=None
                        latest=self.latest
                    if grasp_time is not None:
                        if job is not None:
                            # Defensive: never replace an active job or enqueue another submit.
                            with self.lock:self.grasp_busy=True
                        elif latest is None or not latest['response']['grasp_allowed'] or time.perf_counter()-latest['frame'].ready_monotonic>2:
                            self._reject_grasp('No fresh valid mask/depth; waiting for tracking',revision)
                        else:
                            try:
                                submitted,_=client.request(dict(action='grasp_submit',session=self.session,
                                    frame_id=latest['frame'].frame_id,planner=self.planner,topk=self.topk,
                                    threshold=self.threshold,num_grasps=self.num_grasps))
                            except (RuntimeError,TimeoutError) as exc:
                                self._reject_grasp(exc,revision)
                            else:
                                job=dict(id=submitted['job_id'],snapshot=latest,revision=revision,
                                         requested_at=grasp_time,last_poll=0.)
                                with self.lock:self.grasp_busy=True
                    if job is not None and time.perf_counter()-job['last_poll']>.1:
                        try:
                            response,poll_rtt=client.request(dict(action='grasp_poll',session=self.session,job_id=job['id']))
                        except (RuntimeError,TimeoutError) as exc:
                            self._reject_grasp(exc,job['revision'])
                            job=None
                        else:
                            job['last_poll']=time.perf_counter()
                            if response['status']=='done':
                                result=response['result']
                                if result['frame_id']!=job['snapshot']['frame'].frame_id:
                                    raise ValueError('Mismatched grasp result frame ID')
                                with self.lock:
                                    if job['revision']==self.revision:
                                        self.completed=dict(snapshot=job['snapshot'],result=result,
                                                            requested_at=job['requested_at'],poll_rtt_ms=poll_rtt)
                                    self.grasp_busy=False
                                job=None
                    if frame is None or (prompt is None and (frame.frame_id==last_id or time.perf_counter()<next_frame)):
                        self.stop.wait(.01)
                        continue
                    seq+=1
                    packet,encoding=frame_packet(frame,self.session,seq,cfg,point=point,text=text)
                    response,rtt=client.request(packet)
                    if response.get('frame_id')!=frame.frame_id or response.get('session')!=self.session:
                        raise ValueError('Tracking frame/session mismatch')
                    mask=read_mask(response['mask_png'],frame.depth_m.shape)
                    filtered=read_mask(response['filtered_mask_png'],frame.depth_m.shape)
                    now=time.perf_counter()
                    if last_done is not None:
                        instant=1/max(now-last_done,1e-6)
                        actual_fps=instant if not actual_fps else .8*actual_fps+.2*instant
                    last_done=now
                    matched=dict(frame=frame,response=response,mask=mask,filtered=filtered,
                                 config=cfg,tracking_rtt_ms=rtt,actual_fps=actual_fps,**encoding)
                    with self.lock:
                        if revision==self.revision:
                            # Recording follows accepted selection revisions only.
                            observer=self.frame_observer
                            if observer is not None:
                                try:observer(matched)
                                except Exception as exc:print(f'Frame archive hook error: {exc}',flush=True)
                            self.latest=matched
                            self.status=response['status']
                    last_id=frame.frame_id
                    next_frame=now+max(0,1/self.fps-(rtt+encoding['frame_encode_ms'])/1000)
                    csv_row(self.output/'tracking_frames.csv',dict(frame_id=frame.frame_id,
                        status=response['status'],acquisition_ms=frame.acquisition_ms,
                        alignment_ms=frame.alignment_ms,conversion_ms=frame.conversion_ms,
                        **encoding,tracking_rtt_ms=rtt,service_ms=response['service_ms'],
                        frame_transport_and_serialization_ms=max(0,rtt-response['service_ms']),
                        sam_init_ms=response['metrics']['sam_init_ms'],vos_ms=response['metrics']['vos_ms'],
                        mask_depth_ms=response['mask_depth_ms'],matched_frame_age_ms=(now-frame.ready_monotonic)*1000,
                        actual_tracking_fps=actual_fps,mask_pixels=response['mask_pixels'],
                        filtered_pixels=response['filtered_pixels']))
                except Exception as exc:
                    self.status=f'Tracking/network error: {exc}'
                    print(self.status,flush=True)
                    with self.lock:self.latest=None
                    job=None
                    self.grasp_busy=False
                    # No replay of a timed-out initialization or G request.
                    # A transport timeout may have run remotely; require reset
                    # and a new click before trusting subsequent masks.
                    self.reset_pending=True
                    self.stop.wait(.5)
        finally:
            client.close()
            closing=GraspClient(self.host,self.port,timeout_ms=500)
            try:closing.request(dict(action='close',session=self.session))
            except Exception:pass
            finally:closing.close()


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description='Tracking SAM connection check (CPU only)')
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=15557)
    p.add_argument('--health',action='store_true')
    args=p.parse_args()
    c=GraspClient(args.host,args.port,timeout_ms=5000)
    try:print(json.dumps(c.health(),indent=2))
    finally:c.close()
