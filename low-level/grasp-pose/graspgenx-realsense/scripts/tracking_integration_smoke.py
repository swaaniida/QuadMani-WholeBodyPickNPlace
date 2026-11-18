"""Real SAM+AOTT+GraspGenX on official recorded RGB-D, no invented depth.

The one recorded RGB-D frame is repeated, explicitly a static replay. Motion
tracking is separately checked with the genuine 82-frame official DAVIS demo.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from urllib.request import urlopen
import uuid
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_client'))
from grasp_client import GraspClient,validate_result
from pointcloud import FilterConfig,deproject,downsample
from tracking_protocol import frame_packet,read_mask
from recording import save_result
from visualization import overlay,GraspViewer
from tracking_client import csv_row


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--port',type=int,default=5557)
    p.add_argument('--text-test',action='store_true')
    args=p.parse_args()
    source=ROOT/'upstream/GraspGenX/assets/sample_data/real_world/00'
    md=json.loads((source/'meta_data.json').read_text())
    bgr=cv2.imread(str(source/'rgb.png'))
    depth=np.load(source/'depth.npy').astype(np.float32)
    K=np.asarray(md['intrinsics'],np.float32)
    intr=dict(width=depth.shape[1],height=depth.shape[0],fx=float(K[0,0]),fy=float(K[1,1]),
              ppx=float(K[0,2]),ppy=float(K[1,2]),model='distortion.none',coeffs=[0.]*5)
    cfg=FilterConfig(foreground_band_m=0)
    session='integration-'+uuid.uuid4().hex
    c=GraspClient(port=args.port,timeout_ms=60000)
    frames=[]
    seq=[0]
    def send_frame(point=None,text=None):
        seq[0]+=1
        f=SimpleNamespace(frame_id=f'official-static-rgbd-{seq[0]:05d}',bgr=bgr,
            depth_raw=depth,depth_m=depth,K=K,depth_scale=1.,intrinsics=intr,color_intrinsics=intr,
            unit_rays=None,camera_timestamp_ms=None,host_timestamp_s=None,serial=None,
            source='official recorded real_world/00, repeated static RGB-D replay',ready_monotonic=time.perf_counter())
        packet,encoding=frame_packet(f,session,seq[0],cfg,point=point,text=text)
        response,rtt=c.request(packet)
        assert response['frame_id']==f.frame_id
        mask=read_mask(response['mask_png'],depth.shape)
        filtered=read_mask(response['filtered_mask_png'],depth.shape)
        assert response['grasp_allowed'],response
        frames.append(dict(frame_id=f.frame_id,tracking_rtt_ms=rtt,**encoding,
                           sam_init_ms=response['metrics']['sam_init_ms'],vos_ms=response['metrics']['vos_ms'],
                           status=response['status'],mask_pixels=int(mask.sum()),
                           peak_allocated_mib=response['metrics']['peak_allocated_mib']))
        return f,response,mask,filtered
    viewer=None
    try:
        assert c.health()['status']=='ok'
        matched=send_frame(point=(620,450))  # positive click on recorded mustard bottle
        for _ in range(5):matched=send_frame()
        f,response,mask,filtered=matched
        request_start=time.perf_counter()
        submitted,_=c.request(dict(action='grasp_submit',session=session,frame_id=f.frame_id))
        during=0
        while True:
            send_frame()
            polled,poll_rtt=c.request(dict(action='grasp_poll',session=session,job_id=submitted['job_id']))
            during+=1
            if polled['status']=='done':break
        result=polled['result']
        assert result['frame_id']==f.frame_id
        validate_result(result['grasps'],result['confidences'])
        assert len(result['grasps'])==100
        points,v,u=deproject(f.depth_m,K,filtered>0)
        points,colors=downsample(points,bgr[v,u,::-1],cfg.voxel_m,cfg.max_points)
        image=overlay(bgr,K,result['grasps'],result['confidences'],f.frame_id)
        viewer=GraspViewer(port=18082)
        viewer.show(points,colors,result['grasps'],result['confidences'],f.frame_id)
        assert urlopen('http://127.0.0.1:18082',timeout=5).status==200
        result['network_rtt_ms']=result['grasp_rpc_rtt_ms']
        save_result(ROOT/'outputs/tracking','integration_static_rgbd',f,points,colors,filtered,
                    result,image,cfg,(0,0,bgr.shape[1],bgr.shape[0]),'tracking_scene_depth','graspmoe')
        cv2.imwrite(str(ROOT/'outputs/tracking/depth/integration_raw_tracking_mask.png'),mask*255)
        grasp_pipeline_elapsed_ms=(time.perf_counter()-request_start)*1000
        # Reset invalidates previous result generation and stops G until a new click.
        c.request(dict(action='reset',session=session))
        try:
            c.request(dict(action='grasp_submit',session=session,frame_id=f.frame_id))
            raise AssertionError('G after reset must be rejected')
        except RuntimeError:pass
        text_result=None
        if args.text_test:
            text_frame,text_response,text_mask,_=send_frame(text='mustard bottle')
            text_result=dict(frame_id=text_frame.frame_id,mask_pixels=int(text_mask.sum()),status=text_response['status'])
            cv2.imwrite(str(ROOT/'outputs/tracking/text_mask.png'),text_mask*255)
            c.request(dict(action='reset',session=session))
        assert 'torch' not in sys.modules
        report=dict(source='actual official RGB-D data, static replay; not live RealSense',
            frames=frames,grasp_frame_id=f.frame_id,grasp_count=len(result['grasps']),points=len(points),
            tracking_frames_processed_during_grasp=during,grasp_rpc_rtt_ms=result['grasp_rpc_rtt_ms'],
            grasp_timing=result['server_timing'],grasp_gpu=result['gpu'],
            end_to_end_with_visualization_and_save_ms=grasp_pipeline_elapsed_ms,
            viewer_http=200,cad_count=len(viewer.mesh_handles),text_initialization=text_result,
            reset_guard='passed',client_torch_imported=False,windows_hardware_tested=False)
        (ROOT/'outputs/tracking/integration_summary.json').write_text(json.dumps(report,indent=2))
        for row in frames:
            csv_row(ROOT/'outputs/tracking/integration_frames.csv',row)
        csv_row(ROOT/'outputs/tracking/integration_grasps.csv',dict(
            source='official_static_rgbd_replay',frame_id=f.frame_id,
            grasp_rpc_rtt_ms=result['grasp_rpc_rtt_ms'],
            grasp_infer_ms=result['server_timing']['infer_ms'],
            grasp_handler_sync_ms=result['server_timing'].get('handler_sync_ms'),
            pipeline_with_visualization_and_save_ms=grasp_pipeline_elapsed_ms,
            grasps=len(result['grasps'])))
        print(json.dumps(report,indent=2))
    finally:
        try:c.request(dict(action='close',session=session))
        except Exception:pass
        c.close()
        if viewer is not None:viewer.close()
