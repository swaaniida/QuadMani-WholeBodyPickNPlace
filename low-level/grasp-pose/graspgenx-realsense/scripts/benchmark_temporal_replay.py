"""Offline CPU benchmark only; cached REAL RGB-D/mask/candidates, no model request.
Static RGB-D repeats measure cost, NOT dynamic tracking accuracy or live pose FPS.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_client'))
from background_icp import BackgroundICP
from temporal_selection import TemporalConfig,select_candidates
from pointcloud import FilterConfig
from camera_motion import read_rgbd


def snapshot(path,index):
    rgbd=read_rgbd(path)
    with np.load(path,allow_pickle=False) as d:
        mask=d['instance_mask'].copy();filtered=d['filtered_mask'].copy();raw=d['depth_raw'].copy()
        scale=float(d['depth_scale'])
    now=time.perf_counter()
    f=SimpleNamespace(frame_id=f'offline-cached-rgbd-{index}',ready_monotonic=now,camera_timestamp_ms=index*100.,
        host_timestamp_s=time.time(),depth_m=rgbd['depth'],depth_raw=raw,depth_scale=scale,K=rgbd['K'],
        bgr=rgbd['rgb'][:,:,::-1].copy(),unit_rays=None,intrinsics={},color_intrinsics={},
        acquisition_ms=0.,alignment_ms=0.,source='OFFLINE cached static real RGB-D; replay timestamps; not a live camera')
    return dict(frame=f,mask=mask,filtered=filtered,config=FilterConfig(foreground_band_m=0),actual_fps=0.,tracking_rtt_ms=0.,
        response=dict(session='offline-static',generation=1,tracking_valid=True,grasp_allowed=True,status='offline_replay',
                      metrics=dict(sam_init_ms=None,vos_ms=None),mask_depth_ms=0.))


def run(args):
    import trimesh
    from threadpoolctl import threadpool_limits
    out=ROOT/'outputs/temporal_selection';out.mkdir(exist_ok=True)
    frame=next((ROOT/'local_client/example_auto_session/frames').glob('*.npz'))
    mesh=trimesh.load(ROOT/'local_client/assets/franka_umi_vis_mesh.obj',force='mesh',process=False)
    with np.load(ROOT/'outputs/tracking/grasps/integration_static_rgbd.npz',allow_pickle=False) as d:
        g=d['grasps'].copy();c=d['confidences'].copy();points=d['point_cloud'].copy()
    engine=BackgroundICP(out,TemporalConfig());rows=[];previous=None
    try:
        with threadpool_limits(limits=2):
            for i in range(args.warmup+args.frames):
                snap=snapshot(frame,i);start=time.perf_counter();result=engine.process(snap);end=time.perf_counter()
                T=result['T_world_camera']
                choice=select_candidates(g,c,points,mesh.vertices,result['plane'],engine.cfg,previous,T,result['valid'],
                                         result['plane_metrics'].get('rmse_m',0.))
                if choice['selected'] is not None and result['valid']:previous=T@g[choice['selected']]
                else:previous=None
                row=dict(frame=i,status=result['status'],valid=result['valid'],total_processing_ms=(end-start)*1000,
                    icp_ms=result.get('icp_ms'),plane_ms=result.get('plane_ms'),fitness=result.get('fitness'),rmse_m=result.get('rmse_m'),
                    ground_status=result['plane_metrics']['status'],ground_valid=result['plane'] is not None,
                    valid_candidates=int(choice['valid'].sum()),selection_status=choice['status'],
                    baseline_index=choice['baseline_selected'],new_index=choice['selected'])
                if i>=args.warmup:rows.append(row)
                if i%10==0:print(i,row,flush=True)
    finally:engine.close()
    with (out/'static_replay.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def stats(key):
        values=[r[key] for r in rows if r[key] is not None]
        return dict(mean=float(np.mean(values)),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95))) if values else None
    report=dict(source='Official real_world/00 cached real RGB-D + saved actual SAM mask and GraspGenX candidates, repeated static input',
        frames=len(rows),warmup=args.warmup,no_live_model_requests=True,windows_hardware_tested=False,
        timing_scope='Synchronous CPU replay compute; excludes USB, network, model inference, recording and GUI; NOT achieved live pose/streaming FPS',
        valid_registration=sum(r['valid'] for r in rows),ground_valid=sum(r['ground_valid'] for r in rows),
        total_processing_ms=stats('total_processing_ms'),icp_ms=stats('icp_ms'),plane_ms=stats('plane_ms'))
    report['compute_only_fps']=1000/report['total_processing_ms']['mean']
    (out/'static_replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',type=int,default=100);p.add_argument('--warmup',type=int,default=10)
    args=p.parse_args()
    if args.frames<1 or args.warmup<0:p.error('Invalid frame count')
    run(args)
