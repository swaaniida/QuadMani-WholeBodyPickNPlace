"""Actual official recorded RGB-D → ROI/depth → server → visualization/files.

This is dataset replay, never a claim of Windows USB camera validation.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'local_client'))
from grasp_client import GraspClient, load_sweep_params
from pointcloud import FilterConfig, prepare
from recording import save_result
from visualization import GraspViewer, overlay


if __name__ == '__main__':
    start=time.perf_counter()
    sample=ROOT/'upstream/GraspGenX/assets/sample_data/real_world/00'
    metadata=json.loads((sample/'meta_data.json').read_text())
    bgr=cv2.imread(str(sample/'rgb.png'))
    depth=np.load(sample/'depth.npy').astype(np.float32)
    seg=cv2.imread(str(sample/'seg.png'),cv2.IMREAD_UNCHANGED)
    # Annotation used only to choose an initial object bounding ROI. The actual
    # mask and point cloud sent to inference use our ROI/depth filter.
    v,u=np.where(seg==metadata['label_map']['obj_1'])
    roi=(int(u.min()),int(v.min()),int(u.max()-u.min()+1),int(v.max()-v.min()+1))
    K=np.asarray(metadata['intrinsics'],np.float32)
    intr=dict(width=depth.shape[1],height=depth.shape[0],fx=float(K[0,0]),fy=float(K[1,1]),
              ppx=float(K[0,2]),ppy=float(K[1,2]),model='distortion.none',coeffs=[0.]*5)
    frame=SimpleNamespace(frame_id='OFFICIAL_DATASET_real_world_00_obj1', bgr=bgr,
                          depth_raw=depth,depth_m=depth,K=K,unit_rays=None,
                          intrinsics=intr,color_intrinsics=intr,depth_scale=1.0,
                          camera_timestamp_ms=None,host_timestamp_s=None,serial=None,
                          source='official_recorded_rgbd; depth.npy already in meters; pinhole K supplied by dataset')
    loaded=time.perf_counter()
    cfg=FilterConfig(near_m=.1,far_m=3.,foreground_band_m=.07,voxel_m=.003)
    pc,colors,mask,timing=prepare(frame,roi,cfg)
    client=GraspClient()
    viewer=None
    try:
        result=client.infer(pc,sweep_params=load_sweep_params())
        assert len(result['grasps']) == 100
        viz_start=time.perf_counter()
        image=overlay(bgr,K,result['grasps'],result['confidences'],frame.frame_id)
        assert np.any(image!=bgr)
        viewer=GraspViewer(port=18080)
        viewer.show(pc,colors,result['grasps'],result['confidences'],frame.frame_id)
        with urlopen('http://127.0.0.1:18080',timeout=5) as response:
            assert response.status==200
        assert len(viewer.mesh_handles)==10
        assert viewer.mesh is not None
        vis_ms=(time.perf_counter()-viz_start)*1000
        save_ms=save_result(ROOT/'outputs','official_rgbd_replay',frame,pc,colors,mask,
                            result,image,cfg,roi,'params','graspmoe')
        report=dict(source='official bundled RGB-D replay, not Windows USB capture',
                    dataset='assets/sample_data/real_world/00',roi=roi,points=len(pc),
                    grasps=len(result['grasps']),max_score=float(result['confidences'].max()),
                    dataset_load_ms=(loaded-start)*1000,**timing,
                    network_rtt_ms=result['network_rtt_ms'],server_timing=result['server_timing'],
                    gpu=result['gpu'],visualization_build_and_http_ms=vis_ms,save_ms=save_ms,
                    replay_total_ms=(time.perf_counter()-start)*1000,
                    viewer_http_status=200,official_cad_grippers=10,
                    browser_interaction_tested=False,frame='camera_optical')
        (ROOT/'outputs/benchmarks/rgbd_replay.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally:
        client.close()
        if viewer is not None:
            viewer.close()
