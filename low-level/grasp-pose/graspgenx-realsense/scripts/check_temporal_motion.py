"""Real moving RGB-D background ICP check; no model, no fabricated grasp poses."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_client'))
from background_icp import BackgroundICP
from temporal_selection import TemporalConfig


def main():
    source=ROOT/'outputs/object_fixed/open3d_data/extract/SampleRedwoodRGBDImages'
    out=ROOT/'outputs/temporal_selection/moving_scene_check';out.mkdir(parents=True,exist_ok=True)
    engine=BackgroundICP(out,TemporalConfig(keyframe_interval=2,keyframe_realign_every=1))
    rows=[]
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            for i in range(5):
                depth=cv2.imread(str(source/'depth'/f'{i:05d}.png'),cv2.IMREAD_UNCHANGED).astype(np.float32)*.001
                f=SimpleNamespace(frame_id=f'redwood-moving-{i}',camera_timestamp_ms=i*100.,host_timestamp_s=None,
                    ready_monotonic=time.perf_counter(),depth_m=depth,unit_rays=None,K=np.array([[525,0,319.5],[0,525,239.5],[0,0,1.]]))
                start=time.perf_counter()
                # This static-scene dataset has no target annotations: scene-only ICP validation.
                r=engine.process(dict(frame=f,mask=np.zeros_like(depth,np.uint8),response=dict(tracking_valid=True)))
                rows.append(dict(frame=i,status=r['status'],valid=r['valid'],fitness=r.get('fitness'),rmse_m=r.get('rmse_m'),
                    processing_ms=(time.perf_counter()-start)*1000,T_world_camera=r['T_world_camera'].tolist() if r['valid'] else None))
    finally:engine.close()
    report=dict(source='Official Open3D SampleRedwoodRGBDImages, five genuine moving RGB-D frames',
        scope='Scene-only background ICP/keyframe acceptance; no target mask annotations or model/grasp evaluation; NOT camera-motion ground truth',rows=rows)
    (out/'moving_scene.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    assert sum(r['valid'] for r in rows)>=3,'Too few accepted frames; inspect report'


if __name__=='__main__':main()
