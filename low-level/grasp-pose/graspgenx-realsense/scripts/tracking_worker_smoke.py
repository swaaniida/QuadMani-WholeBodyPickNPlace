"""Exercise the exact Windows networking worker using real recorded RGB-D.

This is a labeled static replay camera provider; SAM/AOTT/GraspGenX are real.
"""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_client'))
from tracking_client import TrackingWorker
from pointcloud import FilterConfig


class RecordedCapture:
    def __init__(self):
        source=ROOT/'upstream/GraspGenX/assets/sample_data/real_world/00'
        self.bgr=cv2.imread(str(source/'rgb.png'))
        self.depth=np.load(source/'depth.npy').astype(np.float32)
        self.K=np.array(json.loads((source/'meta_data.json').read_text())['intrinsics'],np.float32)
        K=self.K
        self.intr=dict(width=self.depth.shape[1],height=self.depth.shape[0],fx=float(K[0,0]),fy=float(K[1,1]),
                       ppx=float(K[0,2]),ppy=float(K[1,2]),model='distortion.none',coeffs=[0.]*5)
        self.epoch=time.perf_counter()
    def latest(self):
        seq=int((time.perf_counter()-self.epoch)*30)
        return SimpleNamespace(frame_id=f'recorded-static-replay-{seq}',bgr=self.bgr,
            depth_raw=self.depth,depth_m=self.depth,K=self.K,depth_scale=1.,
            intrinsics=self.intr,color_intrinsics=self.intr,unit_rays=None,
            camera_timestamp_ms=None,host_timestamp_s=None,acquisition_ms=None,alignment_ms=None,
            conversion_ms=None,ready_monotonic=self.epoch+seq/30,
            source='official recorded RGB-D repeated; not hardware capture')


def wait_for(test,seconds=60):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        result=test()
        if result:return result
        time.sleep(.05)
    raise TimeoutError('Worker check timed out')


if __name__=='__main__':
    capture=RecordedCapture()
    worker=TrackingWorker(capture,FilterConfig(foreground_band_m=0),port=5557,
                          output=ROOT/'outputs/tracking/worker_replay').start()
    try:
        wait_for(lambda:worker.state()[0])
        worker.select(capture.latest(),point=(620,450))
        def fresh():
            s=worker.state()[0]
            return s if s and s['response']['grasp_allowed'] and time.perf_counter()-s['frame'].ready_monotonic<1 else None
        first=wait_for(fresh)
        worker.request_grasp()
        result=wait_for(worker.take_result)
        assert result['result']['frame_id']==result['snapshot']['frame'].frame_id
        assert result['result']['grasps'].shape==(100,4,4)
        current=worker.state()[0]
        assert current['frame'].frame_id!=first['frame'].frame_id
        worker.reset()
        def uninitialized():
            state=worker.state()[0]
            return state if state and state['response']['status']=='click_to_initialize' else None
        wait_for(uninitialized)
        report=dict(source='static recorded RGB-D replay provider; no camera hardware',
                    real_grasps=len(result['result']['grasps']),matched_frame=result['result']['frame_id'],
                    frames_advanced_while_grasping=True,reset='passed',torch_imported='torch' in sys.modules)
        assert not report['torch_imported']
        (ROOT/'outputs/tracking/worker_summary.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally:worker.close()
