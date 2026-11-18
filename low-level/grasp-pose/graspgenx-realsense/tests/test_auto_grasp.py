"""Analytic test fixtures only; actual-model benchmark is a separate script."""
import csv
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_client'))
from auto_grasp import Top1Session,ContinuousGrasp
from pointcloud import FilterConfig


def fixture_snapshot(fid='stream-1',valid=True):
    raw=np.full((10,10),600,np.uint16)
    f=SimpleNamespace(frame_id=fid,depth_raw=raw,depth_m=raw.astype(np.float32)*.001,depth_scale=.001,
        bgr=np.full((10,10,3),140,np.uint8),K=np.array([[100,0,5],[0,100,5],[0,0,1]],np.float32),
        intrinsics={},color_intrinsics={},unit_rays=None,ready_monotonic=time.perf_counter(),
        camera_timestamp_ms=123.,host_timestamp_s=time.time(),acquisition_ms=1.,alignment_ms=2.)
    mask=np.ones((10,10),np.uint8)
    return dict(frame=f,mask=mask,filtered=mask,config=FilterConfig(voxel_m=0),tracking_rtt_ms=4.,actual_fps=8.,
        response=dict(session='test',generation=1,tracking_valid=valid,grasp_allowed=valid,status='tracking' if valid else 'lost',
            mask_depth_ms=1.,metrics=dict(sam_init_ms=None,vos_ms=3.)))


def fixture_completed(snapshot,selected=None):
    g=np.repeat(np.eye(4,dtype=np.float32)[None],2,axis=0)
    g[0,:3,:3]=np.diag([1,-1,-1]);g[:,2,3]=.5
    if selected is not None:g[1]=selected
    return dict(snapshot=snapshot,requested_at=time.perf_counter()-.01,result=dict(
        frame_id=snapshot['frame'].frame_id,grasps=g,confidences=np.array([.95,.85]),branch_tags=['a','b'],
        candidate_count_generated=7,candidate_count_after_threshold=3,candidate_count_after_filter=2,
        server_timing=dict(infer_ms=5.),grasp_rpc_rtt_ms=6.))


def test_top1_schema_exact_context_join_stability_and_no_filled_past_pose(tmp_path):
    session=Top1Session(tmp_path,dict(num_grasps=7))
    try:
        a=fixture_snapshot();session.offer_frame(a);session.requested(time.perf_counter()-.02,a['frame'].frame_id)
        first=session.record(fixture_completed(a))
        pose=np.eye(4,dtype=np.float32);pose[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];pose[:3,3]=[.01,0,.5]
        b=fixture_snapshot('stream-2');session.offer_frame(b)
        second=session.record(fixture_completed(b,pose))
        session.offer_frame(fixture_snapshot('stream-3',False))
        c=fixture_snapshot('stream-4');session.offer_frame(c)
        third=session.record(fixture_completed(c))
    finally:report=session.close()
    assert first['position_delta_mm'] is None and first['rotation_delta_deg'] is None
    assert second['position_delta_mm']==pytest.approx(10,abs=1e-4)
    assert second['rotation_delta_deg']==pytest.approx(90,abs=1e-4)
    assert third['position_delta_mm'] is None  # Loss breaks continuity.
    assert np.isclose(sum(second[k]**2 for k in ('qx','qy','qz','qw')),1)
    assert second['point_count']==100 and second['valid_depth_pixels']==100
    assert second['candidate_count_generated']==7 and second['candidate_count_after_filter']==2
    assert second['pose_generation_fps']>0 and second['end_to_end_latency_ms']>=0
    assert second['capture_utc'].endswith('+00:00')
    with np.load(session.path/first['file']) as d:
        assert d['pose'].shape==(4,4) and 'grasps' not in d
        assert d['original_candidate_index']==1
    with np.load(session.path/first['frame_file']) as d:
        np.testing.assert_array_equal(d['depth_raw'],a['frame'].depth_raw)
        np.testing.assert_array_equal(d['instance_mask'],a['mask'])
        assert json.loads(str(d['metadata_json']))['frame_id']==first['frame_id']
    rows=list(csv.DictReader((session.path/'frames_with_poses.csv').open()))
    assert len(rows)==4 and sum(r['pose_available']=='True' for r in rows)==3
    lost=next(r for r in rows if r['frame_id']=='stream-3')
    assert lost['S']=='' and lost['qx']==''  # Never carry a previous pose into another frame.
    assert report['poses']==3 and report['candidate_count_unknown_records']==0


class SchedulerWorker:
    def __init__(self,snapshot):
        self.lock=threading.Lock();self.revision=0;self.grasp_pending=None
        self.latest=snapshot;self.busy=False;self.status='tracking';self.calls=0
    def state(self):return self.latest,self.status,self.busy
    def request_grasp(self):self.calls+=1;self.grasp_pending=time.perf_counter()


class SchedulerSession:
    def __init__(self):self.requests=[];self.failures=0;self.events=[]
    def requested(self,when,fid):self.requests.append(fid)
    def event(self,*args,**kwargs):self.events.append((args,kwargs))
    def break_stability(self):pass


def test_scheduler_single_flight_freshness_loss_and_distinct_frames():
    s=SchedulerSession();auto=ContinuousGrasp(s);worker=SchedulerWorker(fixture_snapshot())
    assert auto.tick(worker)
    assert not auto.tick(worker) and worker.calls==1
    worker.grasp_pending=None;worker.busy=True
    assert not auto.tick(worker)
    auto.inflight=False;worker.busy=False
    assert not auto.tick(worker)  # Same frame is never inferred twice.
    worker.latest=fixture_snapshot('stream-2',False)
    assert not auto.tick(worker)
    worker.latest=fixture_snapshot('stream-3');worker.latest['frame'].ready_monotonic-=3
    assert not auto.tick(worker)
    worker.latest=fixture_snapshot('stream-4')
    assert auto.tick(worker) and worker.calls==2
    worker.grasp_pending=None;worker.latest=None;worker.status='Tracking/network error: test'
    auto.tick(worker)
    assert not auto.inflight and s.failures==1
    worker.latest=fixture_snapshot('stream-5');worker.status='tracking';auto.enabled=False
    assert not auto.tick(worker)
