"""Offline integration of real cached candidates, recording and comparison renderer.
No live inference or synthetic production poses. Timing here is REPLAY, not live FPS.
"""
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_client'))
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_temporal_replay import snapshot
from temporal_session import TemporalSession
from view_temporal import ComparisonViewer


def main():
    source=next((ROOT/'local_client/example_auto_session/frames').glob('*.npz'))
    with np.load(ROOT/'outputs/tracking/grasps/integration_static_rgbd.npz',allow_pickle=False) as d:
        raw={key:d[key].copy() for key in ('grasps','confidences','branch_tags')}
    original=raw['grasps'].copy()
    cached=json.loads((ROOT/'outputs/tracking/grasps/integration_static_rgbd.json').read_text())
    session=TemporalSession(ROOT/'outputs/temporal_selection',dict(num_grasps=200,source='OFFLINE cached actual models; no hardware/live inference',windows_hardware_tested=False))
    try:
        for i in range(2):
            snap=snapshot(source,i);session.offer_frame(snap)
            registered=session.icp.lookup(snap['frame'].frame_id,5.)
            assert registered and registered['valid'],registered
            now=time.perf_counter();session.requested(now,snap['frame'].frame_id)
            result=dict(raw,frame_id=snap['frame'].frame_id,server_timing=cached['server_timing'],grasp_rpc_rtt_ms=0.,
                candidate_count_generated=len(raw['grasps']),candidate_count_after_threshold=len(raw['grasps']),candidate_count_after_filter=len(raw['grasps']))
            row=session.record(dict(snapshot=snap,result=result,requested_at=now))
            assert row is not None,session.comparisons[-1]
            assert row['S']==session.comparisons[-1]['final_score']
        assert not session.comparisons[0]['temporal_used']
        assert session.comparisons[1]['temporal_used']
        np.testing.assert_array_equal(raw['grasps'],original)
    finally:session.close()
    viewer=ComparisonViewer(session.path,port=18181)
    try:
        assert viewer.motion_ready.wait(40),'Offline replay reconstruction timed out'
        assert viewer.motion_error is None,viewer.motion_error
        assert viewer.motion is not None
        for i in range(len(viewer.rows)):viewer.show(i)
        pivot=viewer.pivot.copy()
        viewer.show(0);viewer.show(len(viewer.rows)-1)
        np.testing.assert_array_equal(viewer.pivot,pivot)
        viewer.set_view('Side');np.testing.assert_array_equal(viewer.pivot,pivot)
        viewer.set_pivot(pivot+[.01,0,0]);viewer.show(0)
        np.testing.assert_allclose(viewer.pivot,pivot+[.01,0,0])
        assert viewer.viewer.mesh is not None
    finally:viewer.close()
    with np.load(session.path/'samples/000001.npz',allow_pickle=False) as d:
        for key in d.files:assert d[key].dtype.kind!='O',(key,d[key].dtype)
    report=dict(session=str(session.path),poses=len(session.rows),same_raw_candidates_unchanged=True,
        first_confidence_only=True,second_temporal_used=True,comparison_renderer='both frames API/HTTP scene created',
        limitations='Two repeats of cached real static RGB-D and real cached model results; no Windows hardware test or fresh model inference; replay throughput is not live pose FPS')
    (ROOT/'outputs/temporal_selection/recording_smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
