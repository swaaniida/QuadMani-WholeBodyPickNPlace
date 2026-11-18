"""Recorded-time playback, endpoint visibility and fixed-cloud reconstruction."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_client'))
from view_session import sample_times,sample_at_time,advance_playback,playback_style,fuse_clouds


def test_capture_timing_preferred_to_variable_inference_latency():
    rows=[dict(host_timestamp_s=100.,completed_elapsed_s=3.),dict(host_timestamp_s=100.4,completed_elapsed_s=8.)]
    np.testing.assert_allclose(sample_times(rows),[0.,.4])
    rows=[dict(host_timestamp_s='',camera_timestamp_ms=2500),dict(host_timestamp_s='',camera_timestamp_ms=2800)]
    np.testing.assert_allclose(sample_times(rows),[0.,.3])


def test_seek_holds_recorded_pose_and_clamps_at_endpoints():
    times=np.array([0.,.4,1.])
    assert [sample_at_time(times,t) for t in [-1,0,.39,.4,.9,1,100]]==[0,0,0,1,1,2,2]


def test_last_pose_is_visible_before_loop_wrap_and_speed_scales_time():
    cursor,playing=advance_playback(.9,.2,1.,1.,True)
    assert cursor==pytest.approx(1.1) and playing
    assert sample_at_time(np.array([0.,1.]),cursor)==1
    assert advance_playback(cursor,.3,2.,1.,True)==pytest.approx((.2,True))
    assert advance_playback(.9,.2,1.,1.,False)==(1.,False)
    assert advance_playback(0.,10.,1.,0.,True)==(0.,False)


def test_fade_spans_transparent_to_opaque_with_large_color_change():
    styles=[playback_style(i,10) for i in range(10)]
    assert styles[0][0]==.25 and styles[-1][0]==1.
    assert np.all(np.diff([s[0] for s in styles])>0)
    assert np.linalg.norm(styles[0][1].astype(float)-styles[-1][1])>250
    assert playback_style(0,1)[0]==1.


def test_fusion_averages_aligned_observations_in_metric_voxels():
    clouds=[np.array([[.0005,0,.8],[.02,0,.8]]),np.array([[.0015,0,.8],[np.nan,0,0]])]
    colors=[np.array([[100,50,0],[0,255,0]],np.uint8),np.array([[200,100,0],[0,0,0]],np.uint8)]
    points,rgb=fuse_clouds(clouds,colors)
    np.testing.assert_allclose(points,[[.001,0,.8],[.02,0,.8]],atol=1e-7)
    np.testing.assert_array_equal(rgb,[[150,75,0],[0,255,0]])


def test_loop_boundary_uses_typical_recorded_pose_delay_and_scales_with_speed():
    from view_session import loop_hold
    hold=loop_hold(np.array([0.,.8,1.8,3.]))
    assert hold==pytest.approx(1.)
    assert advance_playback(3.,.9,1.,3.,True,hold)==pytest.approx((3.9,True))
    assert advance_playback(3.,.55,2.,3.,True,hold)==pytest.approx((.1,True))
