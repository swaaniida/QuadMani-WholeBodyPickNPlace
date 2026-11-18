"""Camera frame numbers can skip; only failed registration breaks a replay path."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_client'))
from replay_camera import trajectory_lines


def test_skipped_raw_frame_numbers_do_not_remove_the_path():
    # Valid processed frames may correspond to raw camera frames 30, 33, 37.
    poses=np.repeat(np.eye(4)[None],3,axis=0);poses[:,0,3]=[0,.02,.05]
    lines=trajectory_lines(poses,np.array([True,True,True]),[0,0,0])
    assert len(lines)==2
    np.testing.assert_allclose(lines[:,1,0],[.02,.05])


def test_rejected_registration_is_not_bridged_or_replaced_with_identity():
    poses=np.repeat(np.eye(4)[None],4,axis=0);poses[:,0,3]=[0,.02,.05,.07];poses[1]=np.nan
    lines=trajectory_lines(poses,np.array([True,False,True,True]),[0,0,1,1])
    assert len(lines)==1
    np.testing.assert_allclose(lines[0,:,0],[.05,.07])


def test_reference_camera_is_visible_while_reconstruction_is_pending_or_failed():
    import threading
    from view_temporal import ComparisonViewer
    v=object.__new__(ComparisonViewer)
    v.rows=[dict(frame_id='camera-30')];v.motion_ready=threading.Event();v.motion_error=None;v.motion=None;v.lookup={}
    np.testing.assert_array_equal(v.display_camera('camera-30'),np.eye(4))
    assert v.display_camera('camera-33') is None
    v.motion_error='registration dependency unavailable';v.motion_ready.set()
    np.testing.assert_array_equal(v.display_camera('camera-30'),np.eye(4))
    assert v.display_camera('camera-33') is None


def test_missing_threadpoolctl_does_not_block_reconstruction():
    from unittest.mock import patch
    from view_temporal import replay_thread_limit
    with patch.dict(sys.modules,{'threadpoolctl':None}):
        with replay_thread_limit():pass


def test_initial_orbit_uses_midpoint_only_when_initial_plane_exists():
    from view_temporal import initial_orbit_pivot
    object_points=np.array([[-.02,0,.8],[.02,0,.8]])
    scene_points=np.vstack((object_points,[0.,0.,0.]))
    pivot,up=initial_orbit_pivot(object_points,scene_points,[0,0,-1,1])
    np.testing.assert_allclose(pivot,[0,0,.9])
    np.testing.assert_allclose(up,[0,0,-1])
    pivot,up=initial_orbit_pivot(object_points,scene_points,None)
    np.testing.assert_allclose(pivot,[0,0,.4])
