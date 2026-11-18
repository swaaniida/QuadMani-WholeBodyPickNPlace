import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'local_client'))
from pointcloud import FilterConfig, segment, deproject, downsample
from grasp_client import validate_result, GraspClient
from visualization import project_points, overlay


def test_depth_mask_rejects_background_invalid_and_other_islands():
    depth = np.full((30, 40), 1.5, dtype=np.float32)
    depth[8:22, 12:28] = .6
    depth[0, 0] = 0
    depth[1, 1] = np.nan
    depth[3:5, 3:5] = .61
    mask = segment(depth, (0, 0, 40, 30), FilterConfig())
    assert mask.sum() == 14*16
    assert np.all(depth[mask] == np.float32(.6))


def test_metric_camera_deprojection_and_reprojection():
    K = np.array([[200, 0, 2], [0, 150, 1], [0, 0, 1]], dtype=np.float32)
    depth = np.full((3, 5), 2, np.float32)
    mask = np.ones_like(depth, dtype=bool)
    pc, v, u = deproject(depth, K, mask)
    assert pc.dtype == np.float32 and pc.shape == (15, 3)
    np.testing.assert_allclose(pc[7], [0, 0, 2])
    np.testing.assert_allclose(pc[0], [-.02, -2/150, 2])
    np.testing.assert_array_equal(project_points(pc, K), np.column_stack((u, v)))


def test_calibrated_rays_used_without_rotating_or_recentering():
    rays = np.array([[[.2, -.1, 1.]]], np.float32)
    pc, _, _ = deproject(np.array([[.7]],np.float32), np.eye(3), np.array([[True]]), rays)
    np.testing.assert_allclose(pc, [[.14,-.07,.7]], atol=1e-7)


def test_downsample_keeps_metric_values_and_color_correspondence():
    pc = np.array([[.001, .001, .6], [.002, .001, .6], [.04,.01,.6]],np.float32)
    colors = np.array([[255,0,0],[0,255,0],[0,0,255]],np.uint8)
    p,c = downsample(pc, colors, .01, 100)
    assert len(p) == 2
    for point, color in zip(p,c):
        idx = np.flatnonzero(np.all(pc==point,axis=1))[0]
        np.testing.assert_array_equal(colors[idx],color)


def test_empty_and_bad_roi_fail():
    with pytest.raises(ValueError):
        segment(np.zeros((20,20)),(0,0,20,20),FilterConfig())
    with pytest.raises(ValueError):
        segment(np.ones((20,20)),(10,10,20,20),FilterConfig())


def test_reject_invalid_se3_and_accept_empty():
    validate_result(np.zeros((0,4,4)),np.zeros(0))
    poses = np.eye(4)[None].copy()
    poses[0,0,0]=2
    with pytest.raises(ValueError):
        validate_result(poses,np.array([.5]))
    with pytest.raises(ValueError):
        validate_result(np.eye(4)[None],np.array([np.nan]))


def test_overlay_projected_axes_and_no_camera_import():
    bgr=np.zeros((480,640,3),np.uint8)
    K=np.array([[500,0,320],[0,500,240],[0,0,1]])
    T=np.eye(4)[None]
    T[0,2,3]=.5
    image=overlay(bgr,K,T,np.array([.9]),'GEOMETRY_TEST')
    assert image[240,350,2]>0  # +X red projects to the right
    assert image[270,320,1]>0  # +Y green projects downward
    assert not bgr.any()


def test_missing_server_times_out_and_resets_req_socket():
    import socket
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        port=s.getsockname()[1]
    client=GraspClient(port=port,timeout_ms=30)
    for _ in range(2):
        with pytest.raises(TimeoutError):
            client.health()
        assert client.socket is None
