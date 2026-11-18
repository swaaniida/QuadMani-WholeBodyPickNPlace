"""Analytic fixtures only: no mock inference is exposed to production clients."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_client'))
from temporal_selection import TemporalConfig,select_candidates,collision_clearance,pose_change


def poses(z):
    g=np.repeat(np.eye(4)[None],len(z),axis=0);g[:,2,3]=z;return g


def test_camera_motion_is_compensated_at_capture_not_result_arrival():
    g=poses([.7,.72]);before=g.copy();c=np.array([.8,.85]);points=np.array([[0,0,.8]])
    T=np.eye(4);T[2,3]=.1;prev=poses([.8])[0]
    result=select_candidates(g,c,points,np.array([[-.01,0,-.01],[.01,0,.01]]),[0,0,-1,1],TemporalConfig(),prev,T,True)
    assert result['selected']==1
    np.testing.assert_allclose(result['temporal'],[1.,1.])
    assert result['baseline_selected']==1
    np.testing.assert_array_equal(g,before)
    mm,angle=pose_change(prev,T@g[0]);assert mm<1e-8 and angle<1e-8


def test_low_registration_confidence_uses_only_confidence_and_unknown_ground_fails_closed():
    g=poses([.7,.72]);points=np.array([[0,0,.8]]);prev=poses([.8])[0]
    r=select_candidates(g,[.7,.9],points,[[0,0,0]],[0,0,-1,1],TemporalConfig(),prev,np.eye(4),False)
    np.testing.assert_allclose(r['scores'],[.7,.9]);assert not r['temporal_used'] and r['selected']==1
    r=select_candidates(g,[.7,.9],points,[[0,0,0]],None,TemporalConfig())
    assert r['selected'] is None and r['status']=='no_ground_plane'


def test_mesh_rotation_and_margin_reject_colliding_candidates_without_changing_pose():
    g=poses([.7,.99]);vertices=np.array([[0,0,.02],[.04,0,.01]])
    r=select_candidates(g,[.5,.99],[[0,0,.8]],vertices,[0,0,-1,1],TemporalConfig())
    assert list(r['valid'])==[True,False] and r['selected']==0
    assert collision_clearance(g,vertices,[0,0,-2,2])[1]==pytest.approx(-.01)
    r=select_candidates(g[1:],[.99],[[0,0,.8]],vertices,[0,0,-1,1],TemporalConfig())
    assert r['selected'] is None and r['status']=='no_valid_grasp'
    rotated=g[:1].copy();rotated[0,:3,:3]=np.diag([1,-1,-1])
    assert collision_clearance(rotated,vertices,[0,0,-1,1])[0]==pytest.approx(.31)


def test_configuration_and_confidence_validation():
    with pytest.raises(ValueError):TemporalConfig(position_sigma_m=0)
    with pytest.raises(ValueError):TemporalConfig(keyframe_interval=0)
    with pytest.raises(ValueError):TemporalConfig(ground_up_camera=[0,0,0])
    with pytest.raises(ValueError):select_candidates(poses([.7]),[1.2],[[0,0,1]],[[0,0,0]],None,TemporalConfig())


def test_background_excludes_dilated_target_and_invalid_depth():
    from background_icp import background_cloud
    from types import SimpleNamespace
    mask=np.zeros((30,30),np.uint8);mask[10:20,10:20]=1
    depth=np.ones((30,30),np.float32);depth[0,:]=np.nan
    f=SimpleNamespace(depth_m=depth,K=np.array([[100,0,15],[0,100,15],[0,0,1.]]),unit_rays=None)
    pc=background_cloud(dict(frame=f,mask=mask),TemporalConfig(mask_dilation_pixels=3))
    assert len(pc)==900-30-144
    assert np.isfinite(pc).all()


def test_ransac_support_plane_and_actual_open_cad_halfspace():
    o3d=pytest.importorskip('open3d')
    from background_icp import estimate_support_plane
    import trimesh
    rng=np.random.default_rng(5);xy=rng.uniform(-.4,.4,(4000,2))
    table=np.column_stack((xy,np.ones(4000)))
    obj=rng.uniform([-.04,-.04,.8],[.04,.04,.9],(400,3))
    plane,metrics=estimate_support_plane(table,obj,TemporalConfig(),o3d)
    assert plane is not None and metrics['rmse_m']<1e-5
    vertices=trimesh.load(Path(__file__).resolve().parents[1]/'local_client/assets/franka_umi_vis_mesh.obj',force='mesh',process=False).vertices
    clearance=collision_clearance(poses([.7,.9]),vertices,plane)
    assert clearance[0]>0 and clearance[1]<0


def test_point_to_plane_transform_convention_and_observability():
    o3d=pytest.importorskip('open3d')
    from background_icp import BackgroundICP
    from camera_motion import transform_points
    rng=np.random.default_rng(9);xy=rng.uniform(-.4,.4,(1500,2))
    points=np.vstack((np.column_stack((xy,np.ones(1500))),np.column_stack((np.full(1500,-.4),xy[:,0],xy[:,1]+1)),np.column_stack((xy[:,0],np.full(1500,-.4),xy[:,1]+1))))
    engine=object.__new__(BackgroundICP);engine.o3d=o3d;engine.cfg=TemporalConfig()
    target=engine._cloud(points);T=np.eye(4);T[:3,3]=[.01,.005,-.008]
    source=engine._cloud(transform_points(points,np.linalg.inv(T)))
    estimated,q,ok=engine._register(source,target,np.eye(4))
    assert ok and q['fitness']>.95
    np.testing.assert_allclose(estimated,T,atol=.004)
    flat=engine._cloud(points[:1500]);_,_,ok=engine._register(flat,flat,np.eye(4))
    assert not ok  # A single plane cannot constrain full 6-DoF camera motion.


def test_capture_clock_prefers_sensor_and_uses_one_clock_domain():
    from types import SimpleNamespace
    from temporal_selection import capture_time
    frame=SimpleNamespace(camera_timestamp_ms=2000.,host_timestamp_s=999.,ready_monotonic=10.)
    assert capture_time(frame)==('camera_timestamp_ms',2.)
    frame.camera_timestamp_ms=None
    assert capture_time(frame)==('host_timestamp_s',999.)
    frame.host_timestamp_s=None
    assert capture_time(frame)==('ready_monotonic',10.)


def test_clearance_passes_only_strictly_above_plane_rmse_even_with_legacy_margin():
    g=poses([.002,.002001,.001999,-.001])
    result=select_candidates(g,[.9,.8,.7,.6],[[0,0,1]],[[0,0,0]],[0,0,1,0],
        TemporalConfig(collision_margin_m=.003),ground_uncertainty_m=.002)
    np.testing.assert_array_equal(result['valid'],[False,True,False,False])
    assert result['selected']==1


def test_rotation_only_temporal_ignores_position_but_keeps_180_degrees_distinct():
    g=poses([.2,.8,.5]);g[2,:3,:3]=np.diag([-1.,-1.,1.])
    result=select_candidates(g,[.8,.8,.8],[[0,0,.8]],[[0,0,0]],[0,0,-1,1],
        TemporalConfig(),previous_world=poses([.8])[0],T_world_camera=np.eye(4),icp_reliable=True)
    np.testing.assert_allclose(result['temporal'][:2],[1.,1.])
    assert result['temporal'][2]==pytest.approx(np.exp(-.5*(180/20)**2),rel=1e-8,abs=0.)
    assert result['temporal'][2]<1e-17


def test_rotation_only_temporal_compensates_camera_rotation():
    T=np.eye(4);T[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    previous=poses([.8])[0];candidate=(np.linalg.inv(T)@previous)[None]
    result=select_candidates(candidate,[.8],[[0,0,.8]],[[0,0,0]],[0,0,-1,1],TemporalConfig(),
        previous_world=previous,T_world_camera=T,icp_reliable=True)
    assert result['temporal'][0]==pytest.approx(1.)


def test_long_capture_gap_does_not_disable_temporal_continuity():
    from temporal_selection import same_capture_sequence
    assert same_capture_sequence('sensor',('session',1),102.9,'sensor',('session',1),100.)
    assert same_capture_sequence('sensor',('session',1),160.,'sensor',('session',1),100.)
    assert not same_capture_sequence('sensor',('session',2),160.,'sensor',('session',1),100.)
    assert not same_capture_sequence('host',('session',1),160.,'sensor',('session',1),100.)
    assert not same_capture_sequence('sensor',('session',1),99.,'sensor',('session',1),100.)
