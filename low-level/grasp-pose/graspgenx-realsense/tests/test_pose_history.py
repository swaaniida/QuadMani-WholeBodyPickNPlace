"""Recorded-pose temporal styling and topology; analytic geometry fixtures only."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_client'))
from pose_history import history_data,direction_arrows,sweep_gripper_mesh


def rows(times,segments=None):
    output=[]
    for i,t in enumerate(times):
        pose=np.eye(4);pose[:3,3]=[i*.01,i*.02,.6]
        row=dict(completed_elapsed_s=t,stability_segment=(segments[i] if segments is not None else 'same'))
        row.update({f't{a}{b}':pose[a,b] for a in range(4) for b in range(4)})
        output.append(row)
    return output


def test_fade_uses_recorded_elapsed_time_and_preserves_metric_poses():
    source=rows([10,11,20]);data=history_data(source)
    np.testing.assert_allclose(data['fraction'],[0,.1,1])
    np.testing.assert_allclose(data['opacity'],[.06,.139,.85])
    assert np.all(np.diff(data['colors'].astype(int),axis=0)<=0)
    np.testing.assert_allclose(data['positions'],[[0,0,.6],[.01,.02,.6],[.02,.04,.6]])
    assert source[0]['completed_elapsed_s']==10


def test_trajectory_never_bridges_reset_or_lost_segments():
    data=history_data(rows([1,2,3,4,5],['a','a','b','b','c']))
    assert data['lines'].shape==(2,2,3)
    np.testing.assert_array_equal(data['lines'][0],data['positions'][[0,1]])
    np.testing.assert_array_equal(data['lines'][1],data['positions'][[2,3]])
    np.testing.assert_array_equal(data['starts'],[0,2,4])
    np.testing.assert_array_equal(data['ends'],[1,3,4])


def test_mesh_limit_never_decimates_the_trajectory():
    data=history_data(rows(range(10)),max_grippers=3)
    assert len(data['gripper_indices'])==3
    assert data['gripper_indices'][0]==0 and data['gripper_indices'][-1]==9
    assert len(data['positions'])==10 and len(data['lines'])==9


def test_arrowheads_follow_forward_direction_and_zero_steps_are_safe():
    lines=np.array([[[0,0,0],[1,0,0]],[[1,0,0],[1,0,0]]],dtype=np.float32)
    colors=np.full((2,2,3),100,np.uint8)
    arrows,arrow_colors=direction_arrows(lines,colors)
    assert arrows.shape==(2,2,3) and arrow_colors.shape==(2,2,3)
    assert np.all(arrows[:,0,0]>arrows[:,1,0])
    np.testing.assert_allclose(arrows[:,0],[[.75,0,0],[.75,0,0]])
    assert np.isfinite(arrows).all()


def test_single_pose_and_equal_timestamps_have_defined_styles():
    data=history_data(rows([2]))
    assert data['opacity'][0]==pytest.approx(.85)
    assert data['lines'].shape==(0,2,3)
    np.testing.assert_array_equal(history_data(rows([2,2]))['fraction'],[0,1])
    arrows,_=direction_arrows(data['lines'],data['line_colors'])
    assert arrows.shape==(0,2,3)


def test_out_of_order_time_is_rejected_not_silently_reordered():
    with pytest.raises(ValueError):history_data(rows([2,1]))


def test_fallback_is_a_finite_sweep_based_mesh():
    mesh=sweep_gripper_mesh(dict(extents_open=[.08,.022,.1],offset_open=[0,0,.14]))
    assert len(mesh.faces)>0 and np.isfinite(mesh.vertices).all()
    assert mesh.bounds[1,2]==pytest.approx(.19)
