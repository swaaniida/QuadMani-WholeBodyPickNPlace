"""Analytic rotations exercise selection policy, not model outputs."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'local_client'))
from grasp_ranking import rank_grasps, rerank_result, display_order, comparison_label

PC = np.array([[0., 0., 1.]])


def poses(*rotations):
    out = np.repeat(np.eye(4)[None], len(rotations), axis=0)
    out[:, :3, :3] = rotations
    out[:, 2, 3] = .8
    return out


def test_camera_aligned_maximum_and_raw_confidence_not_minmax_scaled():
    r = rank_grasps(poses(np.eye(3)), [.8], PC)
    np.testing.assert_allclose(r['components'], [[.8, 1, 1, 1]])
    np.testing.assert_allclose(r['scores'], [.88])


def test_roll_about_camera_z_and_signed_closing_axis():
    rz90 = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    rz180 = np.diag([-1,-1,1])
    r = rank_grasps(poses(rz90, rz180), [1,1], PC)
    np.testing.assert_allclose(r['components'][:, 1:], [[.5,1,1],[0,1,1]])


def test_camera_y_rotation_and_undefined_roll_projection():
    ry90 = np.array([[0,0,1],[0,1,0],[-1,0,0]])
    r = rank_grasps(poses(ry90), [1], PC)
    np.testing.assert_allclose(r['components'][0, 1:], [.5,.5,.5])


def test_backwards_approach_rejected_by_pitch_and_yaw():
    r = rank_grasps(poses(np.diag([1,-1,-1])), [1], PC)
    np.testing.assert_allclose(r['components'][0, 1:], [1,0,0])


def test_pitch_uses_actual_object_ray_not_gripper_base_position():
    ry90 = np.array([[0,0,1],[0,1,0],[-1,0,0]])
    r = rank_grasps(poses(ry90), [.8], np.array([[1.,0.,1.]]))
    assert r['components'][0, 2] == pytest.approx((1+1/np.sqrt(2))/2)


def test_new_best_differs_from_confidence_best_and_does_not_mutate_raw():
    g = poses(np.diag([1,-1,-1]), np.eye(3))
    source = dict(grasps=g.copy(), confidences=np.array([.95,.85]), branch_tags=['a','b'])
    out = rerank_result(source, PC)
    np.testing.assert_array_equal(out['original_candidate_indices'], [1,0])
    np.testing.assert_array_equal(out['confidences'], [.85,.95])
    assert out['branch_tags'] == ['b','a']
    np.testing.assert_array_equal(source['grasps'], g)
    np.testing.assert_array_equal(source['confidences'], [.95,.85])
    ranking = rank_grasps(out['grasps'], out['confidences'], PC)
    assert display_order(ranking, 1) == [0,1]  # C#1 remains visible outside top K
    assert comparison_label(0, ranking) == 'S#1'
    assert comparison_label(1, ranking) == 'C#1'


def test_stable_ties_empty_result_and_same_winner():
    r = rank_grasps(poses(np.eye(3), np.eye(3)), [.8,.8], PC)
    np.testing.assert_array_equal(r['order'], [0,1])
    assert comparison_label(0,r) == 'S#1 = C#1'
    out = rerank_result(dict(grasps=np.empty((0,4,4)), confidences=np.empty(0), branch_tags=[]), PC)
    assert out['selection_metadata']['selected_original_index'] is None
    assert display_order(rank_grasps(out['grasps'],out['confidences'],PC)) == []


def test_bounds_over_real_rotations():
    # Deterministic proper rotations from QR; no fake model inference involved.
    rng = np.random.default_rng(12)
    q, _ = np.linalg.qr(rng.normal(size=(100,3,3)))
    q[:,:,0] *= np.linalg.det(q)[:,None]
    r = rank_grasps(poses(*q), rng.uniform(size=100), PC)
    assert np.all((r['components'] >= 0) & (r['components'] <= 1))
    assert np.all((r['scores'] >= 0) & (r['scores'] <= 1))


@pytest.mark.parametrize('confidence,cloud', [([1.2],PC), ([np.nan],PC), ([.5],[[0,0,0]]), ([.5],[[0,0,-1]])])
def test_invalid_inputs_fail_explicitly(confidence,cloud):
    with pytest.raises(ValueError):
        rank_grasps(poses(np.eye(3)),confidence,cloud)


def test_original_confidence_winner_preserved_when_equal_confidences_reorder():
    source = dict(grasps=poses(np.diag([1,-1,-1]), np.eye(3)),
                  confidences=np.array([.9,.9]), branch_tags=['first','second'])
    out = rerank_result(source, PC)
    r = rank_grasps(out['grasps'],out['confidences'],PC,out['original_candidate_indices'])
    assert r['confidence_best'] == 1
    assert comparison_label(0,r) == 'S#1'
    assert comparison_label(1,r) == 'C#1'
