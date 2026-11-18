"""Geometry/ranking fixtures; no model or service requests."""
import ast
import pytest
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_client'))
from obb_branch_analysis import ground_basis,generate_obb,analyze


def official_helpers():
    path=ROOT/'upstream/GraspGenX/graspgenx/samplers/graspmoe.py'
    if not path.is_file():pytest.skip('Optional official geometry check requires pinned upstream GraspGenX source')
    source=ast.parse(path.read_text())
    functions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in
               ('_obb_from_angle','_compute_obb','_world_aligned_top_down_grasp')]
    scope={'np':np};exec(compile(ast.Module(body=functions,type_ignores=[]),'<official pure geometry>','exec'),scope)
    return scope['_compute_obb'],scope['_world_aligned_top_down_grasp']


def test_two_obb_frames_have_requested_approach_and_original_sweep_dimensions():
    rng=np.random.default_rng(3);pc=rng.uniform([-.04,-.02,.7],[.04,.02,.8],(300,3))
    plane=np.array([.2,-.8,-.5,1.]);helpers=official_helpers()
    branches,meta=generate_obb(pc,plane,.177,*helpers)
    assert branches['obb_camera'].shape==(50,4,4)
    assert branches['obb_world'].shape==(50,4,4)
    np.testing.assert_allclose(branches['obb_camera'][:,:3,2],np.tile([0,0,1],(50,1)),atol=1e-7)
    n=plane[:3]/np.linalg.norm(plane[:3])
    np.testing.assert_allclose(branches['obb_world'][:,:3,2],np.tile(-n,(50,1)),atol=1e-7)
    for g in branches.values():
        np.testing.assert_allclose(np.linalg.det(g[:,:3,:3]),1,atol=1e-6)
        # Adjacent offsets differ by 2 cm opposite the approach direction.
        np.testing.assert_allclose(g[1,:3,3]-g[0,:3,3],-.02*g[0,:3,2],atol=1e-7)


def test_plane_frame_origin_and_basis():
    p=np.array([.2,-.8,-.5,1.]);T=ground_basis(p)
    np.testing.assert_allclose(p[:3]@T[:3,3]+p[3],0,atol=1e-12)
    np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-12)
    assert np.linalg.det(T[:3,:3])>0


def test_shallow_ranking_and_contact_boundary_are_independent_of_confidence_rank():
    g=np.repeat(np.eye(4)[None],4,axis=0);g[:,2,3]=[.02,.005,-.01,0]
    rows,summary,hi=analyze(g,[.3,.9,.99,.8],['diffusion','obb_camera','obb_world','obb_camera'],
                            [[0,0,0]],[0,0,1,0],.005)
    assert [r['candidate_index'] for r in rows]==[1,3,0,2]
    assert [r['collision_status'] for r in rows]==['contact_like','contact_like','safe','penetrating']
    assert hi==dict(least_penetration=0,highest_confidence=2,shallow_high_confidence=1)
    assert summary[0]['safe']==1 and summary[0]['penetrating']==1
