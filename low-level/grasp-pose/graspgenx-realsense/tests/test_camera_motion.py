"""Camera-frame conventions and failure handling; no fabricated inference mode."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_client'))
from camera_motion import compose_camera_pose,transform_points,collect_frames
from view_session import calibrated_frustum
from session_charts import timeline_image


def test_approaching_camera_keeps_the_object_and_grasp_fixed():
    object_world=np.array([[0.,0.,.8],[.05,0,.8]])
    Tc=np.eye(4);Tc[2,3]=.3
    seen_by_camera=transform_points(object_world,np.linalg.inv(Tc))
    np.testing.assert_allclose(seen_by_camera[:,2],.5)
    np.testing.assert_allclose(transform_points(seen_by_camera,Tc),object_world)
    grasp_world=np.eye(4);grasp_world[:3,3]=[.02,0,.63]
    grasp_camera=np.linalg.inv(Tc)@grasp_world
    np.testing.assert_allclose(Tc@grasp_camera,grasp_world)


def test_pose_accumulation_uses_previous_current_not_the_inverse():
    previous=np.eye(4);previous[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];previous[:3,3]=[.1,.2,.3]
    relative=np.eye(4);relative[0,3]=.2
    current=compose_camera_pose(previous,relative)
    np.testing.assert_allclose(current[:3,3],[.1,.4,.3])


def test_frustum_uses_real_principal_point_and_camera_transform():
    K=np.array([[500,0,300],[0,400,200],[0,0,1.]])
    T=np.eye(4);T[:3,3]=[1,2,3]
    lines=calibrated_frustum(T,K,640,480,scale=.1)
    np.testing.assert_allclose(lines[0,0],T[:3,3])
    np.testing.assert_allclose(lines[0,1],T[:3,3]+[-.06,-.05,.1],atol=1e-6)


def test_archive_frame_order_and_anchor_do_not_depend_on_pose_arrival(tmp_path):
    rows=[]
    for n in (10,2,7):
        file=f'{n}.npz';(tmp_path/file).write_bytes(b'fixture')
        rows.append(dict(frame_id=f'camera-stream-{n}',frame_file=file))
    result=collect_frames(tmp_path,[rows[1],rows[0],rows[2]])
    assert [fid for fid,_ in result]==['camera-stream-2','camera-stream-7','camera-stream-10']


def test_timeline_handles_missing_odometry_and_a_single_frame():
    rows=[dict(completed_elapsed_s=1,C=.8,S=.9)]
    image=timeline_image(rows,0,[np.nan])
    assert image.shape==(300,640,3) and image.dtype==np.uint8


def test_failed_camera_alignment_exports_blank_world_pose(tmp_path):
    import csv
    from camera_motion import export_aligned_grasps
    (tmp_path/'camera_motion').mkdir()
    source=[]
    for i in range(2):
        row=dict(frame_id=f'camera-{i}',C=.8,R=.9,P=.9,Y=.9,S=.83,stability_segment='same')
        pose=np.eye(4);pose[2,3]=.6
        row.update({f't{a}{b}':pose[a,b] for a in range(4) for b in range(4)});source.append(row)
    motion=dict(frame_ids=np.array(['camera-0','camera-1']),T_reference_camera=np.array([np.eye(4),np.full((4,4),np.nan)]),
                valid=np.array([True,False]),segments=np.array([0,0]),status=np.array(['reference','rejected']))
    export_aligned_grasps(tmp_path,source,motion)
    rows=list(csv.DictReader((tmp_path/'camera_motion/aligned_grasps.csv').open()))
    assert rows[0]['camera_alignment_valid']=='True' and float(rows[0]['reference_z_m'])==pytest.approx(.6)
    assert rows[1]['camera_alignment_valid']=='False' and rows[1]['reference_z_m']=='' and rows[1]['reference_t00']==''
    assert float(rows[1]['S'])==pytest.approx(.83)
