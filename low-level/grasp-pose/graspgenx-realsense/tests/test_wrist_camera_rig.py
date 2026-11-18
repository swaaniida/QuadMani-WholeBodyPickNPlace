"""CAD geometry and transform checks; no robot/camera/model requests."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_client'))
from rig_geometry import align_pair,point_transform
from export_realsense_calibration import extrinsic_matrix


def test_pair_alignment_uses_normal_to_resolve_rotation():
    source=np.array([[-.0225,0,-.02505],[.0225,0,-.02505]])
    target=np.array([[-.0225,-.02,.07],[.0225,-.02,.07]])
    normal=np.array([0,-np.sqrt(3)/2,-.5])
    T,error=align_pair(source,target,[0,0,1],normal)
    np.testing.assert_allclose(point_transform(source,T),target,atol=1e-12)
    np.testing.assert_allclose(T[:3,2],normal,atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(T[:3,:3]),1,atol=1e-12)


def test_generated_rig_has_single_root_and_all_meshes():
    path=ROOT/'local_client/assets/wrist_camera_rig/piper_umi_d435i.urdf'
    tree=ET.parse(path).getroot();links=[x.get('name') for x in tree.findall('link')]
    assert len(links)==len(set(links))
    children=[]
    for joint in tree.findall('joint'):
        assert joint.get('type')=='fixed'
        assert joint.find('parent').get('link') in links
        children.append(joint.find('child').get('link'))
    assert len(children)==len(set(children))
    assert set(links)-set(children)=={'piper_link6'}
    for node in tree.iter('mesh'):assert (path.parent/node.get('filename')).is_file()


def test_real_cad_registration_and_tcp_inverse():
    d=json.loads((ROOT/'local_client/assets/wrist_camera_rig/rig.json').read_text())
    assert max(d['camera_hole_center_residual_mm'])<.01
    assert max(d['wrist_axis_coaxial_residual_mm'])<.01
    assert d['mount_seat_evidence_vertices']>100
    np.testing.assert_allclose(np.array(d['T_color_tcp'])@d['T_tcp_color'],np.eye(4),atol=1e-10)
    np.testing.assert_allclose(np.array(d['T_flange_color'])@d['T_color_tcp'],d['T_flange_tcp'],atol=1e-10)
    assert d['tcp_frame']=='piper_umi_nominal_tcp'


def test_sdk_column_major_extrinsics_and_direction():
    from types import SimpleNamespace
    R=np.array([[0,-1.,0],[1.,0,0],[0,0,1.]])
    t=np.array([.015,.0001,-.0003]);e=SimpleNamespace(rotation=R.flatten(order='F').tolist(),translation=t.tolist())
    rs=SimpleNamespace(rs2_transform_point_to_point=lambda e,p:R@p+t)
    T=extrinsic_matrix(rs,e)
    np.testing.assert_allclose(T[:3,:3],R);np.testing.assert_allclose(T[:3,3],t)


def test_calibrated_urdf_keeps_nominal_and_replaces_only_device_color_transform(tmp_path):
    import shutil
    from mounted_wrist_rig import calibrated_urdf
    from trimesh.transformations import euler_matrix
    path=ROOT/'local_client/assets/wrist_camera_rig/piper_umi_d435i.urdf'
    original=path.read_bytes();shutil.copy2(path,tmp_path/path.name)
    T=euler_matrix(.001,-.002,.003);T[:3,3]=[.0149,.0002,-.0001]
    target=calibrated_urdf(tmp_path/'rig.json',dict(T_color_optical_depth_optical=T.tolist()))
    assert path.read_bytes()==original and target.is_file()
    root=ET.parse(target).getroot();poses={'camera_link':np.eye(4)}
    for _ in range(10):
        for joint in root.findall('joint'):
            parent=joint.find('parent').get('link');child=joint.find('child').get('link')
            if parent not in poses:continue
            o=joint.find('origin');Q=np.eye(4)
            if o is not None:
                Q=euler_matrix(*map(float,o.get('rpy','0 0 0').split()));Q[:3,3]=list(map(float,o.get('xyz','0 0 0').split()))
            poses[child]=poses[parent]@Q
    actual=np.linalg.inv(poses['camera_color_optical_frame'])@poses['camera_depth_optical_frame']
    np.testing.assert_allclose(actual,T,atol=1e-10)


def test_upper_wrist_mount_keeps_camera_attachment():
    d=json.loads((ROOT/'local_client/assets/wrist_camera_rig/rig.json').read_text())
    old=json.loads((ROOT/'tests/fixtures/lower_mount_reference.json').read_text())
    assert d['clocking_deg']==180
    np.testing.assert_allclose(d['T_mount_camera_raw_mesh'],old['T_mount_camera_raw_mesh'],atol=1e-12)
    np.testing.assert_allclose(d['T_flange_mount'],np.diag([-1.,-1.,1.,1.])@old['T_flange_mount'],atol=1e-12)
    # Both selected screw axes now lie on the flange -X (upper) side.
    assert all(b['axis'][0]<0 for b in d['wrist_radial_bores'])
