"""Render the actual nominal PiPER/UMI/camera rig at a color optical pose."""
import json
from pathlib import Path
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from trimesh.transformations import euler_matrix,euler_from_matrix
from official_viser_helpers import matrix_to_wxyz_position

DEFAULT_RIG=Path(__file__).parent/'assets/wrist_camera_rig/rig.json'


def calibrated_urdf(rig_path,calibration):
    """Write a new URDF with device depth/color extrinsics; keep original intact."""
    path=Path(rig_path).parent/'piper_umi_d435i.urdf';root=ET.parse(path).getroot()
    def transform(node):
        o=node.find('origin');T=euler_matrix(*map(float,o.get('rpy','0 0 0').split())) if o is not None else np.eye(4)
        if o is not None:T[:3,3]=list(map(float,o.get('xyz','0 0 0').split()))
        return T
    poses={'camera_link':np.eye(4)}
    for _ in range(10):
        for j in root.findall('joint'):
            p=j.find('parent').get('link');c=j.find('child').get('link')
            if p in poses:poses[c]=poses[p]@transform(j)
    optical_joint=root.find("joint[@name='camera_color_optical_joint']")
    joint=root.find("joint[@name='camera_color_joint']")
    color_depth=np.asarray(calibration['T_color_optical_depth_optical'],float)
    T=poses['camera_depth_optical_frame']@np.linalg.inv(color_depth)@np.linalg.inv(transform(optical_joint))
    o=joint.find('origin');o.set('xyz',' '.join(f'{v:.12g}' for v in T[:3,3]));o.set('rpy',' '.join(f'{v:.12g}' for v in euler_from_matrix(T)))
    target=path.with_name(path.stem+'_device_calibrated.urdf')
    ET.indent(root);ET.ElementTree(root).write(target,encoding='utf-8',xml_declaration=True)
    return target


class MountedWristRig:
    def __init__(self,path=DEFAULT_RIG,calibration=None):
        self.path=Path(path);self.data=json.loads(self.path.read_text());self.visuals=[]
        self.calibration=json.loads(Path(calibration).read_text()) if calibration else None
        if self.calibration:
            self.color_depth=np.asarray(self.calibration['T_color_optical_depth_optical'],float)
            self.T_camera_tcp=self.color_depth@np.asarray(self.data['T_depth_tcp'])
        else:
            self.color_depth=None;self.T_camera_tcp=np.asarray(self.data['T_color_tcp'])
        for item in self.data['visuals']:
            mesh=trimesh.load(self.path.parent/item['mesh'],force='mesh',process=False)
            T=(self.color_depth@np.asarray(item['T_depth_visual']) if self.calibration else np.asarray(item['T_color_visual']))
            self.visuals.append((item,mesh,T))

    def draw(self,server,T_reference_color,show_mesh=True,show_tcp=True):
        scene=server.scene
        if show_mesh:
            for i,(item,mesh,T) in enumerate(self.visuals):
                q,p=matrix_to_wxyz_position(np.asarray(T_reference_color)@T)
                scene.add_mesh_simple(f'/mounted_rig/{i}_{item["link"]}',vertices=mesh.vertices,faces=mesh.faces,
                    wxyz=q,position=p,color=tuple(item['color']),opacity=.85,side='double')
        if show_tcp:
            q,p=matrix_to_wxyz_position(np.asarray(T_reference_color)@self.T_camera_tcp)
            scene.add_frame('/mounted_rig/TCP',wxyz=q,position=p,axes_length=.06,axes_radius=.002)
            scene.add_label('/mounted_rig/TCP/label','Actual UMI V7 nominal TCP (OPEN CAD)',position=p)
        return np.asarray(T_reference_color)@self.T_camera_tcp

