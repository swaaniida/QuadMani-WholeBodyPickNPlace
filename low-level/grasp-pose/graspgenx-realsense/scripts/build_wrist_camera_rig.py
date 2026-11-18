"""Build a visualization-only wrist/mount/D435i/UMI assembly from measured CAD.

Never edits source meshes, existing robot URDFs, inference models or services.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from trimesh.transformations import euler_matrix,euler_from_matrix

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_client'));sys.path.insert(0,str(ROOT/'.deps/rig_tools'))
from rig_geometry import planar_circles,align_pair,point_transform


def origin(element):
    node=element.find('origin');T=np.eye(4)
    if node is not None:
        T=euler_matrix(*map(float,node.get('rpy','0 0 0').split()))
        T[:3,3]=list(map(float,node.get('xyz','0 0 0').split()))
    return T


def set_origin(element,T):
    node=element.find('origin')
    if node is None:node=ET.SubElement(element,'origin')
    node.set('xyz',' '.join(f'{x:.12g}' for x in T[:3,3]))
    node.set('rpy',' '.join(f'{x:.12g}' for x in euler_from_matrix(T)))


def forward(root,base):
    poses={base:np.eye(4)}
    for _ in range(len(root.findall('joint'))+1):
        for joint in root.findall('joint'):
            parent=joint.find('parent').get('link');child=joint.find('child').get('link')
            if parent in poses:poses[child]=poses[parent]@origin(joint)
    return poses


def fit_radial_bore(mesh,azimuth_deg):
    """Recover flange radial cylinder axis from its actual faceted bore surface."""
    a=np.array([np.cos(np.deg2rad(azimuth_deg)),np.sin(np.deg2rad(azimuth_deg)),0])
    b=np.cross([0,0,1],a);n=mesh.face_normals;c=mesh.triangles_center
    ids=(np.abs(n@a)<1e-4)&(c@a>20)&(np.abs(c@b)<3)&(np.abs(n[:,2])>.05)&(np.abs(n[:,2])<.95)
    xyz=mesh.vertices[np.unique(mesh.faces[ids])]
    if len(xyz)<20:raise ValueError('Insufficient radial-bore surface evidence')
    uv=np.column_stack((xyz@b,xyz[:,2]));s=np.linalg.lstsq(np.column_stack((2*uv,np.ones(len(uv)))),np.sum(uv**2,axis=1),rcond=None)[0]
    r=np.sqrt(s[2]+s[:2]@s[:2]);error=np.max(np.abs(np.linalg.norm(uv-s[:2],axis=1)-r))
    if error>.01 or abs(r-1.7)>.01:raise ValueError('Unexpected flange radial bore; registration aborted')
    return dict(axis=a,point=b*s[0]+np.array([0,0,s[1]]),radius_mm=float(r),fit_error_mm=float(error))


def expand_camera():
    import xacro
    package=ROOT/'upstream/realsense-ros/realsense2_description'
    temp=ROOT/'outputs/wrist_camera_mount/xacro_source';(temp/'urdf').mkdir(parents=True,exist_ok=True)
    for p in (package/'urdf').glob('*.xacro'):
        (temp/'urdf'/p.name).write_text(p.read_text().replace('$(find realsense2_description)',str(temp)))
    wrapper=temp/'camera.xacro'
    wrapper.write_text(f'<robot name="d435i" xmlns:xacro="http://www.ros.org/wiki/xacro"><link name="rig_anchor"/><xacro:include filename="{temp}/urdf/_d435i.urdf.xacro"/><xacro:sensor_d435i parent="rig_anchor" name="camera" use_nominal_extrinsics="true" add_plug="false"><origin xyz="0 0 0" rpy="0 0 0"/></xacro:sensor_d435i></robot>')
    camera=ET.fromstring(xacro.process_file(str(wrapper)).toxml())
    return camera,package


def visual_lod(mesh,cell=.00035):
    # Display-only reduction. Original CAD is retained for every registration test.
    xyz=np.asarray(mesh.vertices);keys=np.rint(xyz/cell).astype(np.int64)
    _,inverse=np.unique(keys,axis=0,return_inverse=True);n=inverse.max()+1
    v=np.column_stack([np.bincount(inverse,weights=xyz[:,i],minlength=n) for i in range(3)])
    v/=np.bincount(inverse,minlength=n)[:,None];f=inverse[mesh.faces]
    f=f[(f[:,0]!=f[:,1])&(f[:,1]!=f[:,2])&(f[:,0]!=f[:,2])]
    return trimesh.Trimesh(vertices=v,faces=f,process=False)


def build(a):
    source=a.robot_urdf.resolve();robot=ET.parse(source).getroot()
    out=a.output.resolve();meshes=out/'meshes';meshes.mkdir(parents=True,exist_ok=True)
    raw_mount=ROOT/'assets/wrist_camera_mount/RealSense_D435_camera_piper_wrist.stl'
    mount=trimesh.load(raw_mount,force='mesh');circles=planar_circles(mount)
    # Identify the camera contact side by normal; two Ø3.4 holes at 45 mm pitch.
    contact=[r for r in circles if abs(r['radius']-1.7)<.01 and r['center'][2]>60 and r['normal'][1]<-.8]
    contact.sort(key=lambda r:r['center'][0])
    if len(contact)!=2 or abs(np.linalg.norm(np.array(contact[1]['center'])-contact[0]['center'])-45)>.01:
        raise ValueError('Expected camera contact pair not found')
    bottom=[r for r in circles if abs(r['radius']-1.7)<.01 and r['center'][2]<20]
    bottom.sort(key=lambda r:r['center'][0])
    if len(bottom)!=2:raise ValueError('Expected wrist mounting pair not found')
    flange_node=robot.find("link[@name='piper_flange_link']/collision/geometry/mesh")
    flange_path=(source.parent/flange_node.get('filename')).resolve()
    flange=trimesh.load(flange_path,force='mesh');flange.apply_scale(1000)
    # Default +X wrist side: camera width follows -Y, perpendicular to UMI approach.
    angles=[45.,315.];bores=[fit_radial_bore(flange,d) for d in angles]
    n=np.asarray([r['normal'] for r in bottom]);p=np.asarray([r['center'] for r in bottom])
    coefficients=np.linalg.lstsq(np.column_stack((n[0],-n[1])),p[1]-p[0],rcond=None)[0]
    seat_center=(p[0]+coefficients[0]*n[0]+p[1]+coefficients[1]*n[1])/2
    A=np.column_stack((n[0],n[1],np.cross(n[0],n[1])))
    B=np.column_stack((bores[0]['axis'],bores[1]['axis'],np.cross(bores[0]['axis'],bores[1]['axis'])))
    u,_,v=np.linalg.svd(B@A.T);R=u@v
    if np.linalg.det(R)<.99999:raise ValueError('Improper registration rotation')
    host=np.mean([b['point'] for b in bores],axis=0)
    F=np.eye(4);F[:3,:3]=R;F[:3,3]=(host-R@seat_center)/1000
    # Clocking is selectable but not inferred from a symmetric bolt pattern.
    clock=euler_matrix(0,0,np.deg2rad(a.clocking_deg));F=clock@F
    axis_errors=[]
    for point,normal,bore in zip(p,n,bores):
        q=point_transform(point[None]/1000,F)[0];axis=clock[:3,:3]@bore['axis']
        ref=point_transform(bore['point'][None]/1000,clock)[0]
        axis_errors.append(float(np.linalg.norm(np.cross(q-ref,axis))*1000))
    if max(axis_errors)>.01:raise ValueError('Wrist bolt-axis residual exceeds 0.01 mm')
    radial=np.linalg.norm((mount.vertices-seat_center)[:,[0,2]],axis=1)
    seat_vertices=np.abs(radial-28.5)<.02
    if seat_vertices.sum()<100:raise ValueError('Cannot verify 28.5 mm cylindrical seating surface')
    camera,package=expand_camera()
    official_mesh=trimesh.load(package/'meshes/d435.dae',force='mesh')
    rear=planar_circles(trimesh.Trimesh(vertices=official_mesh.vertices*1000,faces=official_mesh.faces),min_radius=1,max_radius=2,min_area=15,max_error=.03)
    rear=[r for r in rear if abs(r['center'][2]+25.05)<.02 and abs(r['radius']-1.25)<.01]
    rear.sort(key=lambda r:r['center'][0])
    if len(rear)!=2:raise ValueError('Official camera rear-hole pair not found')
    targets=contact if a.camera_roll_deg==0 else contact[::-1]
    H,residual=align_pair(np.array([r['center'] for r in rear])/1000,
        np.array([r['center'] for r in targets])/1000,[0,0,1],contact[0]['normal'])
    if max(residual)>.00001:raise ValueError('Camera hole residual exceeds 0.01 mm')
    camera_fk=forward(camera,'rig_anchor');link_mesh=origin(camera.find("link[@name='camera_link']/visual"))
    mount_link=H@np.linalg.inv(link_mesh)
    mount_bottom=mount_link@np.linalg.inv(camera_fk['camera_link'])
    result=ET.Element('robot',{'name':'piper_umi_v7_d435i_wrist_visualization'})
    result.append(ET.Comment('CAD nominal TCP and camera housing extrinsics; not hardware hand-eye calibration. Static OPEN UMI V7.'))
    descendants={'piper_link6'};joints=[]
    for _ in range(8):
        for joint in robot.findall('joint'):
            if joint.find('parent').get('link') in descendants and not joint.find('child').get('link').startswith(('wrist_d435i_', 'piper_wrist_camera_mount', 'camera_')):
                child=joint.find('child').get('link')
                if child not in descendants:descendants.add(child);joints.append(copy.deepcopy(joint))
    for link in robot.findall('link'):
        if link.get('name') in descendants:result.append(copy.deepcopy(link))
    for joint in joints:
        if joint.get('type')!='fixed':raise ValueError('Use the fixed-open V7 source URDF')
        result.append(joint)
    mount_link_node=ET.SubElement(result,'link',{'name':'wrist_camera_mount'})
    for kind in ('visual','collision'):
        node=ET.SubElement(mount_link_node,kind);geometry=ET.SubElement(node,'geometry')
        ET.SubElement(geometry,'mesh',{'filename':'meshes/wrist_mount.stl','scale':'0.001 0.001 0.001'})
    shutil.copy2(raw_mount,meshes/'wrist_mount.stl')
    joint=ET.SubElement(result,'joint',{'name':'wrist_camera_mount_joint','type':'fixed'})
    ET.SubElement(joint,'parent',{'link':'piper_flange_link'});ET.SubElement(joint,'child',{'link':'wrist_camera_mount'});set_origin(joint,F)
    for element in camera:
        if element.tag=='link' and element.get('name')=='rig_anchor':continue
        node=copy.deepcopy(element)
        if node.tag=='joint' and node.get('name')=='camera_joint':
            node.find('parent').set('link','wrist_camera_mount');set_origin(node,mount_bottom)
        result.append(node)
    # Resolve every mesh into the self-contained generated model, without source edits.
    copied={}
    for node in result.iter('mesh'):
        filename=node.get('filename')
        if filename=='meshes/wrist_mount.stl':continue
        path=(package/filename.split('package://realsense2_description/',1)[1]) if filename.startswith('package://realsense2_description/') else (source.parent/filename).resolve()
        key=str(path)
        if key not in copied:
            dest=meshes/(hashlib.sha256(key.encode()).hexdigest()[:8]+'_'+path.stem+'.obj')
            full=trimesh.load(path,force='mesh');full.export(dest);copied[key]=dest
        node.set('filename','meshes/'+copied[key].name)
    poses=forward(result,'piper_link6')
    root_flange=np.linalg.inv(poses['piper_flange_link'])
    poses={name:root_flange@T for name,T in poses.items()}
    color='camera_color_optical_frame';depth='camera_depth_optical_frame';tcp='piper_umi_nominal_tcp'
    Tc=np.linalg.inv(poses[color]);Td=np.linalg.inv(poses[depth])
    visuals=[]
    for link in result.findall('link'):
        name=link.get('name')
        if name not in poses:continue
        for k,node in enumerate(link.findall('visual')):
            mesh_node=node.find('geometry/mesh')
            if mesh_node is None:continue
            path=out/mesh_node.get('filename');full=trimesh.load(path,force='mesh',process=False)
            scale=np.array(list(map(float,mesh_node.get('scale','1 1 1').split())))
            full.vertices*=scale
            lod=visual_lod(full);dest=meshes/f'view_{name}_{k}.obj';lod.export(dest)
            T=poses[name]@origin(node);color_node=node.find('material/color')
            rgba=list(map(float,color_node.get('rgba').split())) if color_node is not None else [.42,.46,.5,1.]
            if name=='wrist_camera_mount':rgba=[.15,.4,.8,1.]
            visuals.append(dict(link=name,mesh='meshes/'+dest.name,T_color_visual=(Tc@T).tolist(),
                T_depth_visual=(Td@T).tolist(),color=[int(x*255) for x in rgba[:3]],display_lod_grid_m=.00035))
    report=dict(status='CAD_nominal_not_hand_eye_calibrated',tcp_frame=tcp,camera_frame=color,
        camera_model='D435i official realsense-ros',camera_commit=json.loads((ROOT/'upstream/realsense-ros/SOURCE.json').read_text())['commit'],
        source_mount=str(raw_mount),source_robot=str(source),mount_units='mm inferred and cross-checked with 45 mm camera pitch / 28.5 mm flange radius',
        clocking_deg=a.clocking_deg,camera_roll_deg=a.camera_roll_deg,
        clocking_note='Upper wrist mounting: default 180 degrees about flange Z, using radial bore pair 225/135 degrees (-X side). Camera-to-mount registration is unchanged.',
        camera_hole_center_residual_mm=(residual*1000).tolist(),wrist_axis_coaxial_residual_mm=axis_errors,
        wrist_radial_bores=[dict(axis=(clock[:3,:3]@b['axis']).tolist(),point_mm=(clock[:3,:3]@b['point']).tolist(),radius_mm=b['radius_mm'],fit_error_mm=b['fit_error_mm']) for b in bores],
        mount_seat_radius_mm=28.5,mount_seat_evidence_vertices=int(seat_vertices.sum()),
        T_flange_mount=F.tolist(),T_mount_camera_raw_mesh=H.tolist(),
        T_color_tcp=(Tc@poses[tcp]).tolist(),T_tcp_color=(np.linalg.inv(poses[tcp])@poses[color]).tolist(),
        T_depth_tcp=(Td@poses[tcp]).tolist(),T_flange_color=poses[color].tolist(),T_flange_depth=poses[depth].tolist(),
        T_flange_tcp=poses[tcp].tolist(),visuals=visuals,
        limitations=['Original UMI V7 visual-to-carriage fit and TCP are inherited nominal values; actual hand-eye/TCP calibration not supplied.',
                      'Camera URDF housing-to-optical extrinsics are nominal; use device depth-to-color extrinsics when available.',
                      'Only static fully-open gripper geometry is rendered; no robot control, validated inertias or physical collision guarantee.',
                      'Flange screw centers on different radial surfaces are coaxial, not forced to coincide along the screw axis. 1.5 mm bracket wall retained.'])
    ET.indent(result);ET.ElementTree(result).write(out/'piper_umi_d435i.urdf',encoding='utf-8',xml_declaration=True)
    (out/'rig.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('status','camera_hole_center_residual_mm','wrist_axis_coaxial_residual_mm','T_color_tcp')},indent=2))
    print('GENERATED',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot-urdf',type=Path,default=Path('/data/team/users/suhwan/vbc-as2piper/low-level-as2piper-hw/resources/robots/as2piper/urdf/as2piper_umi_open_pose_v7.urdf'))
    p.add_argument('--output',type=Path,default=ROOT/'local_client/assets/wrist_camera_rig')
    p.add_argument('--clocking-deg',type=float,choices=(0.,90.,180.,270.),default=180.)
    p.add_argument('--camera-roll-deg',type=int,choices=(0,180),default=0)
    build(p.parse_args())
