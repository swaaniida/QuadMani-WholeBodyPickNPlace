#!/usr/bin/env python3
"""Triangle-level mounting audit and identical-camera MuJoCo renders."""
import json
import struct
import re
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from build_mount_scenes import HERE, SOURCE

OUT=HERE/'results/mesh_seated'

def stl_triangles(path):
    d=Path(path).read_bytes();n=struct.unpack_from('<I',d,80)[0]
    if len(d)==84+50*n:
        return np.ndarray(n,dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]),buffer=d,offset=84)['v'].astype(float)
    return np.array([[float(v) for v in x] for x in re.findall(rb'vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)',d)]).reshape(-1,3,3)

def roi_triangles(t,rect):
    x0,x1,y0,y1=rect
    return t[(t[:,:,0].max(1)>=x0)&(t[:,:,0].min(1)<=x1)&(t[:,:,1].max(1)>=y0)&(t[:,:,1].min(1)<=y1)]

def top_at(t,x,y):
    a=t[:,0,:2];b=t[:,1,:2]-a;c=t[:,2,:2]-a;d=np.array([x,y])-a
    det=b[:,0]*c[:,1]-b[:,1]*c[:,0];det=np.where(abs(det)>1e-14,det,np.nan)
    u=(d[:,0]*c[:,1]-d[:,1]*c[:,0])/det;v=(b[:,0]*d[:,1]-b[:,1]*d[:,0])/det
    m=(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)
    z=t[:,0,2]+u*(t[:,1,2]-t[:,0,2])+v*(t[:,2,2]-t[:,0,2])
    return float(z[m].max()) if m.any() else None

def clipped_max(t,rect):
    """Clip each original triangle to XY footprint, including edge intersections."""
    zs=[]
    for tri in roi_triangles(t,rect):
        poly=list(tri)
        for axis,bound,sign in [(0,rect[0],1),(0,rect[1],-1),(1,rect[2],1),(1,rect[3],-1)]:
            new=[]
            for a,b in zip(poly,poly[1:]+poly[:1]):
                ia=sign*(a[axis]-bound)>=0;ib=sign*(b[axis]-bound)>=0
                if ia:new.append(a)
                if ia!=ib:new.append(a+(b-a)*(bound-a[axis])/(b[axis]-a[axis]))
            poly=new
            if not poly:break
        zs.extend(p[2] for p in poly)
    return float(max(zs))

def surface(t,rect,bottom):
    local=roi_triangles(t,rect);tops=[]
    for x in np.linspace(rect[0],rect[1],19):
        for y in np.linspace(rect[2],rect[3],19):
            z=top_at(local,x,y)
            if z is not None:tops.append(z)
    exact=clipped_max(t,rect)
    return dict(footprint_xy_m=rect,bottom_z_m=bottom,exact_top_max_m=exact,
                min_gap_mm=1000*(bottom-exact),max_sampled_gap_mm=1000*(bottom-min(tops)),
                supported_sample_fraction_1mm=float(np.mean(np.abs(np.array(tops)-bottom)<=.001)),
                sampled_rays=len(tops),mesh_penetration_mm=max(0.,1000*(exact-bottom)))

def audit():
    import mujoco
    OUT.mkdir(parents=True,exist_ok=True)
    t=stl_triangles(SOURCE.parent/'../meshes/as2/base_link.STL')
    p=stl_triangles(SOURCE.parent/'../meshes/piper/base_link.stl')
    old=json.loads((HERE/'audit/before/mount_config.json').read_text())
    new=json.loads((HERE/'models/mesh_seated/mount_config.json').read_text())
    source=ET.parse(SOURCE).getroot();report={'status':'MESH_SEATED_CANDIDATE_STRUCTURAL_FASTENING_UNVERIFIED',
      'units':'metres; URDF visual origins zero and scales identity for both bases',
      'original_urdf_mount':source.find("joint[@name='as2_to_piper']/origin").attrib,
      'central_upper_triangle_intersection_z_m':top_at(t,0,0),
      'original_urdf_piper_central_gap_mm':1000*(.15+float(p[:,:,2].min())-top_at(t,0,0)),
      'piper_mesh_bottom_z_m':float(p[:,:,2].min()),'variants':{}}
    dae=ET.parse(SOURCE.parent/'../meshes/piper/dae/base_link.dae').getroot()
    report['piper_dae_unit']=[e.attrib for e in dae.iter() if e.tag.endswith('unit')]
    for variant in ['profiles_plate','plate_only']:
        res={}
        for label,cfg in [('before',old),('after',new)]:
            x=cfg['robot_mount_x_m'];y=cfg['robot_mount_y_m'];z=cfg['estimated_as2_mount_surface_z_m']
            h=cfg['profile_height_m'] if variant=='profiles_plate' else 0
            L=cfg['plate_length_x_m'];W=cfg['plate_width_y_m'];pt=cfg['plate_thickness_m']
            res[label]={'plate_to_as2':surface(t,[x-L/2,x+L/2,y-W/2,y+W/2],z+h),
                        'plate_to_piper_gap_mm':1000*float(p[:,:,2].min()),
                        'piper_base_z_m':z+h+pt}
            if h:
                L=cfg['profile_length_m'];W=cfg['profile_width_m'];cy=cfg['profile_center_y_m']
                res[label]['profiles_to_as2']=[surface(t,[x-L/2,x+L/2,y+s*cy-W/2,y+s*cy+W/2],z) for s in [-1,1]]
                res[label]['profile_top_to_plate_bottom_gap_mm']=0.
        model=mujoco.MjModel.from_xml_path(str(HERE/'models/mesh_seated'/(variant+'.xml')))
        assert (model.nq,model.nv,model.nu)==(27,26,20)
        names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(model.actuator_trnid[a,0])) for a in range(model.nu)]
        assert names[-2:]==['piper_gripper_joint1','piper_gripper_joint2']
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        baseid=model.body('base_link').id;plateid=model.body('mount_plate').id;pid=model.body('piper_base_link').id
        plate_top=float(data.xpos[plateid,2]-data.xpos[baseid,2])+new['plate_thickness_m']/2
        piper_bottom=float(data.xpos[pid,2]-data.xpos[baseid,2])+float(p[:,:,2].min())
        assert abs(piper_bottom-plate_top)<1e-8
        res['compiled_mjcf_plate_to_piper_gap_mm']=1000*(piper_bottom-plate_top)
        res['compiled_mjcf_piper_base_z_m']=float(data.xpos[pid,2]-data.xpos[baseid,2])
        res['hardware_inertias']={}
        for bid in range(1,model.nbody):
            if model.body_mass[bid]>0:
                eig=model.body_inertia[bid]
                assert np.all(eig>0) and 2*max(eig)<=sum(eig)+1e-10
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,bid)
            if name=='mount_plate' or name.startswith('al_profile_'):
                res['hardware_inertias'][name]={'mass_kg':float(model.body_mass[bid]),'principal_inertia_kg_m2':model.body_inertia[bid].tolist()}

        res['model']={'nq':model.nq,'nv':model.nv,'nu':model.nu,'nbody':model.nbody,
                      'total_mass_kg':float(model.body_mass.sum()),'actuated_joint_order':names,
                      'wbc_action_joint_names':[n for n in names if not n.startswith('piper_gripper_')],
                      'qpos_addresses':[int(model.jnt_qposadr[int(model.actuator_trnid[a,0])]) for a in range(model.nu)]}
        assert len(res['model']['wbc_action_joint_names'])==18
        # Every original link and every joint except the replaced mount must be identical.
        robot=ET.parse(SOURCE.parent/('as2piper_umi_v7_mesh_seated_'+variant+'.urdf')).getroot()
        for e in source:
            if e.tag not in ['joint','link'] or e.get('name')=='as2_to_piper':continue
            other=robot.find("%s[@name='%s']"%(e.tag,e.get('name')))
            assert ET.tostring(e)==ET.tostring(other),e.get('name')
        res['original_robot_subtrees_preserved']=True
        supports=res['after'].get('profiles_to_as2',[res['after']['plate_to_as2']])
        for s in supports:assert s['mesh_penetration_mm']<.01 and s['min_gap_mm']<1 and s['supported_sample_fraction_1mm']>.5
        assert res['after']['plate_to_as2']['mesh_penetration_mm']<.01
        report['variants'][variant]=res
    before=json.loads((HERE/'audit/before_sha256.json').read_text())
    report['preserved_source_hashes']={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha for name,sha in before.items()}
    assert all(report['preserved_source_hashes'].values())
    (OUT/'geometry_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report


def renders():
    import mujoco
    from PIL import Image,ImageDraw
    from run_mount_demo import setup
    cfg=json.loads((HERE/'models/mesh_seated/mount_config.json').read_text())
    cameras={'front':(0,-12,2.1,[0,0,.48]),'side':(90,-12,2.1,[0,0,.48]),
             'top':(90,-89,1.7,[0,0,.25]),'mount_close':(115,-25,.65,[-.02,0,.43])}
    for state in ['before','after']:
        for variant in ['profiles_plate','plate_only']:
            path=HERE/'models'/('mesh_seated/'+variant+'.xml' if state=='after' else variant+'.xml')
            m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);setup(m,d,cfg)
            r=mujoco.Renderer(m,height=360,width=640)
            for view,(az,el,dist,look) in cameras.items():
                cam=mujoco.MjvCamera();cam.azimuth=az;cam.elevation=el;cam.distance=dist;cam.lookat[:]=look
                if view=='mount_close':cam.lookat[2]=float(d.xpos[m.body('base_link').id,2])+.085
                r.update_scene(d,camera=cam)
                Image.fromarray(r.render()).save(str(OUT/(state+'_'+variant+'_'+view+'.png')))
            r.close()
    for view in cameras:
        montage=Image.new('RGB',(1280,760),'white');draw=ImageDraw.Draw(montage)
        for row,state in enumerate(['before','after']):
            for col,v in enumerate(['profiles_plate','plate_only']):
                montage.paste(Image.open(OUT/(state+'_'+v+'_'+view+'.png')),(col*640,row*380+20))
                draw.text((col*640+8,row*380+4),state+' / '+v,fill='black')
        montage.save(OUT/('compare_'+view+'.png'))
    print('Rendered before/after, two variants, four identical cameras')

if __name__=='__main__':
    audit();renders()
