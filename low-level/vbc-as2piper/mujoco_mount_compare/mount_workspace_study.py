#!/usr/bin/env python3
"""CPU mounting/workspace study; never changes a running training configuration."""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,lsq_linear
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import mujoco
from build_verified_mounts import DirectConverter
from build_mount_scenes import nums,vec,box_inertia
from verify_mounts import stl_triangles,clipped_max

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/mount_workspace'
MODELS=HERE/'models/mount_workspace'
URDFS=HERE.parent/'low-level/resources/robots/as2piper/urdf'
CONFIG=json.loads((HERE/'models/mesh_seated/mount_config.json').read_text())
ARM_C=[0,1.1,-.55,0,0,0]


def origin(e):
    T=np.eye(4)
    if e is not None:
        T[:3,3]=nums(e.get('xyz'),'0 0 0')
        T[:3,:3]=Rotation.from_euler('xyz',nums(e.get('rpy'),'0 0 0')).as_matrix()
    return T


def dae(path):
    r=ET.parse(path).getroot();ns={'c':'http://www.collada.org/2005/11/COLLADASchema'}
    points=[];faces=[]
    for mesh in r.findall('.//c:geometry/c:mesh',ns):
        sources={s.get('id'):np.fromstring(s.find('c:float_array',ns).text,sep=' ').reshape(-1,3)
                 for s in mesh.findall('c:source',ns) if s.find('c:float_array',ns) is not None and 'position' in s.get('id','').lower()}
        vertex_sources={v.get('id'):v.find("c:input[@semantic='POSITION']",ns).get('source')[1:] for v in mesh.findall('c:vertices',ns)}
        for primitive in list(mesh):
            if primitive.tag.split('}')[-1] not in ('triangles','polylist'):continue
            inputs=primitive.findall('c:input',ns);v=next(x for x in inputs if x.get('semantic')=='VERTEX')
            p=sources[vertex_sources[v.get('source')[1:]]];start=len(points);points.extend(p.tolist())
            stride=max(int(x.get('offset','0')) for x in inputs)+1
            indices=np.fromstring(primitive.find('c:p',ns).text,sep=' ',dtype=int).reshape(-1,stride)[:,int(v.get('offset','0'))]
            count=primitive.find('c:vcount',ns)
            counts=[3]*(len(indices)//3) if count is None else np.fromstring(count.text,sep=' ',dtype=int)
            offset=0
            for n in counts:
                polygon=indices[offset:offset+n];offset+=n
                for k in range(1,n-1):faces.append([start+polygon[0],start+polygon[k],start+polygon[k+1]])
    if not points:raise ValueError('No COLLADA position triangles: '+str(path))
    return np.array(points),np.array(faces)


class StudyConverter(DirectConverter):
    def __init__(self,source,out):
        self.source=source;self.output_path=out;self.strict=True
        self.robot=ET.parse(source).getroot();self.links={x.get('name'):x for x in self.robot.findall('link')}
        self.children={};self.joints=self.robot.findall('joint')
        for j in self.joints:self.children.setdefault(j.find('parent').get('link'),[]).append(j)
        self.asset_cache={};self.missing=[];self.skipped=[]
        self.root=ET.Element('mujoco',model='workspace_'+source.stem)
        ET.SubElement(self.root,'compiler',angle='radian',autolimits='true')
        ET.SubElement(self.root,'option',timestep='.002',gravity='0 0 -9.81')
        visual=ET.SubElement(self.root,'visual');ET.SubElement(visual,'global',offwidth='1280',offheight='720')
        self.assets=ET.SubElement(self.root,'asset');self.world=ET.SubElement(self.root,'worldbody')
        ET.SubElement(self.world,'light',pos='0 -2 3',dir='0 .3 -1')
        ET.SubElement(self.world,'geom',name='ground',type='plane',size='3 3 .1',rgba='.4 .45 .45 1')
        self.act=ET.SubElement(self.root,'actuator');self.ns=[]
    def _find_mesh(self,e):
        p=(self.source.parent/e.get('filename')).resolve()
        if p.suffix.lower()=='.dae':
            # PiPER ships matching STL in metres; one DAE declares millimetres.
            # Reuse the official STL mapping from the verified converter.
            official=p.parent.parent/(p.stem+'.stl')
            if official.exists():return official
            dest=MODELS/'meshes'/(p.stem+'.obj')
            if not dest.exists():
                v,f=dae(p);dest.parent.mkdir(exist_ok=True)
                with dest.open('w') as h:
                    for x in v:h.write('v '+vec(x)+'\n')
                    for x in f:h.write('f '+' '.join(str(int(i)+1) for i in x)+'\n')
            return dest
        if not p.exists():raise FileNotFoundError(p)
        return p
    def build(self):
        child={j.find('child').get('link') for j in self.joints}
        name=next(n for n in self.links if n not in child)
        b=self.make_link(name,self.world,dict(pos='0 0 .5',quat='1 0 0 0'))
        ET.SubElement(b,'freejoint',name='base_free')
        ET.ElementTree(self.root).write(str(self.output_path),encoding='utf-8',xml_declaration=True)


def build():
    MODELS.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    # Same plate and same support footprint in both variants. Only PiPER moves.
    v=stl_triangles(URDFS/'../meshes/piper/base_link.stl').reshape(-1,3)
    pmax=float(v[:,0].max());edge=.005
    limit=CONFIG['robot_mount_x_m']+CONFIG['plate_length_x_m']/2-pmax-edge
    cases=[]
    for variant in ('plate_only','profiles_plate'):
        source=URDFS/('as2piper_umi_v7_mesh_seated_'+variant+'.urdf')
        for label,x in [('current',-.02),('forward',limit)]:
            root=ET.parse(source).getroot()
            root.find("joint[@name='plate_to_piper_fixed']/origin").set('xyz',vec([x-CONFIG['robot_mount_x_m'],0,.003]))
            path=URDFS/('as2piper_study_'+variant+'_'+label+'.urdf')
            ET.ElementTree(root).write(str(path),encoding='utf-8',xml_declaration=True)
            output=MODELS/(variant+'_'+label+'.xml');StudyConverter(path,output).build()
            cases.append(dict(name=variant+'_'+label,variant=variant,mount_x_m=x,model=str(output),urdf=str(path)))
    # A distinct redesign proposal: lengthen the plate over existing profiles.
    # The plate is a real cantilever; no fictitious support is added under it.
    body=stl_triangles(URDFS/'../meshes/as2/base_link.STL')
    bottom=CONFIG['estimated_as2_mount_surface_z_m']+.020
    front=.05
    for edge_x in np.arange(.051,.15,.001):
        # Reserve 5 mm to the rising front shell; this is a design assumption,
        # not a measured assembly tolerance or a certified structural margin.
        if clipped_max(body,[-.09,float(edge_x),-.06,.06])>bottom-.005:break
        front=float(edge_x)
    x=front-pmax-edge;length=front+.09;center=(front-.09)/2
    root=ET.parse(URDFS/'as2piper_umi_v7_mesh_seated_profiles_plate.urdf').getroot()
    plate=root.find("link[@name='mount_plate']");mass=length*.12*.006*2700
    plate.find('inertial/mass').set('value',str(mass))
    for tag,value in zip(('ixx','iyy','izz'),box_inertia(mass,[length,.12,.006])):plate.find('inertial/inertia').set(tag,str(value))
    for shape in plate.findall('.//box'):shape.set('size',vec([length,.12,.006]))
    root.find("joint[@name='mount_plate_fixed']/origin").set('xyz',vec([center+.02,0,.023]))
    root.find("joint[@name='plate_to_piper_fixed']/origin").set('xyz',vec([x-center,0,.003]))
    path=URDFS/'as2piper_study_profiles_cantilever.urdf';ET.ElementTree(root).write(str(path),encoding='utf-8',xml_declaration=True)
    output=MODELS/'profiles_cantilever.xml';StudyConverter(path,output).build()
    cases.append(dict(name='profiles_cantilever',variant='profiles_plate',mount_x_m=x,model=str(output),urdf=str(path),
                      redesign=True,plate_length_m=length,cantilever_m=front-.05,
                      plate_as2_clearance_m=bottom-clipped_max(body,[-.09,front,-.06,.06])))
    # B1 original file has malformed numeric suffix; normalize only in a new copy.
    b1root=HERE.parent/'low-level/resources/robots/b1z1/urdf'
    b1=b1root/'b1z1.urdf';copy=b1root/'b1z1_workspace_study.urdf'
    copy.write_text(b1.read_text().replace('0.09>>','0.09'))
    output=MODELS/'b1z1.xml';StudyConverter(copy,output).build()
    cases.append(dict(name='b1z1',variant='b1z1',mount_x_m=.3,model=str(output),urdf=str(copy)))
    scan=[]
    for x in [-.02,0.,limit,.01,.05,.1,.15]:
        front=x+pmax
        scan.append(dict(arm_base_x_m=x,base_front_x_m=front,plate_edge_clearance_m=.05-front,
                         supported_with_5mm_edge=front<=.045+1e-9,
                         comment='Existing plate envelope only; fastening holes not verified'))
    (OUT/'design.json').write_text(json.dumps(dict(cases=cases,scan=scan,forward_limit_m=limit,edge_assumption_m=edge),indent=2))
    return cases


def setup(m,name):
    d=mujoco.MjData(m);d.qpos[:]=m.qpos0;d.qpos[:7]=[0,0,.5,1,0,0,0]
    b1=name=='b1z1'
    for leg in ('FL','FR','RL','RR'):
        for suffix,v in [('hip',(.2 if leg.endswith('L') else -.2) if b1 else 0),('thigh',.8),('calf',-1.5 if b1 else -1.6)]:
            d.qpos[int(m.joint(leg+'_'+suffix+'_joint').qposadr)]=v
    names=['z1_waist','z1_shoulder','z1_elbow','z1_wrist_angle','z1_forearm_roll','z1_wrist_rotate'] if b1 else ['piper_joint'+str(i) for i in range(1,7)]
    q=[0,1.48,-.63,-.84,0,1.57] if b1 else ARM_C
    for n,v in zip(names,q):d.qpos[int(m.joint(n).qposadr)]=v
    if b1:d.qpos[int(m.joint('z1_jointGripper').qposadr)]=-.785
    else:
        d.qpos[int(m.joint('piper_gripper_joint1').qposadr)]=.049
        d.qpos[int(m.joint('piper_gripper_joint2').qposadr)]=-.049
    mujoco.mj_forward(m,d)
    feet=[m.body(leg+'_foot').id for leg in ('FL','FR','RL','RR')]
    radii=[]
    for bid in feet:
        gids=[i for i in range(m.ngeom) if m.geom_bodyid[i]==bid and m.geom_contype[i] and m.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE]
        radii.append(float(m.geom_size[gids[0],0]))
    d.qpos[2]-=min(float(d.xpos[i,2])-r for i,r in zip(feet,radii));mujoco.mj_forward(m,d)
    return d,names,feet,radii


def mass_properties(m,d):
    mass=m.body_mass[1:];pos=d.xipos[1:];com=(mass[:,None]*pos).sum(0)/mass.sum()
    R=d.ximat[1:].reshape(-1,3,3);r=pos-com
    I=np.einsum('nik,nk,njk->nij',R,m.body_inertia[1:],R)+mass[:,None,None]*(np.sum(r*r,axis=1)[:,None,None]*np.eye(3)-r[:,:,None]*r[:,None,:])
    B=d.xmat[1].reshape(3,3)
    return dict(mass_kg=float(mass.sum()),com_world_m=com.tolist(),inertia_diag_kg_m2=np.diag(B.T@I.sum(0)@B).tolist())


def self_collision(m,d):
    ground=m.geom('ground').id
    return any(c.geom1!=ground and c.geom2!=ground and c.dist<-.0005 for c in d.contact)


def margin(m,d,feet):
    p=d.xpos[feet,:2];h=ConvexHull(p);com=np.array(mass_properties(m,d)['com_world_m'])[:2]
    return float(min(-(e[:2]@com+e[2])/np.linalg.norm(e[:2]) for e in h.equations))


def static_loads(m,d,feet,arm_names,efforts):
    props=mass_properties(m,d);com=np.array(props['com_world_m']);weight=props['mass_kg']*9.81
    p=d.xpos[feet];A=np.vstack([np.ones(4),p[:,0],p[:,1]])
    load=lsq_linear(A,weight*np.array([1,com[0],com[1]]),bounds=(0,np.inf))
    tau=d.qfrc_bias.copy()
    for foot,force in zip(feet,load.x):
        jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,jp,jr,foot)
        tau-=jp[2]*force
    arm=[float(tau[int(m.joint(n).dofadr)]) for n in arm_names]
    leg_names=[l+'_'+j+'_joint' for l in ('FL','FR','RL','RR') for j in ('hip','thigh','calf')]
    ratio=max(abs(float(tau[int(m.joint(n).dofadr)]))/efforts[n] for n in leg_names)
    return dict(foot_normal_N=load.x.tolist(),equilibrium_residual=float(np.linalg.norm(A@load.x-weight*np.array([1,com[0],com[1]]))),
                arm_gravity_torque_Nm=arm,max_arm_gravity_torque_Nm=max(abs(v) for v in arm),max_leg_effort_fraction=ratio)


def body_tilt(m,d,angle,standing,feet,foot_targets):
    R=Rotation.from_euler('y',angle).as_matrix();pivot=standing[:3].copy()
    # Rotate around the sagittal rear hip line: rear height stays nearly fixed.
    hip=m.body('RL_hip').id
    d.qpos[:]=standing;mujoco.mj_forward(m,d);local=d.xpos[hip]-standing[:3]
    pivot+=np.array([local[0],0,0]);d.qpos[:3]=pivot-R@np.array([local[0],0,0])
    xyzw=Rotation.from_matrix(R).as_quat();d.qpos[3:7]=xyzw[[3,0,1,2]]
    err=0
    for leg,bid,target in zip(('FL','FR','RL','RR'),feet,foot_targets):
        ids=[int(m.joint(leg+'_'+j+'_joint').qposadr) for j in ('hip','thigh','calf')]
        jids=[m.joint(leg+'_'+j+'_joint').id for j in ('hip','thigh','calf')]
        lo=m.jnt_range[jids,0]+.001;hi=m.jnt_range[jids,1]-.001
        def fun(q):
            d.qpos[ids]=q;mujoco.mj_forward(m,d);return d.xpos[bid]-target
        sol=least_squares(fun,np.clip(standing[ids],lo,hi),bounds=(lo,hi),max_nfev=60)
        err=max(err,float(np.linalg.norm(fun(sol.x))))
    return err


def contact_cloud(m,d,b1,tcp):
    # Actual rendered finger surfaces in the TCP frame (gripper held at open).
    allpoints=[]
    for gid in range(m.ngeom):
        name=m.geom(gid).name
        selected=('gripperMover' in name or 'gripperStator' in name) if b1 else ('piper_gripper_link' in name and name.endswith('_1'))
        if not selected or not name.startswith('v_') or m.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        mid=int(m.geom_dataid[gid]);a=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);v=m.mesh_vert[a:a+n]
        world=v@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
        allpoints.extend(((world-d.xpos[tcp])@d.xmat[tcp].reshape(3,3)).tolist())
    points=np.array(allpoints);h=ConvexHull(points);points=points[h.vertices]
    axis=0 if b1 else 2;tip=points[points[:,axis]>points[:,axis].max()-.004].mean(0)
    return points,tip


def analyze(cases):
    rng=np.random.default_rng(20261028);results=[]
    body_as2=stl_triangles(URDFS/'../meshes/as2/base_link.STL').reshape(-1,3)
    b1verts,_=dae(HERE.parent/'low-level/resources/robots/b1z1/meshes/trunk.dae')
    for case in cases:
        rng=np.random.default_rng(20261028)  # identical normalized joint samples across mount variants
        print('Analyzing',case['name'],flush=True)
        m=mujoco.MjModel.from_xml_path(case['model']);d,names,feet,radii=setup(m,case['name']);standing=d.qpos.copy()
        b1=case['name']=='b1z1';tcp=m.body('ee_gripper_link' if b1 else 'piper_umi_nominal_tcp').id
        arm_base=m.body('link00' if b1 else 'piper_base_link').id
        vertices=b1verts if b1 else body_as2
        nose=vertices[vertices[:,0]>vertices[:,0].max()-.001]
        head=np.array([vertices[:,0].max(),0.,(nose[:,2].min()+nose[:,2].max())/2])
        head_world=head+standing[:3]
        cloud,tip=contact_cloud(m,d,b1,tcp)
        ids=[int(m.joint(n).qposadr) for n in names];jids=[m.joint(n).id for n in names]
        lo=m.jnt_range[jids,0]+.015;hi=m.jnt_range[jids,1]-.015
        efforts={j.get('name'):float(j.find('limit').get('effort')) for j in ET.parse(case['urdf']).getroot().findall('joint') if j.find('limit') is not None}
        props=mass_properties(m,d);props['support_margin_m']=margin(m,d,feet)
        props['static_loads']=static_loads(m,d,feet,names,efforts)
        feet_target=d.xpos[feet].copy();points=[];localpoints=[]
        for k in range(16000):
            d.qpos[:]=standing;d.qpos[ids]=rng.uniform(lo,hi);mujoco.mj_forward(m,d)
            if self_collision(m,d):continue
            p=d.xpos[tcp]
            if p[2]<.02:continue
            points.append((p-head_world).tolist());localpoints.append((d.xmat[arm_base].reshape(3,3).T@(p-d.xpos[arm_base])).tolist())
        points=np.array(points);np.savez_compressed(OUT/(case['name']+'_workspace.npz'),head=points,arm=np.array(localpoints))
        front=points[points[:,0]>0];vox=np.unique(np.floor(front/.03).astype(int),axis=0)
        case.update(head_origin_in_base_m=head.tolist(),standing_qpos=standing.tolist(),standing=props,
                    workspace=dict(accepted=len(points),front_count=len(front),front_x_max_m=float(front[:,0].max()),
                                   voxel_occupancy_m3=float(len(vox)*.03**3),arm_frame='link00' if b1 else 'piper_base_link'),
                    contact_tip_in_tcp_m=tip.tolist())
        trials=[];bestposes={};kinematic={}
        for degrees in np.arange(0,40.1,2.5):
            error=body_tilt(m,d,np.deg2rad(degrees),standing,feet,feet_target)
            tilt_q=d.qpos.copy()
            if error>.003:continue
            for distance in (.1,.2,.3):
                # Same world floor locations measured ahead of the standing nose.
                target=np.array([head_world[0]+distance,0.,0.])
                def fun(q):
                    d.qpos[:]=tilt_q;d.qpos[ids]=q;mujoco.mj_forward(m,d)
                    R=d.xmat[tcp].reshape(3,3);p=d.xpos[tcp]+R@tip
                    axis=R[:,0] if b1 else R[:,2]
                    return np.r_[p-target,.12*(axis-np.array([0,0,-1])),.0005*(q-standing[ids])]
                successful=None
                for seed in (standing[ids],(lo+hi)/2,rng.uniform(lo,hi)):
                    sol=least_squares(fun,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=100,ftol=1e-7)
                    fun(sol.x);R=d.xmat[tcp].reshape(3,3);p=d.xpos[tcp]+R@tip
                    axis=R[:,0] if b1 else R[:,2];worldcloud=cloud@R.T+d.xpos[tcp]
                    # Ground reach is actual finger tip, plus whole-finger clearance.
                    reach=float(np.linalg.norm(p-target));lowest=float(worldcloud[:,2].min())
                    stability=margin(m,d,feet)
                    loads=static_loads(m,d,feet,names,efforts) if reach<.015 and lowest>=-.003 and axis[2]<-.85 and not self_collision(m,d) and stability>.03 else None
                    if loads is not None:
                        kinematic.setdefault(str(distance),dict(angle_deg=float(degrees),
                            tip_error_m=reach,lowest_finger_z_m=lowest,support_margin_m=stability,
                            static_loads=loads,qpos=d.qpos.tolist()))
                    arm_limit_ok=loads is not None and all(abs(t)<(.8*efforts[n] if b1 else 9.) for t,n in zip(loads['arm_gravity_torque_Nm'],names))
                    if arm_limit_ok and loads['max_leg_effort_fraction']<.8 and min(loads['foot_normal_N'])>1 and loads['equilibrium_residual']<.01:
                        successful=dict(angle_deg=float(degrees),target_head_forward_m=distance,
                            tip_error_m=reach,lowest_finger_z_m=lowest,support_margin_m=stability,
                            mass_properties=mass_properties(m,d),static_loads=loads,qpos=d.qpos.tolist())
                        break
                if successful:
                    trials.append(successful)
                    bestposes.setdefault(str(distance),successful)
            if len(bestposes)==3:break
        case['ground_reach']=bestposes
        case['ground_reach_kinematic']=kinematic
        case['ground_reach_screen']='Illustrative AS2 9 Nm per joint; B1 80% URDF effort. Not verified continuous hardware limits.'
        (OUT/(case['name']+'_reach_trials.json')).write_text(json.dumps(trials,indent=2))
        results.append(case)
        print(case['name'],'head reach',case['workspace'],'tilt',[(k,v['angle_deg']) for k,v in bestposes.items()],flush=True)
        (OUT/'results.json').write_text(json.dumps(results,indent=2))
    return results


if __name__=='__main__':analyze(build())
