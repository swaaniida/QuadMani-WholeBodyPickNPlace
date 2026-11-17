"""Finite-grid 6D IK and URDF convex-collision screening, outside training."""
from pathlib import Path
import itertools, json, time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parent
model = mujoco.MjModel.from_xml_path(str(ROOT/'validation/collision_model.urdf'))
data = mujoco.MjData(model)
adjacent = {frozenset((j.find('parent').get('link'), j.find('child').get('link')))
            for j in ET.parse(ROOT/'validation/collision_model.urdf').getroot().findall('joint')}
arm_ids = np.array([model.joint('piper_joint'+str(i)).id for i in range(1,7)])
qids = model.jnt_qposadr[arm_ids]; dids = model.jnt_dofadr[arm_ids]
low, high = model.jnt_range[arm_ids].T
neutral = np.array([0.,1.1,-.55,0.,0.,0.])
for leg in ('FL','FR','RL','RR'):
    for name,value in [('hip',0.),('thigh',.8),('calf',-1.6)]:data.qpos[model.joint(leg+'_'+name+'_joint').qposadr]=value
for name,value in [('piper_gripper_joint1',.049),('piper_gripper_joint2',-.049)]:data.qpos[model.joint(name).qposadr]=value
tcp=model.body('piper_umi_nominal_tcp').id
base=np.array([.22435,0,.1319]);jp=np.zeros((3,model.nv));jr=jp.copy()

def solve(position,rotation,starts):
    target_quat=np.zeros(4);mujoco.mju_mat2Quat(target_quat,rotation.ravel())
    for initial in starts:
        data.qpos[qids]=np.clip(initial,low+1e-5,high-1e-5)
        for _ in range(90):
            mujoco.mj_forward(model,data)
            actual=np.zeros(4);mujoco.mju_mat2Quat(actual,data.xmat[tcp]);inverse=actual.copy();inverse[1:]*=-1
            delta=np.zeros(4);mujoco.mju_mulQuat(delta,target_quat,inverse)
            if delta[0]<0:delta*=-1
            angle=np.zeros(3);mujoco.mju_quat2Vel(angle,delta,1)
            error=position-data.xpos[tcp]
            if np.linalg.norm(error)<.005 and np.linalg.norm(angle)<np.deg2rad(3):
                hits=[]
                for contact in data.contact:
                    names=[model.body(model.geom_bodyid[g]).name for g in (contact.geom1,contact.geom2)]
                    if contact.dist<-.001 and frozenset(names) not in adjacent and any(n.startswith('piper_') for n in names):hits.append(names)
                return True,hits,data.qpos[qids].copy(),float(np.linalg.norm(error)),float(np.linalg.norm(angle))
            mujoco.mj_jacBody(model,data,jp,jr,tcp)
            J=np.vstack([jp[:,dids],.25*jr[:,dids]])
            residual=np.r_[error,.25*angle]
            step=J.T@np.linalg.solve(J@J.T+.0004*np.eye(6),residual)
            data.qpos[qids]=np.clip(data.qpos[qids]+np.clip(step,-.15,.15),low+1e-5,high-1e-5)
    return False,[],data.qpos[qids].copy(),float(np.linalg.norm(error)),float(np.linalg.norm(angle))

from check_geometry import rotation
rng=np.random.default_rng(20261111)
results={};start=time.monotonic()
for name,bounds in [('proposed',[(.38,.52),(-.25,.60),(-.4,.4),(-.15,.15)]),('medium',[(.40,.52),(.0,.55),(-.35,.35),(-.10,.10)]),('balanced',[(.40,.50),(.0,.55),(-.30,.30),(-.10,.10)]),('inner',[(.40,.48),(.10,.50),(-.25,.25),(-.08,.08)])]:
    poses=list(itertools.product(*[np.linspace(*b,4) for b in bounds[:3]],[-1,0,1],[-1,0,1],[-1,0,1]))
    counts={'samples':len(poses),'ik_failed':0,'collision':0,'valid':0};examples=[];last=neutral
    for i,(radius,pitch,yaw,dr,dp,dy) in enumerate(poses):
        position=base+radius*np.array([np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)])
        target=rotation([dr*bounds[3][1],-pitch+2.455216137744035+dp*bounds[3][1],yaw+dy*bounds[3][1]])
        ok,hits,q,err,ang=solve(position,target,[last,neutral])
        if ok:last=q
        key='ik_failed' if not ok else ('collision' if hits else 'valid');counts[key]+=1
        if key!='valid' and len(examples)<10:examples.append({'sphere':[radius,pitch,yaw],'orientation_delta':[dr,dp,dy],'kind':key,'position_error_m':err,'rotation_error_deg':float(np.rad2deg(ang)),'contacts':hits})
        if i%300==0:print(name,i,len(poses),counts,flush=True)
    results[name]={'bounds':bounds,**counts,'examples':examples}
    (ROOT/'validation/workspace.json').write_text(json.dumps(results,indent=2)+'\n')
    print(name,counts,'wall',time.monotonic()-start,flush=True)
# Follow the actual spherical interpolation from the current posture.
path_failures = 0
path_contacts = 0
for path_index in range(128):
    start_sphere = np.array([.427898468676838, .421686284147847, 0.])
    end_sphere = rng.uniform([.40,.10,-.25],[.48,.50,.25])
    if path_index % 2:
        start_sphere = rng.uniform([.40,.10,-.25],[.48,.50,.25])
    delta = rng.uniform(-.08,.08,3)
    q = neutral.copy()
    failed = False
    collided = False
    for t in np.linspace(0,1,11):
        radius,pitch,yaw = (1-t)*start_sphere+t*end_sphere
        position = base+radius*np.array([np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)])
        target = rotation([delta[0],-pitch+2.455216137744035+delta[1],yaw+delta[2]])
        ok,hits,q,_,_ = solve(position,target,[q])
        failed |= not ok
        collided |= bool(hits)
    path_failures += int(failed)
    path_contacts += int(collided)
results['selected_paths'] = {'paths':128,'samples_per_path':11,'ik_failed_paths':path_failures,'collision_paths':path_contacts,'seed':20261111}
(ROOT/'validation/workspace.json').write_text(json.dumps(results,indent=2)+'\n')
print('paths',results['selected_paths'],flush=True)

# Nominal load and gain design, with explicit design allowances.
data.qpos[qids]=neutral;mujoco.mj_forward(model,data)
feet=[model.body(leg+'_foot').id for leg in ('FL','FR','RL','RR')]
mass=float(model.body_mass.sum());com=data.subtree_com[model.body('base_link').id];feet_pos=data.xpos[feet]
A=np.vstack([np.ones(4),feet_pos[:,0]-com[0],feet_pos[:,1]-com[1]])
loads=lsq_linear(A,[mass*9.81,0,0],bounds=(0,np.inf)).x
torque=data.qfrc_bias.copy()
for body,load in zip(feet,loads):
    mujoco.mj_jacBody(model,data,jp,jr,body);torque-=jp.T@np.array([0,0,load])
M=np.zeros((model.nv,model.nv));mujoco.mj_fullM(model,M,data.qM)
leg_ids=[model.joint(leg+'_'+name+'_joint').dofadr[0] for leg in ('FL','FR','RL','RR') for name in ('hip','thigh','calf')]
static=np.abs(torque[leg_ids]).reshape(4,3).max(0)
inertia=np.diag(M)[leg_ids].reshape(4,3).max(0)+[.026,.026,.038]
kp=np.maximum(80,np.maximum(static/.12,inertia*(2*np.pi*5)**2));kd=1.4*np.sqrt(kp*inertia)
half=(feet_pos.max(0)-feet_pos.min(0))[:2]/2;com_range=.05*half*mass/8.8
hardware={'mass_kg':mass,'static_foot_load_N':loads.tolist(),'static_leg_torque_max_Nm':static.tolist(),'inertia_plus_armature_kg_m2':inertia.tolist(),'kp':kp.tolist(),'kd':kd.tolist(),'armature':[.026,.026,.038],'added_base_mass_kg':[0,mass*.1],'base_com_half_range_m':[float(com_range[0]),float(com_range[1]),float(min(com_range))],'max_push_velocity_m_s':float(.5*np.sqrt(.3244069024/.5161639912)),'design':'nominal sag <=0.12rad, damping ratio 0.7, swing bandwidth 5Hz; approximate diagonal model'}
(ROOT/'validation/hardware.json').write_text(json.dumps(hardware,indent=2)+'\n');print(hardware,flush=True)
