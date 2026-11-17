"""Offline multi-posture 6D workspace inspection. No sampler/config/training edits."""
import json,time,hashlib,sys
import numpy as np
import mujoco
from scipy.optimize import lsq_linear
import geometry_tools as g
OUT=g.OUT;m,d=g.m,g.d
MODE=sys.argv[1] if len(sys.argv)>1 else 'candidate'
assert MODE in ('candidate','old_orientation')
center=np.array([.22435,0,.4563069024246581]);baseid=m.body('base_link').id
postures=[]
for h,t in [(.3244069024246581,0),(.3244069024246581,-15),(.30,15),(.30,25),(.27,30),(.27,35)]:
 if g.leg_ik(h,np.deg2rad(t)):
  postures.append(dict(h=h,t=t,root=d.qpos[:7].copy(),legs=d.qpos[g.lq].copy(),last=g.neutral.copy()))
lim={j.get('name'):float(j.find('limit').get('effort')) for j in g.c.joints if j.find('limit') is not None}
leglim=np.array([lim[m.joint(int(i)).name] for i in g.leg_ids]);armlim=np.array([lim[m.joint(int(i)).name] for i in g.arm_ids])

def gravity_support():
 d.qvel[:]=0;mujoco.mj_forward(m,d)
 J=[]
 for bid in g.feet:
  jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,bid);J.append(jp[2])
 J=np.array(J)
 forces=lsq_linear(J[:,:6].T,d.qfrc_bias[:6],bounds=(0,np.inf),tol=1e-10).x
 residual=J[:,:6].T@forces-d.qfrc_bias[:6]
 tau=d.qfrc_bias-J.T@forces
 return dict(foot_fz_N=forces.tolist(),arm_gravity_tau_Nm=tau[g.av].tolist(),leg_gravity_support_tau_Nm=tau[g.lv].tolist(),
             max_leg_urdf_effort_ratio=float(np.max(abs(tau[g.lv])/leglim)),max_arm_urdf_effort_ratio=float(np.max(abs(tau[g.av])/armlim)),
             force_residual_N=float(np.linalg.norm(residual[:3])),moment_residual_Nm=float(np.linalg.norm(residual[3:])))

def path_check(p,arm):
 # Screening witness only: joint-space arm path + contact-preserving leg IK.
 start_arm=g.neutral.copy();minmargin=1.
 for s in np.linspace(0,1,9):
  h=(1-s)*.3244069024246581+s*p['h'];theta=s*p['t']
  if not g.leg_ik(h,np.deg2rad(theta)):return dict(ok=False,reason='leg_path_ik',at=float(s))
  d.qpos[g.aq]=(1-s)*start_arm+s*arm;g.fk()
  hits=g.collisions()
  if hits:return dict(ok=False,reason='path_collision',at=float(s),hits=hits)
  com=d.subtree_com[baseid];margin=float(-np.max(g.hull.equations[:,:2]@com[:2]+g.hull.equations[:,2]));minmargin=min(minmargin,margin)
  if margin<.02:return dict(ok=False,reason='path_com_margin',at=float(s))
 return dict(ok=True,samples=9,min_com_margin_m=minmargin)

def spherical_path_check(p,target_sphere):
 start=np.array([.427898468676838,.421686284147847,0.])
 previous=g.neutral.copy();max_tau=np.zeros(6);min_margin=1.
 for t in np.linspace(0,1,13):
  if not g.leg_ik((1-t)*.3244069024246581+t*p['h'],np.deg2rad(t*p['t'])):return dict(ok=False,reason='leg_path_ik',at=float(t))
  r,phi,yaw=(1-t)*start+t*target_sphere
  target=center+r*np.array([np.cos(phi)*np.cos(yaw),np.cos(phi)*np.sin(yaw),np.sin(phi)])
  x=center[0]+np.linalg.norm((target-center)[:2]);z=target[2]
  beta_old=2.455216137744035-phi
  beta_floor=np.deg2rad(175-25*np.clip((x-.55)/.425,0,1));blend=np.clip((.25-z)/.15,0,1)
  beta=beta_old if MODE=='old_orientation' else (1-blend)*beta_old+blend*beta_floor
  ok,pe,ae=g.arm_ik(target,beta,[previous,g.neutral],yaw)
  if not ok:return dict(ok=False,reason='arm_path_ik',at=float(t),position_error_m=float(pe))
  previous=d.qpos[g.aq].copy();hits=g.collisions()
  if hits:return dict(ok=False,reason='path_collision',at=float(t),hits=hits)
  com=d.subtree_com[baseid];margin=float(-np.max(g.hull.equations[:,:2]@com[:2]+g.hull.equations[:,2]));min_margin=min(min_margin,margin)
  if margin<.02:return dict(ok=False,reason='path_com_margin',at=float(t))
  torque=gravity_support();max_tau=np.maximum(max_tau,np.abs(torque['arm_gravity_tau_Nm']))
  if torque['max_leg_urdf_effort_ratio']>1 or torque['max_arm_urdf_effort_ratio']>1 or torque['force_residual_N']>.5 or torque['moment_residual_Nm']>.2:return dict(ok=False,reason='path_static_support_or_urdf_torque',at=float(t))
 return dict(ok=True,samples=13,min_com_margin_m=min_margin,max_arm_gravity_tau_Nm=max_tau.tolist())

rows=[];start=time.time()
for x in [.45,.60,.75,.90,.975]:
 for z in [.02,.10,.30,.50,.70,.85]:
  for yaw in [-.30,0,.30]:
   dx=x-center[0];target=np.array([center[0]+dx*np.cos(yaw),dx*np.sin(yaw),z]);vec=target-center
   radius=np.linalg.norm(vec);phi=np.arctan2(vec[2],np.linalg.norm(vec[:2]))
   beta_old=2.455216137744035-phi
   # Candidate only: preserve home orientation aloft; increasingly forward-slant down near far floor.
   beta_floor=np.deg2rad(175-25*np.clip((x-.55)/.425,0,1))
   blend=np.clip((.25-z)/.15,0,1);beta=beta_old if MODE=='old_orientation' else (1-blend)*beta_old+blend*beta_floor
   row=dict(target=target.tolist(),nominal_x=x,z=z,yaw_rad=yaw,r_m=float(radius),phi_rad=float(phi),tool_pitch_deg=float(np.rad2deg(beta)),endpoint_pass=False,path_pass=False,attempts=[])
   for p in postures:
    d.qpos[:7]=p['root'];d.qpos[g.lq]=p['legs']
    ok,pe,ae=g.arm_ik(target,beta,[p['last'],g.neutral,np.array([0,1.8,-1.4,0,-.5,0])],yaw)
    attempt=dict(body_z=p['h'],body_pitch_deg=p['t'],ik=bool(ok),position_error_m=float(pe),orientation_error_deg=float(np.rad2deg(ae)))
    row['attempts'].append(attempt)
    if not ok:continue
    arm=d.qpos[g.aq].copy();p['last']=arm.copy()
    reserve=float(np.rad2deg(np.min(np.r_[d.qpos[g.aq]-g.alo,g.ahi-d.qpos[g.aq],d.qpos[g.lq]-g.llo,g.lhi-d.qpos[g.lq]])))
    hits=g.collisions();com=d.subtree_com[baseid].copy();margin=float(-np.max(g.hull.equations[:,:2]@com[:2]+g.hull.equations[:,2]))
    attempt.update(joint_reserve_deg=reserve,collision_count=len(hits),com_margin_m=margin)
    if hits or margin<.02 or reserve<5:continue
    torque=gravity_support();attempt.update(torque)
    if torque['max_leg_urdf_effort_ratio']>1 or torque['max_arm_urdf_effort_ratio']>1 or torque['force_residual_N']>.5 or torque['moment_residual_Nm']>.2:continue
    row['endpoint_pass']=True
    witness=dict(body_z=p['h'],body_pitch_deg=p['t'],leg_q=p['legs'].tolist(),arm_q=arm.tolist(),com_xyz=com.tolist(),com_margin_m=margin,joint_reserve_deg=reserve,**torque)
    joint_path=path_check(p,arm);witness['joint_path']=joint_path
    path=spherical_path_check(p,np.array([radius,phi,yaw])) if joint_path['ok'] else joint_path
    witness['path']=path
    if 'witness' not in row or path['ok']:row['witness']=witness
    if path['ok']:
     row['path_pass']=True;break
   rows.append(row)
 print('x',x,'rows',len(rows),'endpoints',sum(v['endpoint_pass'] for v in rows),'paths',sum(v['path_pass'] for v in rows),'seconds',round(time.time()-start,1),flush=True)
 (OUT/(MODE+'_progress.json')).write_text(json.dumps(dict(samples=rows),indent=2))
result=dict(orientation_mode=MODE,urdf=str(g.URDF),urdf_sha256=hashlib.sha256(g.URDF.read_bytes()).hexdigest(),mass_kg=float(m.body_mass.sum()),center=center.tolist(),postures=[dict(body_z=p['h'],body_pitch_deg=p['t']) for p in postures],
 tested=len(rows),endpoint_pass=sum(r['endpoint_pass'] for r in rows),path_pass=sum(r['path_pass'] for r in rows),
 assumptions=dict(com_margin_m=.02,joint_limit_reserve_deg=5,position_tolerance_m=.003,orientation_tolerance_deg=2,path_samples=9,torque_basis='URDF effort ceilings only; PiPER 100Nm not manufacturer-verified. Gravity only, no payload/inertia/contact transient.'),samples=rows,
 scope='Finite nominal flat-ground static screen plus coarse joint-space witnesses. Not continuous workspace proof, not spherical runtime trajectory validation, not dynamic or hardware torque certification. No sampler/reward/PPO edits.')
(OUT/(MODE+'_inspection.json')).write_text(json.dumps(result,indent=2))
for rel,sha in json.loads((OUT/'source_before.json').read_text()).items():assert hashlib.sha256((OUT.parents[1]/rel).read_bytes()).hexdigest()==sha,rel
print('DONE',result['tested'],result['endpoint_pass'],result['path_pass'],round(time.time()-start,1),flush=True)
