"""Offline fixed-foot whole-body pose search, latest URDF. No training edits."""
from pathlib import Path
import json,math,time,hashlib
import numpy as np,mujoco
from scipy.optimize import least_squares,lsq_linear
from scipy.spatial.transform import Rotation
import geometry_tools as g
O=Path(__file__).resolve().parent;m,d=g.m,g.d;bid=m.body('base_link').id
center=np.array([.22435,0,.4563069024246581]);start=time.time()
choices=[(.975,.02,0,'① 최전방 바닥'),(.975,.02,.3,'② 좌측 전방 바닥'),(.975,.02,-.3,'③ 우측 전방 바닥'),(.45,1.,0,'④ 가까운 상단 1.0m'),(.75,1.,0,'⑤ 전방 상단 1.0m'),(.975,1.,0,'⑥ 최전방 상단 1.0m')]
old=json.loads((O.parent/'workspace_limits_20261115/measurements.json').read_text())['poses']
# Keep the feet at the same world locations; shift/tilt the trunk through Leg IK.
postures=[]
for x in [0.]:
 for h in [.3244069024246581,.35,.37,.39,.30,.27,.25]:
  for angle in [0,-15,-25,15,25,30,35]:
   if g.leg_ik(h,np.deg2rad(angle),x):postures.append((h,angle,x,d.qpos[:7].copy(),d.qpos[g.lq].copy()))
print('Valid leg postures',len(postures),flush=True)

def support():
 d.qvel[:]=0;mujoco.mj_forward(m,d);J=[]
 for foot in g.feet:
  jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,foot);J.append(jp[2])
 J=np.array(J);f=lsq_linear(J[:,:6].T,d.qfrc_bias[:6],bounds=(0,np.inf),tol=1e-10).x
 res=J[:,:6].T@f-d.qfrc_bias[:6]
 return dict(foot_forces_N=f.tolist(),force_residual_N=float(np.linalg.norm(res[:3])),moment_residual_Nm=float(np.linalg.norm(res[3:])),arm_gravity_tau_Nm=(d.qfrc_bias-J.T@f)[g.av].tolist())

records=[]
for x,z,yaw,title in choices:
 target=np.array([center[0]+(x-center[0])*np.cos(yaw),(x-center[0])*np.sin(yaw),z]);v=target-center;r=np.linalg.norm(v);phi=np.arctan2(v[2],np.linalg.norm(v[:2]))
 blend=np.clip((.25-z)/.15,0,1);beta=(1-blend)*(2.455216137744035-phi)+blend*np.deg2rad(175-25*np.clip((x-.55)/.425,0,1));rot=Rotation.from_euler('ZYX',[yaw,beta,0]).as_matrix()
 found=None;best=None;tried=0;last=g.neutral.copy()
 # Reuse known ground witnesses as starting points, but recompute target error.
 candidates=postures
 if z==.02:
  rec=old[[p['title'] for p in old].index(title)];q=np.array(rec['qpos']);d.qpos[:]=q
  candidates=[(rec['body_z_m'],rec['body_pitch_deg'],0.,q[:7],q[g.lq])]+postures
  last=q[g.aq].copy()
 for h,angle,shift,root,legs in candidates:
  tried+=1;d.qpos[:7]=root;d.qpos[g.lq]=legs
  ok,pe,ae=g.arm_ik(target,beta,[last,g.neutral,np.array([0,.5,-.6,0,.5,0])],yaw)
  last=d.qpos[g.aq].copy()
  hits=g.collisions();margin=float(-np.max(g.hull.equations[:,:2]@d.subtree_com[bid,:2]+g.hull.equations[:,2]))
  # Even a failed solve is retained honestly for a best-effort illustration.
  score=pe+.25*ae+(.5 if hits else 0)+max(0,.02-margin)*5
  if best is None or score<best['score']:
   best=dict(score=float(score),qpos=d.qpos.tolist(),position_error_m=float(pe),orientation_error_deg=float(np.rad2deg(ae)),body_z_m=h,body_pitch_deg=angle,body_x_m=shift,com_margin_m=margin,collision_hits=hits)
  if not ok or hits or margin<.02:continue
  # Tight 6D pose polishing, preserving hard joint limits.
  def err(q):
   d.qpos[g.aq]=q;g.fk()
   return np.r_[d.xpos[g.tcp]-target,.25*Rotation.from_matrix(rot@d.xmat[g.tcp].reshape(3,3).T).as_rotvec()]
  fit=least_squares(err,last,bounds=(g.alo+1e-6,g.ahi-1e-6),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=120)
  d.qpos[g.aq]=fit.x;mujoco.mj_forward(m,d)
  pe=np.linalg.norm(d.xpos[g.tcp]-target);ae=np.linalg.norm(Rotation.from_matrix(rot@d.xmat[g.tcp].reshape(3,3).T).as_rotvec());hits=g.collisions();margin=float(-np.max(g.hull.equations[:,:2]@d.subtree_com[bid,:2]+g.hull.equations[:,2]));stat=support()
  if pe>.0001 or ae>np.deg2rad(.05) or hits or margin<.02 or stat['force_residual_N']>.5 or stat['moment_residual_Nm']>.2:continue
  reserve=float(np.rad2deg(np.min(np.r_[d.qpos[g.aq]-g.alo,g.ahi-d.qpos[g.aq],d.qpos[g.lq]-g.llo,g.lhi-d.qpos[g.lq]])))
  found=dict(qpos=d.qpos.tolist(),position_error_m=float(pe),orientation_error_deg=float(np.rad2deg(ae)),body_z_m=h,body_pitch_deg=angle,body_x_m=shift,com_margin_m=margin,collision_hits=hits,joint_reserve_deg=reserve,**stat);break
 rec=dict(title=title,target_xyz=target.tolist(),sphere=[float(r),float(phi),yaw],tool_rpy_rad=[0,float(beta),yaw],reached=found is not None,postures_tried=tried,**(found if found else best))
 d.qpos[:]=rec['qpos'];mujoco.mj_forward(m,d);rec['tcp_xyz']=d.xpos[g.tcp].tolist();records.append(rec)
 print(title,'reached',rec['reached'],'error_mm',round(rec['position_error_m']*1000,4),'postures',tried,'time',round(time.time()-start,1),flush=True)
 (O/'poses_progress.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
(O/'poses.json').write_text(json.dumps(dict(urdf=str(g.URDF),urdf_sha256=hashlib.sha256(g.URDF.read_bytes()).hexdigest(),leg_postures=len(postures),poses=records,scope='Static FK/IK illustrations, no policy rollout/training. Fixed feet, nominal flat ground, zero payload. Finite search does not prove global infeasibility. No online IK validity or bank.'),indent=2,ensure_ascii=False))
