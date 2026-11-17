"""Re-test failed goals allowing static foot placement changes up to 5cm/axis."""
import check as c
import json,numpy as np,mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
c.lo[:3]=[.20,np.deg2rad(-35),np.deg2rad(-35)];c.hi[:3]=[.42,np.deg2rad(35),np.deg2rad(50)]
records=[]
for mode in ['grid','random','corners']:
 for rec in json.loads((c.O/('refine_'+mode+'_relaxed.json')).read_text())['records']:
  if not rec['passed']:records.append(rec)
out=[]
for rec in records:
 phi,yaw,delta=rec['phi'],rec['yaw'],rec['delta'];target=np.array(rec['target'])
 rot=Rotation.from_euler('ZYX',[yaw+delta[2],c.pitch(target,phi)+delta[1],delta[0]]).as_matrix()
 q=np.array(rec['qpos']);a=Rotation.from_quat(q[3:7][[1,2,3,0]]).as_euler('xyz');seed=np.r_[q[2],a[:2],q[c.g.lq],q[c.g.aq]]
 def err(x):
  c.put(x);fe=c.d.xpos[c.g.feet]-c.g.foot_targets
  fe[:,:2]=np.sign(fe[:,:2])*np.maximum(np.abs(fe[:,:2])-.05,0)
  return np.r_[fe.ravel(),c.d.xpos[c.g.tcp]-target,.25*Rotation.from_matrix(rot@c.d.xmat[c.g.tcp].reshape(3,3).T).as_rotvec()]
 fit=least_squares(err,np.clip(seed,c.lo+1e-6,c.hi-1e-6),bounds=(c.lo,c.hi),max_nfev=700,ftol=1e-11,gtol=1e-11,xtol=1e-11)
 e=err(fit.x);mujoco.mj_forward(c.m,c.d);hits=c.g.collisions();hull=ConvexHull(c.d.xpos[c.g.feet,:2]);margin=float(-np.max(hull.equations[:,:2]@c.d.subtree_com[c.bid,:2]+hull.equations[:,2]))
 pe=float(np.linalg.norm(e[12:15]));ae=float(np.rad2deg(np.linalg.norm(e[15:])/.25));fe=float(np.max(np.abs(e[:12])))
 ok=pe<.003 and ae<2 and fe<.002 and not hits and margin>=.02
 displacement=c.d.xpos[c.g.feet]-c.g.foot_targets
 out.append({**rec,'passed':ok,'position_error_m':pe,'orientation_error_deg':ae,'foot_error_m':fe,'hits':hits,'com_margin_m':margin,'qpos':c.d.qpos.tolist(),'foot_displacement_m':displacement.tolist(),'body_h_roll_pitch':fit.x[:3].tolist(),'foot_placement_adjusted':True})
 print(len(out),len(records),ok,'pos_mm',pe*1000,'deg',ae,'maxfootxy_mm',np.abs(displacement[:,:2]).max()*1000,flush=True)
(c.O/'step_refined.json').write_text(json.dumps(out,indent=2))
