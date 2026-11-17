"""Diagnostic only: remove orientation constraint for residual failed poses."""
import check as c
import json,numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
c.lo[:3]=[.20,np.deg2rad(-35),np.deg2rad(-35)];c.hi[:3]=[.42,np.deg2rad(35),np.deg2rad(50)]
out=[]
for rec in json.loads((c.O/'step_refined.json').read_text()):
 if rec['passed']:continue
 target=np.array(rec['target']);q=np.array(rec['qpos']);angles=Rotation.from_quat(q[3:7][[1,2,3,0]]).as_euler('xyz');seed=np.r_[q[2],angles[:2],q[c.g.lq],q[c.g.aq]]
 def err(x):
  c.put(x)
  return np.r_[(c.d.xpos[c.g.feet]-c.g.foot_targets).ravel(),c.d.xpos[c.g.tcp]-target]
 fit=least_squares(err,np.clip(seed,c.lo+1e-6,c.hi-1e-6),bounds=(c.lo,c.hi),max_nfev=400,ftol=1e-11,xtol=1e-11,gtol=1e-11)
 e=err(fit.x);mujoco.mj_forward(c.m,c.d);hits=c.g.collisions();margin=float(-np.max(c.g.hull.equations[:,:2]@c.d.subtree_com[c.bid,:2]+c.g.hull.equations[:,2]))
 pe=float(np.linalg.norm(e[12:]));fe=float(np.max(np.abs(e[:12])))
 out.append(dict(target=target.tolist(),position_only_pass=pe<.003 and fe<.002 and not hits and margin>=.02,position_error_m=pe,foot_error_m=fe,hits=hits,com_margin_m=margin,orientation_unconstrained=True,qpos=c.d.qpos.tolist()))
 print(target.tolist(),out[-1]['position_only_pass'],pe*1000,flush=True)
(c.O/'position_only.json').write_text(json.dumps(out,indent=2))
