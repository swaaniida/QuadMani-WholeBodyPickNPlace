import check as c
import json,numpy as np,sys
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
mode=sys.argv[1] if len(sys.argv)>1 else 'random'
records=json.loads((c.O/(mode+'.json')).read_text())['records'];out=[]
for rec in records:
 if rec['passed']:continue
 r,phi,yaw,delta=[rec[k] for k in ('r','phi','yaw','delta')]
 target=np.array(rec['target']);rot=Rotation.from_euler('ZYX',[yaw+delta[2],c.pitch(target,phi)+delta[1],delta[0]]).as_matrix()
 def err(x):
  c.put(x)
  return np.r_[(c.d.xpos[c.g.feet]-c.g.foot_targets).ravel(),c.d.xpos[c.g.tcp]-target,.25*Rotation.from_matrix(rot@c.d.xmat[c.g.tcp].reshape(3,3).T).as_rotvec()]
 q=np.array(rec['qpos']);tilt=2*np.arctan2(q[5],q[3]);seed=np.r_[q[2],tilt,q[c.g.lq],q[c.g.aq]]
 fit=least_squares(err,np.clip(seed,c.lo+1e-6,c.hi-1e-6),bounds=(c.lo,c.hi),max_nfev=1500,ftol=1e-12,gtol=1e-12,xtol=1e-12)
 e=err(fit.x);c.mujoco.mj_forward(c.m,c.d)
 hits=c.g.collisions();pe=float(np.linalg.norm(e[12:15]));ae=float(np.linalg.norm(e[15:])/.25)
 result=dict(target=rec['target'],position_error_m=pe,orientation_error_deg=float(np.rad2deg(ae)),foot_error_m=float(np.max(np.abs(e[:12]))),hits=hits,qpos=c.d.qpos.tolist(),x=fit.x.tolist(),nfev=fit.nfev)
 print(result,flush=True);out.append(result)
(c.O/('refined_'+mode+'.json')).write_text(json.dumps(out,indent=2))
