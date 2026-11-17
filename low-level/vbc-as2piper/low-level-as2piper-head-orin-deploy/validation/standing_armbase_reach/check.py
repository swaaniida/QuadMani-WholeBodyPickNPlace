"""Offline pose reachability; fixed feet, variable trunk height/roll/pitch."""
from pathlib import Path
import json,time,hashlib,sys,math,os
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import geometry_tools as g
O=Path(__file__).resolve().parent;m,d=g.m,g.d;start=time.time();bid=m.body('base_link').id
# Derive nominal arm base from actual URDF FK, not a visual mesh origin.
center=d.xpos[m.body('piper_base_link').id].copy()
assert np.max(np.abs(center-np.array([.22435,0,.4563069024246581])))<1e-8
rmax=.80;pmin=-math.asin((center[2]-.02)/rmax);pmax=math.asin((1-center[2])/rmax)
lo=np.r_[.25,np.deg2rad([-20,-25]),g.llo+1e-5,g.alo+1e-5]
hi=np.r_[.39,np.deg2rad([20,35]),g.lhi-1e-5,g.ahi-1e-5]
if os.environ.get('RELAX_BODY')=='1':
 lo[:3]=[.20,np.deg2rad(-35),np.deg2rad(-35)]
 hi[:3]=[.42,np.deg2rad(35),np.deg2rad(50)]
home=np.r_[.3244069024246581,0,0,[0,.8,-1.6]*4,g.neutral];cached=home.copy()

def put(x):
 d.qpos[:3]=[0,0,x[0]];quat=Rotation.from_euler('xyz',[x[1],x[2],0]).as_quat();d.qpos[3:7]=quat[[3,0,1,2]]
 d.qpos[g.lq]=x[3:15];d.qpos[g.aq]=x[15:];g.fk()

def pitch(target,phi):
 nominalx=center[0]+np.linalg.norm((target-center)[:2]);blend=np.clip((.25-target[2])/.15,0,1)
 return (1-blend)*(2.455216137744035-phi)+blend*np.deg2rad(175-25*np.clip((nominalx-.55)/.425,0,1))

def solve(r,phi,yaw,delta,refine=False,seedq=None):
 global cached
 target=center+r*np.array([np.cos(phi)*np.cos(yaw),np.cos(phi)*np.sin(yaw),np.sin(phi)])
 rot=Rotation.from_euler('ZYX',[yaw+delta[2],pitch(target,phi)+delta[1],delta[0]]).as_matrix()
 def err(x):
  put(x)
  return np.r_[(d.xpos[g.feet]-g.foot_targets).ravel(),d.xpos[g.tcp]-target,.25*Rotation.from_matrix(rot@d.xmat[g.tcp].reshape(3,3).T).as_rotvec()]
 seeds=[cached,home,np.r_[.27,0,np.deg2rad(30),[0,.8,-1.6]*4,[0,.5,-.6,0,.5,0]]]
 if seedq is not None:
  q=np.array(seedq);angles=Rotation.from_quat(q[3:7][[1,2,3,0]]).as_euler('xyz');seeds.insert(0,np.r_[q[2],angles[:2],q[g.lq],q[g.aq]])
 best=None
 for seed in seeds:
  fit=least_squares(err,np.clip(seed,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=500 if refine else 65,ftol=1e-11 if refine else 1e-7,gtol=1e-11 if refine else 1e-7,xtol=1e-11 if refine else 1e-7)
  e=err(fit.x);pe=float(np.linalg.norm(e[12:15]));ae=float(np.rad2deg(np.linalg.norm(e[15:])/.25));fe=float(np.max(np.abs(e[:12])))
  mujoco.mj_forward(m,d);hits=g.collisions();margin=float(-np.max(g.hull.equations[:,:2]@d.subtree_com[bid,:2]+g.hull.equations[:,2]))
  reached=pe<.003 and ae<2 and fe<.002;ok=reached and not hits and margin>=.02
  rec=dict(r=float(r),phi=float(phi),yaw=float(yaw),delta=list(delta),target=target.tolist(),passed=ok,kinematic_pass=reached,position_error_m=pe,orientation_error_deg=ae,foot_error_m=fe,hits=hits,com_margin_m=margin,body_h_roll_pitch=fit.x[:3].tolist(),qpos=d.qpos.tolist())
  score=pe+.25*np.deg2rad(ae)+fe+(.5 if hits else 0)+max(0,.02-margin)
  if best is None or score<best[0]:best=(score,rec)
  if ok:cached=fit.x.copy();return rec
 return best[1]

if __name__=='__main__':
 mode=sys.argv[1] if len(sys.argv)>1 else 'grid';jobs=[];records=[]
 if mode=='current':
  for phi in np.linspace(-math.asin((center[2]-.02)/.65),math.asin((1-center[2])/.65),5):
   for r in [.35,.5,.65]:
    for yaw in [-.3,0,.3]:jobs.append((r,phi,yaw,[0,0,0]))
 elif mode=='grid':
  for phi in np.linspace(pmin,pmax,5):
   for r in [.35,.575,.8]:
    for yaw in [-1.2,-.6,0,.6,1.2]:jobs.append((r,phi,yaw,[0,0,0]))
 elif mode=='random':
  rng=np.random.default_rng(20261115)
  for _ in range(96):jobs.append((rng.uniform(.35,.8),rng.uniform(pmin,pmax),rng.uniform(-1.2,1.2),rng.uniform(-.08,.08,3).tolist()))
 elif mode=='corners':
  for phi in [pmin,pmax]:
   for r in [.35,.8]:
    for yaw in [-1.2,0,1.2]:
     for dr in [-.08,.08]:
      for dp in [-.08,.08]:
       for dy in [-.08,.08]:jobs.append((r,phi,yaw,[dr,dp,dy]))
 elif mode=='refine':
  originals=json.loads((O/(os.environ.get('REFINE_SOURCE','grid')+'.json')).read_text())['records']
  for rec in originals:
   if not rec['passed']:jobs.append((rec['r'],rec['phi'],rec['yaw'],rec['delta'],True,rec['qpos']))
 for i,job in enumerate(jobs):
  rec=solve(*job);records.append(rec)
  print(i+1,len(jobs),rec['passed'],'p_mm',round(rec['position_error_m']*1000,2),'deg',round(rec['orientation_error_deg'],2),'hits',len(rec['hits']),'sec',round(time.time()-start,1),flush=True)
  (O/((('refine_'+os.environ.get('REFINE_SOURCE','grid')+('_relaxed' if os.environ.get('RELAX_BODY')=='1' else '')) if mode=='refine' else mode)+'.json')).write_text(json.dumps(dict(center=center.tolist(),radius=[.35,.65 if mode=='current' else .8],elevation=([-math.asin((center[2]-.02)/.65),math.asin((1-center[2])/.65)] if mode=='current' else [pmin,pmax]),yaw=([-.3,.3] if mode=='current' else [-1.2,1.2]),urdf_sha256=hashlib.sha256(g.URDF.read_bytes()).hexdigest(),records=records),indent=2))
