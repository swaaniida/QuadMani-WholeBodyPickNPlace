"""Current EE goals: offline whole-body IK and unilateral static-support LP."""
from pathlib import Path
import json,time,math,hashlib,xml.etree.ElementTree as ET
import numpy as np,mujoco
from scipy.optimize import least_squares,linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import geometry_tools as g
O=Path(__file__).resolve().parent;m,d=g.m,g.d;bid=m.body('base_link').id;start=time.time()
center=np.array([.22435,0,.4563069024246581]);mass=float(m.body_mass.sum());weight=mass*9.81
cfg=g.ROOT/'low-level-as2piper-head-orin-deploy/legged_gym/envs/manip_loco/as2_head_orin_config.py'
protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [g.URDF,cfg,g.ROOT/'low-level-as2piper-head-orin-deploy/legged_gym/envs/manip_loco/manip_loco.py']}
rootxml=ET.parse(g.URDF).getroot();limits=np.array([float(rootxml.find("./joint[@name='%s']/limit"%m.joint(i).name).get('effort')) for i in g.leg_ids])
lo=np.r_[.20,np.deg2rad([-35,-35]),g.llo+1e-5,g.alo+1e-5];hi=np.r_[.42,np.deg2rad([35,50]),g.lhi-1e-5,g.ahi-1e-5]
home=np.r_[.3244069024246581,0,0,[0,.8,-1.6]*4,g.neutral];cached=home.copy()
def put(x):
 d.qpos[:3]=[0,0,x[0]];q=Rotation.from_euler('xyz',[x[1],x[2],0]).as_quat();d.qpos[3:7]=q[[3,0,1,2]]
 d.qpos[g.lq]=x[3:15];d.qpos[g.aq]=x[15:];g.fk()
def support():
 d.qvel[:]=0;mujoco.mj_forward(m,d)
 feet=d.xpos[g.feet].copy();com=d.subtree_com[bid].copy();hull=ConvexHull(feet[:,:2]);margin=float(-np.max(hull.equations[:,:2]@com[:2]+hull.equations[:,2]))
 A=np.vstack([np.ones(4),feet[:,0],feet[:,1]]);b=np.array([weight,weight*com[0],weight*com[1]])
 aub=np.c_[-np.eye(4),np.ones(4)];a_eq=np.c_[A,np.zeros(3)]
 result=linprog([0,0,0,0,-1],A_ub=aub,b_ub=np.zeros(4),A_eq=a_eq,b_eq=b,bounds=[(0,None)]*5,method='highs')
 out=dict(com_xyz=com.tolist(),com_margin_m=margin,front_edge_x_margin_m=float(feet[:2,0].mean()-com[0]),static_support_feasible=bool(result.success),foot_positions=feet.tolist())
 if result.success:
  lower=linprog([0,0,1,1],A_eq=A,b_eq=b,bounds=[(0,None)]*4,method='highs');upper=linprog([0,0,-1,-1],A_eq=A,b_eq=b,bounds=[(0,None)]*4,method='highs')
  out.update(max_possible_min_foot_N=float(result.x[4]),feasible_foot_force_N=result.x[:4].tolist(),rear_total_min_N=float(lower.fun) if lower.success else None,rear_total_max_N=float(-upper.fun) if upper.success else None)
  J=[]
  for body in g.feet:
   jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(m,d,jp,jr,body);J.append(jp[2,g.lv])
  J=np.array(J).T;tau=d.qfrc_bias[g.lv]
  au=np.vstack([aub,np.c_[-J,np.zeros(12)],np.c_[J,np.zeros(12)]]);bu=np.r_[np.zeros(4),limits-tau,limits+tau]
  capped=linprog([0,0,0,0,-1],A_ub=au,b_ub=bu,A_eq=a_eq,b_eq=b,bounds=[(0,None)]*5,method='highs')
  out['URDF_leg_effort_support_feasible']=bool(capped.success)
  out['URDF_effort_max_possible_min_foot_N']=float(capped.x[4]) if capped.success else None
 return out
put(home);baseline=support();jobs=[]
for r in [.35,.5,.65,.75,.8]:
 for phi in np.deg2rad([-72,-50,-30,-10,10,35,60]):
  for yaw in [-1.2,-.6,0,.6,1.2]:jobs.append((r,phi,yaw,[0,0,0],'grid'))
rng=np.random.default_rng(20261115)
for _ in range(80):jobs.append((rng.uniform(.35,.8),rng.uniform(-math.pi/2.5,math.pi/3),rng.uniform(-1.2,1.2),rng.uniform(-.08,.08,3).tolist(),'random'))
records=[];skipped=0
for k,(r,phi,yaw,delta,kind) in enumerate(jobs):
 target=center+r*np.array([np.cos(phi)*np.cos(yaw),np.cos(phi)*np.sin(yaw),np.sin(phi)])
 if target[2]<.02:skipped+=1;continue
 # Same nominal-standing body/head EE path box from initial target, no online IK gate.
 path=np.linspace([.427898468676838,.421686284147847,0],[r,phi,yaw],10)
 pts=np.c_[path[:,0]*np.cos(path[:,1])*np.cos(path[:,2]),path[:,0]*np.cos(path[:,1])*np.sin(path[:,2]),path[:,0]*np.sin(path[:,1])]+center
 local=pts-np.array([.22435,0,.4563069024246581])
 if np.any(np.all((local>[-.32435,-.112,-.1959])&(local<[.12165,.112,.0158]),axis=1)) or np.any(pts[:,2]<.02):skipped+=1;continue
 rot=Rotation.from_euler('ZYX',[yaw+delta[2],2.455216137744035-phi+delta[1],delta[0]]).as_matrix()
 def err(x):
  put(x);return np.r_[(d.xpos[g.feet]-g.foot_targets).ravel(),d.xpos[g.tcp]-target,.25*Rotation.from_matrix(rot@d.xmat[g.tcp].reshape(3,3).T).as_rotvec()]
 sols=[];best=None
 seeds=[cached,home,np.r_[.26,-np.sign(yaw)*np.deg2rad(20),np.deg2rad(30),[0,.8,-1.6]*4,[yaw,.7,-.6,0,.5,0]]]
 for seed in seeds:
  fit=least_squares(err,np.clip(seed,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=120,ftol=1e-9,xtol=1e-9,gtol=1e-9)
  e=err(fit.x);pe=float(np.linalg.norm(e[12:15]));ae=float(np.rad2deg(np.linalg.norm(e[15:])/.25));fe=float(np.max(np.abs(e[:12])));mujoco.mj_forward(m,d);hits=g.collisions()
  info=dict(position_error_m=pe,orientation_error_deg=ae,foot_error_m=fe,collision_hits=hits,qpos=d.qpos.tolist(),body_h_roll_pitch=fit.x[:3].tolist())
  score=pe+.25*np.deg2rad(ae)+fe+(.5 if hits else 0)
  if best is None or score<best[0]:best=(score,info)
  if pe<.003 and ae<2 and fe<.002 and not hits:
   info.update(support());sols.append(info);cached=fit.x.copy()
 record=dict(r=float(r),phi=float(phi),yaw=float(yaw),delta=list(delta),target=target.tolist(),kind=kind,solutions=sols,best_attempt=best[1]);records.append(record)
 if len(records)%10==0:
  print(len(records),'jobs',k+1,'valid',sum(bool(x['solutions']) for x in records),'sec',round(time.time()-start,1),flush=True)
  (O/'progress.json').write_text(json.dumps(records))
result=dict(center=center.tolist(),mass_kg=mass,weight_N=weight,baseline=baseline,records=records,raw_jobs=len(jobs),skipped_by_ground_or_box=skipped,protected_hashes=protected,wall_seconds=time.time()-start,leg_effort_model_Nm=limits.tolist(),scope='Static zero-payload, no DR, no inertial forces. CoM not used to reject IK solutions. Maximum minimum-foot load is support feasibility, not controller force prediction.')
for p,h in protected.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
(O/'scan.json').write_text(json.dumps(result,indent=2));print('DONE',len(records),'seconds',round(time.time()-start,1),flush=True)
