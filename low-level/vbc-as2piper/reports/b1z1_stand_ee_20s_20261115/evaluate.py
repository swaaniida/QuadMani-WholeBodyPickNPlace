"""CPU PhysX original B1+Z1 policy, 20s zero-velocity EE tracking."""
import json,hashlib,time,random,xml.etree.ElementTree as ET
from pathlib import Path
import isaacgym
from isaacgym import gymapi,gymutil
import torch,numpy as np
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg,B1Z1RoughCfgPPO
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.modules import ActorCritic
O=Path(__file__).resolve().parent;root=O.parents[1];seed=20261115
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
ckpt=Path('/data/team/users/suhwan/vbc-legacy/low-level/logs/b1z1-low/original-run01/model_45000.pt')
urdf=Path('/data/team/users/suhwan/vbc-legacy/low-level/resources/robots/b1z1/urdf/b1z1.urdf')
protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ckpt,urdf,Path('/data/team/users/suhwan/vbc-legacy/low-level/legged_gym/envs/manip_loco/manip_loco.py'),Path('/data/team/users/suhwan/vbc-legacy/low-level/legged_gym/envs/manip_loco/b1z1_config.py')]}
cfg=B1Z1RoughCfg();cfg.env.num_envs=4;cfg.env.observe_gait_commands=True;cfg.env.stop_update_goal=False;cfg.env.pitch_control=False;cfg.env.env_spacing=4;cfg.env.episode_length_s=60
cfg.terrain.num_rows=cfg.terrain.num_cols=2;cfg.terrain.border_size=3;cfg.terrain.height=[0.,0.]
for name in ('randomize_friction','randomize_base_mass','randomize_base_com','randomize_motor','randomize_gripper_mass','push_robots'):setattr(cfg.domain_rand,name,False)
params=gymapi.SimParams();gymutil.parse_sim_config(class_to_dict(cfg.sim),params);params.dt=cfg.sim.dt;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81);params.use_gpu_pipeline=False;params.physx.use_gpu=False;params.physx.num_threads=2
start=time.time();env=ManipLoco(cfg,params,gymapi.SIM_PHYSX,'cpu',True)
assert tuple(env.obs_buf.shape)==(4,799)
x=torch.load(str(ckpt),map_location='cpu');model=ActorCritic(71,71,18,**class_to_dict(B1Z1RoughCfgPPO.policy),num_priv=18,num_hist=10,num_prop=71);model.load_state_dict(x['model_state_dict'],strict=True);model.eval()
env._resample_commands=lambda ids:env.commands.__setitem__(ids,0.)
env.reset();env.global_steps=45000*24
# Original spherical bounds, target orientation law and IK. No new orientation law.
waypoints=[('front',[.65,.05,0],[0,0,0]),('left',[.65,.25,.70],[.15,-.12,.10]),('right',[.65,.25,-.70],[-.15,.12,-.10]),('upper',[.60,.70,0],[.1,-.1,0]),('near ground',[.80,-.95,0],[0,0,0]),('far low',[.85,-.50,.25],[-.12,.10,.12])]
feet_names=[env.body_names[int(i)] for i in env.feet_indices]
et=ET.parse(urdf).getroot();radii=[];visual_radii=[]
for name in feet_names:
 link=et.find("./link[@name='%s']"%name);collision=link.find('collision');assert collision.find('origin') is None or np.allclose([float(v) for v in collision.find('origin').get('xyz','0 0 0').split()],0)
 radii.append(float(collision.find('geometry/sphere').get('radius')));visual_radii.append(float(link.find('visual/geometry/sphere').get('radius')))
radii=np.array(radii)
# Validate each chosen endpoint with original geometric checker; never bypass it.
for _,sp,_ in waypoints:
 env.ee_start_sphere[:]=torch.tensor(sp);env.ee_goal_sphere[:]=torch.tensor(sp)
 assert not env._collision_check(torch.arange(4)).any(),sp
home=torch.tensor(cfg.goal_ee.ranges.init_pos_end)
env.ee_start_sphere[:]=home;env.ee_goal_sphere[:]=home;env.goal_timer.zero_();env.traj_timesteps.fill_(1);env.traj_total_timesteps.fill_(1e9);env.ee_goal_orn_delta_rpy.zero_()
records=[];phase=-1
try:
 with torch.no_grad():
  for tick in range(round(22/env.dt)):
   t=tick*env.dt-2
   if t>=0:
    nextphase=min(5,int(t/(20/6)))
    if nextphase!=phase:
     phase=nextphase;label,sp,delta=waypoints[phase]
     env.ee_start_sphere[:]=env.curr_ee_goal_sphere.clone();env.ee_goal_sphere[:]=torch.tensor(sp)
     assert not env._collision_check(torch.arange(4)).any(),('transition',phase)
     env.ee_goal_orn_delta_rpy[:]=torch.tensor(delta);env.goal_timer.zero_();env.traj_timesteps.fill_(2.3/env.dt);env.traj_total_timesteps.fill_(1e9)
   env.commands.zero_();actions=model.act_inference(env.obs_buf,hist_encoding=True);env.step(actions.clone());env.commands.zero_()
   if t>=-1e-6:
    origin=env.env_origins.cpu().numpy();base=env.root_states[:,:7].cpu().numpy().copy();base[:,:3]-=origin
    foot=env.rigid_body_state[:,env.feet_indices,:3].cpu().numpy().copy()-origin[:,None,:]
    target=env.curr_ee_goal_cart_world.cpu().numpy().copy()-origin;actual=env.ee_pos.cpu().numpy().copy()-origin
    forces=env.contact_forces[:,env.feet_indices].cpu().numpy().copy()
    records.append(dict(time=t+env.dt,phase=phase,root=base,q=env.dof_pos.cpu().numpy().copy(),target=target,actual=actual,feet=foot,forces=forces,v=env.base_lin_vel.cpu().numpy().copy(),w=env.base_ang_vel.cpu().numpy().copy(),gravity=env.projected_gravity.cpu().numpy().copy(),done=env.reset_buf.cpu().numpy().copy(),timeout=env.time_out_buf.cpu().numpy().copy()))
   if tick%250==0:print('tick',tick,'wall',round(time.time()-start,1),'phase',phase,flush=True)
 def stack(name):return np.stack([a[name] for a in records])
 arrays=dict(time_s=np.array([a['time'] for a in records]),phase_id=np.array([a['phase'] for a in records]),root_xyzw=stack('root'),dof_pos=stack('q'),ee_target_world=stack('target'),ee_actual_world=stack('actual'),foot_positions=stack('feet'),foot_forces=stack('forces'),base_lin_vel=stack('v'),base_ang_vel=stack('w'),projected_gravity=stack('gravity'),done=stack('done'),timeout=stack('timeout'))
 arrays['ee_position_error']=np.linalg.norm(arrays['ee_target_world']-arrays['ee_actual_world'],axis=-1);arrays['foot_clearance']=arrays['foot_positions'][...,2]-radii[None,None,:]
 assert len(records)==round(20/env.dt) and all(np.isfinite(v).all() for v in arrays.values())
 np.savez_compressed(O/'states.npz',**arrays)
 def longest(mask):
  padded=np.r_[False,mask,False];edges=np.flatnonzero(padded[1:]!=padded[:-1]);return float(np.max(edges[1::2]-edges[::2])*env.dt) if len(edges) else 0.
 summary=[]
 for eid in range(4):
  metrics=[]
  for j,name in enumerate(feet_names):
   fz=arrays['foot_forces'][:,eid,j,2];gap=arrays['foot_clearance'][:,eid,j];contact=fz>2
   rolling=np.array([gap[max(0,i-24):i+1].min() for i in range(len(gap))])
   metrics.append(dict(foot=name,contact_Fz_gt2_fraction=float(contact.mean()),contact_force_norm_gt2_fraction=float((np.linalg.norm(arrays['foot_forces'][:,eid,j],axis=-1)>2).mean()),Fz_mean_N=float(fz.mean()),clearance_mean_mm=float(gap.mean()*1000),clearance_p95_mm=float(np.quantile(gap,.95)*1000),clearance_max_mm=float(gap.max()*1000),longest_gap20mm_and_no_contact_s=longest((gap>.02)&~contact),longest_no_Fz_contact_s=longest(~contact),persistent_min_gap5mm_last_halfsec_fraction=float((rolling[24:]>.005).mean())))
  summary.append(dict(env=eid,feet=metrics,ee_error_mean_m=float(arrays['ee_position_error'][:,eid].mean()),ee_error_p95_m=float(np.quantile(arrays['ee_position_error'][:,eid],.95)),mean_vx=float(arrays['base_lin_vel'][:,eid,0].mean()),mean_vy=float(arrays['base_lin_vel'][:,eid,1].mean()),mean_yaw_rate=float(arrays['base_ang_vel'][:,eid,2].mean()),reset_count=int(arrays['done'][:,eid].sum()),upright_fraction=float((np.linalg.norm(arrays['projected_gravity'][:,eid,:2],axis=-1)<.5).mean())))
 meta=dict(checkpoint=str(ckpt),iteration=int(x['iter']),seed=seed,seconds=20,warmup_seconds=2,dt=float(env.dt),envs=4,command=[0,0,0],device='CPU PhysX + CPU deterministic history-encoder policy',observation=799,DR_push=False,terrain='original flat trimesh, height=0',episode_length_s=60,foot_names=feet_names,collision_radius_m=radii.tolist(),visual_radius_m=visual_radii,waypoints=waypoints,dof_names=env.dof_names,summary=summary,protected_hashes=protected,wall_seconds=time.time()-start,video_env=0,note='Original code and policy unchanged; only evaluation command sequence, no DR/push and timeout extension. Net-force measurements are simulation ground truth. Render is state replay, not MuJoCo re-simulation.')
 for p,h in protected.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 (O/'metrics.json').write_text(json.dumps(meta,indent=2));print(json.dumps(summary,indent=2),flush=True)
finally:env.gym.destroy_sim(env.sim)
