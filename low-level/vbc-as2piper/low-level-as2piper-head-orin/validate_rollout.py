"""Small physics/IK rollout without policy training or W&B."""
import argparse,json
from pathlib import Path
import isaacgym
from isaacgym import gymapi,gymutil,gymtorch
import numpy as np
import torch
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.envs.manip_loco.as2_head_orin_config import AS2HeadOrinCfg
from legged_gym.utils.helpers import class_to_dict

p=argparse.ArgumentParser();p.add_argument('--case',choices=['hold','ee','dr'],default='hold');p.add_argument('--seconds',type=float,default=12.);p.add_argument('--terrain',choices=['plane','trimesh'],default='plane');args=p.parse_args()
root=Path(__file__).resolve().parent
prefix=args.case if args.terrain=='plane' else args.case+'_rough'
np.random.seed(20261111);torch.manual_seed(20261111)
cfg=AS2HeadOrinCfg();cfg.env.num_envs=16;cfg.terrain.mesh_type=args.terrain;cfg.terrain.measure_heights=args.terrain=='trimesh'
if args.terrain=='trimesh':
 cfg.terrain.num_rows=4;cfg.terrain.num_cols=4;cfg.terrain.border_size=5;cfg.terrain.max_init_terrain_level=3
cfg.env.episode_length_s=args.seconds+5
if args.case!="dr":cfg.init_state.init_vel_perturb_range=0
cfg.asset.measure_dof_forces=True
if args.case!='dr':
 for key in ['randomize_friction','randomize_base_mass','randomize_base_com','randomize_motor','randomize_gripper_mass','push_robots']:setattr(cfg.domain_rand,key,False)
params=gymapi.SimParams();params.dt=cfg.sim.dt;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81);params.use_gpu_pipeline=True;params.physx.use_gpu=True
gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
env=ManipLoco(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True)
# Hold nominal legs to isolate support and arm IK. DR retains the original leg reset spread.
if args.case!='dr':
 def reset_dofs(ids):
  env.dof_pos[ids]=env.default_dof_pos;env.dof_vel[ids]=0
  env.gym.set_dof_state_tensor(env.sim,gymtorch.unwrap_tensor(env.dof_state))
 env._reset_dofs=reset_dofs
env.reset();samples=[];initial=env.check_termination
forces=gymtorch.wrap_tensor(env.gym.acquire_dof_force_tensor(env.sim)).view(env.num_envs,env.num_dofs)
force_samples=[]
arm=[i for i,name in enumerate(env.body_names) if name.startswith('piper_') and name not in ('piper_base_link','piper_link1')]
def check():
 initial();rp=env._get_body_orientation();h=env.root_states[:,2]-env.env_origins[:,2]
 env.gym.refresh_dof_force_tensor(env.sim)
 force_samples.append(forces.cpu().numpy().copy())
 err=(env.ee_pos-target).norm(dim=-1);angle=2*torch.acos((env.ee_orn*orientation).sum(-1).abs().clamp(0,1))
 samples.append(torch.cat([env.reset_buf[:,None].float(),h[:,None],rp,err[:,None],torch.rad2deg(angle)[:,None],env.contact_forces[:,env.feet_indices,2],env.torques[:,:12].abs().max(-1).values[:,None],env.contact_forces[:,arm].norm(dim=-1).max(-1).values[:,None]],dim=-1).cpu().numpy())
env.check_termination=check
try:
 with torch.no_grad():
  for step in range(round(args.seconds/env.dt)):
   env.commands.zero_()
   if args.case=='hold':env.goal_timer.zero_()
   target=env.curr_ee_goal_cart_world.clone();orientation=env.ee_goal_orn_quat.clone()
   env.step(torch.zeros(env.num_envs,18,device=env.device))
   assert torch.isfinite(env.obs_buf).all() and env.obs_buf.shape==(16,744)
   if step%100==0:print(args.case,step,'resets',sum(x[:,0].sum() for x in samples),flush=True)
 x=np.stack(samples);v=x[min(100,len(x)-1):]
 result={'terrain':args.terrain,'case':args.case,'envs':16,'seconds':args.seconds,'controller':'zero leg policy actions + original VBC EE DLS position drive; no learned policy','resets':int(x[:,:,0].sum()),'height_mean_m':float(v[:,:,1].mean()),'roll_pitch_p95_deg':float(np.rad2deg(np.quantile(np.abs(v[:,:,2:4]),.95))),'ee_position_mean_m':float(v[:,:,4].mean()),'ee_position_p95_m':float(np.quantile(v[:,:,4],.95)),'ee_angle_p95_deg':float(np.quantile(v[:,:,5],.95)),'feet_loaded_fraction':(v[:,:,6:10]>5).mean((0,1)).tolist(),'leg_torque_peak_Nm':float(v[:,:,10].max()),'arm_contact_peak_N':float(v[:,:,11].max()),'observation_shape':list(env.obs_buf.shape),'dof_order':env.dof_names,'rigid_bodies':env.body_names}
 measured=np.stack(force_samples)
 result['arm_measured_effort_peak_abs_Nm']=np.abs(measured[:,:,12:18]).max((0,1)).tolist()
 result['arm_effort_limit_Nm']=env.torque_limits[12:18].cpu().tolist()
 result['arm_effort_saturation_fraction']=float((np.abs(measured[:,:,12:18])>=.98*env.torque_limits[12:18].cpu().numpy()).mean())
 result['reset_events']=[{'time_s':float(t*env.dt),'env':int(i)} for t,i in np.argwhere(x[:,:,0]>.5)]
 np.savez_compressed(root/'validation'/f'{prefix}_rollout.npz',samples=x,measured_dof_effort=measured)
 (root/'validation'/f'{prefix}_rollout.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
finally:env.gym.destroy_sim(env.sim)
