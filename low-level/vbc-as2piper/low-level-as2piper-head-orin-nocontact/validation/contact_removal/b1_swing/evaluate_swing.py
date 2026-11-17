"""Read-only original B1 policy evaluation; independent reward and input ablation."""
import argparse,hashlib,json,random,time
from pathlib import Path
import isaacgym
from isaacgym import gymapi,gymutil
import torch,numpy as np
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg,B1Z1RoughCfgPPO
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.modules import ActorCritic

parser=argparse.ArgumentParser();parser.add_argument('--terrain',choices=['flat','rough'],required=True);parser.add_argument('--ablate',action='store_true');parser.add_argument('--seconds',type=float,default=40);args=parser.parse_args()
out=Path(__file__).resolve().parent/(args.terrain+('_zero44' if args.ablate else '_normal'));out.mkdir(exist_ok=True)
seed=20261111
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
cfg=B1Z1RoughCfg();cfg.env.num_envs=32;cfg.env.observe_gait_commands=True;cfg.env.stop_update_goal=False;cfg.env.pitch_control=False;cfg.env.env_spacing=3
cfg.terrain.num_rows=cfg.terrain.num_cols=4;cfg.terrain.border_size=5
if args.terrain=='flat':cfg.terrain.height=[0.,0.]
for name in ('randomize_friction','randomize_base_mass','randomize_base_com','randomize_motor','randomize_gripper_mass','push_robots'):setattr(cfg.domain_rand,name,False)
params=gymapi.SimParams();params.dt=cfg.sim.dt;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81);params.use_gpu_pipeline=True;params.physx.use_gpu=True;gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
env=ManipLoco(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True)
assert cfg.env.num_proprio==71 and env.obs_buf.shape==(32,799)
ckpt=Path('/data/team/users/suhwan/vbc-legacy/low-level/logs/b1z1-low/original-run01/model_45000.pt')
x=torch.load(str(ckpt),map_location=env.device)
model=ActorCritic(71,71,18,**class_to_dict(B1Z1RoughCfgPPO.policy),num_priv=18,num_hist=10,num_prop=71).to(env.device);model.load_state_dict(x['model_state_dict'],strict=True);model.eval()
commands=torch.tensor([[0.,0.,0.],[.4,0.,0.],[0.,0.,.6],[.4,0.,.6]],device=env.device).repeat_interleave(8,dim=0)
env._resample_commands=lambda ids:env.commands.__setitem__(ids,commands[ids])
env.reset();env.global_steps=45000*24 # checkpoint's post-delay operating regime
# Keep original IK and EE transforms. Supply a repeatable central-workspace trajectory.
def ee_at(step):
 t=step*env.dt
 goal=torch.tensor([.60+.06*np.sin(2*np.pi*t/8),.10+.12*np.sin(2*np.pi*t/10),.20*np.sin(2*np.pi*t/12)],device=env.device)
 env.ee_start_sphere[:]=goal;env.ee_goal_sphere[:]=goal;env.ee_goal_orn_delta_rpy.zero_();env.goal_timer.zero_();env.traj_timesteps.fill_(1);env.traj_total_timesteps.fill_(1e9)
 env._update_curr_ee_goal()
 env.commands[:]=commands
 env.compute_observations()
# Directly update only EE observation slots between steps, without shifting history twice.
def set_goal(step):
 t=step*env.dt
 goal=torch.tensor([.60+.06*np.sin(2*np.pi*t/8),.10+.12*np.sin(2*np.pi*t/10),.20*np.sin(2*np.pi*t/12)],device=env.device)
 env.ee_start_sphere[:]=goal;env.ee_goal_sphere[:]=goal;env.ee_goal_orn_delta_rpy.zero_();env.goal_timer.zero_();env.traj_timesteps.fill_(1);env.traj_total_timesteps.fill_(1e9)
 # post_physics_step updates command target using these original buffers.

def masked(obs):
 o=obs.clone();o[:,53:57]=0;o[:,89:].view(32,10,71)[:,:,53:57]=0;return o
terms={};scales={}
for branch,names,functions,weights in [('leg',env.reward_names,env.reward_functions,env.reward_scales),('arm',env.arm_reward_names,env.arm_reward_functions,env.arm_reward_scales)]:
 for i,(name,fn) in enumerate(zip(names,functions)):
  key=branch+'/'+name;terms[key]=[];scales[key]=float(weights[name])
  def capture(fn=fn,key=key):
   raw,metric=fn();v=torch.as_tensor(raw,device=env.device).expand(32);terms[key].append(v.detach().cpu().numpy().copy());return raw,metric
  functions[i]=capture
foot_heights=[];clock_targets=[];physical=[];sensors=[];forces=[];velocities=[];ages=[];leg_actions=[];offline=[];states=[]
original_compute=env.compute_reward

def capture_physics():
 original_compute()
 sensor=env.force_sensor_tensor.clone();force=env.contact_forces[:,env.feet_indices].clone();feet=env.rigid_body_state[:,env.feet_indices]
 xy=feet[:,:,:2]
 pixels=((xy+cfg.terrain.border_size)/cfg.terrain.horizontal_scale).long()
 gx=pixels[...,0].clamp(0,env.height_samples.shape[0]-2);gy=pixels[...,1].clamp(0,env.height_samples.shape[1]-2)
 ground=torch.minimum(torch.minimum(env.height_samples[gx,gy],env.height_samples[gx+1,gy]),env.height_samples[gx,gy+1])*cfg.terrain.vertical_scale
 foot_heights.append(torch.stack([feet[:,:,2],ground,feet[:,:,2]-ground,feet[:,:,2]-ground-.04],dim=-1).cpu().numpy())
 clock_targets.append(env.desired_contact_states.cpu().numpy().copy())
 sensors.append(sensor.cpu().numpy());forces.append(force.cpu().numpy());velocities.append(feet[:,:,7:10].cpu().numpy());ages.append(env.episode_length_buf.cpu().numpy().copy())
 err=(env.curr_ee_goal_cart_world-env.ee_pos).norm(dim=-1)
 orn=1-(env.ee_goal_orn_quat*env.ee_orn).sum(-1).abs().clamp(max=1)
 physical.append(torch.stack([env.base_lin_vel[:,0],env.base_lin_vel[:,1],env.base_ang_vel[:,2],err,orn,env.reset_buf.float(),env.time_out_buf.float(),env.rew_buf,env.arm_rew_buf,env.root_states[:,2]],dim=-1).cpu().numpy())
env.compute_reward=capture_physics
start=time.time()
try:
 with torch.no_grad():
  for step in range(int(args.seconds/env.dt)):
   set_goal(step);env.commands[:]=commands
   obs=env.obs_buf
   full=model.act_inference(obs,hist_encoding=True);zero=model.act_inference(masked(obs),hist_encoding=True)
   privfull=model.act_inference(obs,hist_encoding=False);privzero=model.act_inference(masked(obs),hist_encoding=False)
   offline.append(torch.stack([(full[:,:12]-zero[:,:12]).square().mean(-1).sqrt(),(full[:,12:]-zero[:,12:]).square().mean(-1).sqrt(),(privfull[:,:12]-privzero[:,:12]).square().mean(-1).sqrt(),full[:,:12].square().mean(-1).sqrt()],dim=-1).cpu().numpy())
   action=zero if args.ablate else full;leg_actions.append(action[:,:12].cpu().numpy().copy())
   if step==0:states.append({'initial_obs':obs.cpu().numpy().tolist(),'initial_root':env.root_states.cpu().numpy().tolist(),'initial_dof':env.dof_pos.cpu().numpy().tolist()})
   env.step(action.clone())
   if step%500==0:print('progress',step,'elapsed',round(time.time()-start,1),flush=True)
 data=np.stack(physical);sense=np.stack(sensors);force=np.stack(forces);vel=np.stack(velocities);age=np.stack(ages);actions=np.stack(leg_actions);counter=np.stack(offline);raw={k:np.stack(v) for k,v in terms.items()}
 weighted={k:v*scales[k] for k,v in raw.items()}
 assert np.isfinite(data).all() and np.isfinite(actions).all()
 np.savez_compressed(out/'samples.npz',foot_heights=np.stack(foot_heights),clock_targets=np.stack(clock_targets),physical=data,sensors=sense,forces=force,foot_velocities=vel,episode_age=age,actions=actions,offline_action=counter,**{k.replace('/','__'):v for k,v in raw.items()})
 summary={'terrain':args.terrain,'zero_contact_44':args.ablate,'checkpoint':str(ckpt),'checkpoint_iteration':x['iter'],'observation':799,'proprio':71,'privileged':18,'seconds':args.seconds,'envs':32,'warmup_seconds':2,'dt':env.dt,'inference':'deterministic history branch','conditions':{},'scales':scales,'wall_seconds':time.time()-start,'gait_active':cfg.env.observe_gait_commands,'action_delay':'one policy step, post-training regime','protocol':'original env/reward/policy; DR/push off; original random reset, 10s episodes; prescribed central EE sphere; flat=original trimesh with zero rough height; rough=original rough-flat height range 0..0.1m on 4x4 terrain','initial_state':states[0]}
 for i,name in enumerate(('stand','forward','turn','forward_turn')):
  ix=np.arange(i*8,(i+1)*8);sel=np.s_[100:,i*8:(i+1)*8];d=data[sel];sf=sense[sel];nf=force[sel];v=vel[sel];sn=np.linalg.norm(sf,axis=-1);nn=np.linalg.norm(nf,axis=-1);s3=np.linalg.norm(sf[...,:3],axis=-1);valid=age[sel]>2/env.dt
  total=sum(weighted[k][sel] for k in raw if k.startswith('leg/'));total_abs=sum(np.abs(weighted[k][sel]) for k in raw if k.startswith('leg/'))
  info={'command':commands[i*8].cpu().tolist(),'vx_mean':float(d[:,:,0].mean()),'vx_rmse':float(np.sqrt(((d[:,:,0]-float(commands[i*8,0]))**2).mean())),'yaw_mean':float(d[:,:,2].mean()),'yaw_rmse':float(np.sqrt(((d[:,:,2]-float(commands[i*8,2]))**2).mean())),'ee_error_mean_m':float(d[:,:,3].mean()),'ee_error_p95_m':float(np.quantile(d[:,:,3],.95)),'failure_resets':int((data[:,ix,5]*(1-data[:,ix,6])).sum()),'timeouts':int(data[:,ix,6].sum()),'leg_reward_mean_before_div100':float(total.mean()),'leg_abs_reward_mean':float(total_abs.mean()),'arm_reward_mean_before_div100':float(sum(weighted[k][sel] for k in raw if k.startswith('arm/')).mean()),'contact_disagreement_sensor_net':float(((sn>1.5)!=(nn>2)).mean()),'sensor6_threshold40_exceed_fraction_after_gate':float((sn[valid]>40).mean()) if valid.any() else None,'net_threshold40_exceed_fraction_after_gate':float((nn[valid]>40).mean()) if valid.any() else None,'offline_action_rms_delta_history':float(counter[sel][:,:,0].mean()),'offline_action_rms_delta_privileged':float(counter[sel][:,:,2].mean()),'normal_action_rms':float(counter[sel][:,:,3].mean()),'rewards':{}}
  for tag,arr in [('sensor6_mixed_units',sn),('sensor_force3_N',s3),('net_contact_force_N',nn)]:info[tag]={'mean_per_foot':arr.mean((0,1)).tolist(),'p95_per_foot':np.quantile(arr,.95,axis=(0,1)).tolist(),'p99_per_foot':np.quantile(arr,.99,axis=(0,1)).tolist(),'mean':float(arr.mean()),'p95':float(np.quantile(arr,.95)),'p99':float(np.quantile(arr,.99))}
  # Existing threshold and scale applied to net force is a counterfactual, not an intervention.
  netpen=(np.maximum(nn-40,0).sum(-1)*valid)*scales.get('leg/feet_contact_forces',0)
  info['same40N_same_scale_net_force_counterfactual_mean']=float(netpen.mean())
  for key,arr in raw.items():
   w=weighted[key][sel];denom=total if key.startswith('leg/') else sum(weighted[k][sel] for k in raw if k.startswith('arm/'));ab=sum(np.abs(weighted[k][sel]) for k in raw if k.split('/')[0]==key.split('/')[0])
   info['rewards'][key]={'raw_mean':float(arr[sel].mean()),'weighted_mean_before_div100':float(w.mean()),'signed_share_of_net_reward':float(w.mean()/denom.mean()) if abs(denom.mean())>1e-9 else None,'abs_share':float(np.abs(w).mean()/ab.mean()) if ab.mean()>1e-9 else None,'nonzero_fraction':float((arr[sel]!=0).mean())}
  summary['conditions'][name]=info
 summary['reward_sum_matches']=bool(np.allclose(sum(weighted[k] for k in raw if k.startswith('leg/'))/100,data[:,:,7],atol=1e-5))
 (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print('saved',out,summary['reward_sum_matches'],flush=True)
finally:env.gym.destroy_sim(env.sim)
