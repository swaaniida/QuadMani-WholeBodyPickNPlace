"""Four-environment CPU PhysX integration test; no policy/optimizer/training."""
import isaacgym
from isaacgym import gymapi,gymutil
import torch,numpy as np,json
from pathlib import Path
from legged_gym.envs.manip_loco.as2_head_orin_config import AS2HeadOrinCfg
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.utils.helpers import class_to_dict,set_seed
O=Path(__file__).parent
from types import SimpleNamespace
cfg0=AS2HeadOrinCfg();mock=SimpleNamespace(cfg=cfg0,device='cpu',goal_ee_ranges=class_to_dict(cfg0.goal_ee.ranges),ee_goal_sphere=torch.zeros(6144,3))
lo=torch.full((3,),float('inf'));hi=-lo.clone();zlo=1.;zhi=0.;xlo=2.;xhi=0.
for _ in range(20):
 ManipLoco._resample_ee_goal_sphere_once(mock,torch.arange(6144))
 q=mock.ee_goal_sphere;r0,p,y=q.T
 x=.22435+r0*torch.cos(p);z=.4563069024246581+r0*torch.sin(p)
 assert (x>=.45-1e-6).all() and (x<=.975+1e-6).all()
 assert (z>=.02-1e-6).all() and (z<=.85+1e-6).all()
 lo=torch.minimum(lo,q.min(0).values);hi=torch.maximum(hi,q.max(0).values)
 zlo=min(zlo,float(z.min()));zhi=max(zhi,float(z.max()));xlo=min(xlo,float(x.min()));xhi=max(xhi,float(x.max()))
assert zlo<.025 and zhi>.845 and xlo<.455 and xhi>.970
poses=json.loads((O.parents[2]/'reports/workspace_limits_20261115/measurements.json').read_text())['poses']
q=torch.tensor([p['sphere'] for p in poses]);pitch=ManipLoco._get_ee_goal_pitch(mock,q)
assert np.allclose(pitch.numpy(),[p['tool_rpy_rad'][1] for p in poses],atol=1e-6)
(O/'geometry_verification.json').write_text(json.dumps(dict(samples=20*6144,passed=True,sphere_min=lo.tolist(),sphere_max=hi.tolist(),ground_z_min=zlo,ground_z_max=zhi,nominal_x_min=xlo,nominal_x_max=xhi,six_rendered_goal_pitches_match=True,IK_checks=0,bank=False),indent=2))
cfg=AS2HeadOrinCfg();cfg.env.num_envs=4;cfg.terrain.mesh_type='plane';cfg.terrain.curriculum=False;cfg.commands.curriculum=False
cfg.domain_rand.randomize_base_mass=False;cfg.domain_rand.randomize_base_com=False;cfg.domain_rand.randomize_gripper_mass=False;cfg.domain_rand.randomize_friction=False;cfg.domain_rand.push_robots=False
cfg.domain_rand.randomize_motor=True;cfg.domain_rand.leg_motor_strength_range=[1.,1.];cfg.domain_rand.arm_motor_strength_range=[1.,1.]
params=gymapi.SimParams();gymutil.parse_sim_config(class_to_dict(cfg.sim),params);params.use_gpu_pipeline=False;params.physx.use_gpu=False;params.physx.num_threads=2
set_seed(123)
e=ManipLoco(cfg,params,gymapi.SIM_PHYSX,'cpu',True)
e.reset_idx(torch.arange(4), start=True)
props=[e.gym.get_actor_dof_properties(env,actor) for env,actor in zip(e.envs,e.actor_handles)]
s=e.motor_strength[:,12:18].numpy()
assert np.allclose(np.array([p['stiffness'][12:18] for p in props]),400*s)
assert np.allclose(np.array([p['damping'][12:18] for p in props]),40*s)
assert np.allclose(np.array([p['velocity'][12:18] for p in props]),3)
# Verify real continuous resampler calls collision checks but never control IK.
collision_calls=[0]
original_ik=e._control_ik;original_collision=e._collision_check
def forbidden_ik(*args):raise AssertionError('IK used for target feasibility')
def counted_collision(ids):
 collision_calls[0]+=1
 return original_collision(ids)
e._control_ik=forbidden_ik;e._collision_check=counted_collision
sampled=[];deltas=[]
for _ in range(100):
 e._resample_ee_goal(torch.arange(4))
 assert not original_collision(torch.arange(4)).any()
 sampled.append(e.ee_goal_sphere.clone());deltas.append(e.ee_goal_orn_delta_rpy.clone())
# Force ten failed geometric draws; the last rejected target must not leak.
old_goal=e.ee_goal_sphere.clone();old_orn=e.ee_goal_orn_delta_rpy.clone()
e._collision_check=lambda ids:torch.ones(len(ids),dtype=torch.bool)
e._resample_ee_goal(torch.arange(4))
assert torch.equal(e.ee_goal_sphere,old_goal) and torch.equal(e.ee_goal_orn_delta_rpy,old_orn)
e._control_ik=original_ik;e._collision_check=original_collision
sampled=torch.stack(sampled);deltas=torch.stack(deltas)
assert collision_calls[0]>=100
for k,bounds in enumerate([[.22565,.8682391004806224],[-1.0934979298811465,1.0503443030006583],[-.30,.30]]):
 assert sampled[:,:,k].min()>=bounds[0]-1e-6 and sampled[:,:,k].max()<=bounds[1]+1e-6
assert deltas.min()>=-.080001 and deltas.max()<=.080001 and deltas.std()>0
assert not hasattr(e,'feasible_workspace')
e.reset_idx(torch.arange(4),start=True)
# Freeze command goals at home, no walking; untrained zero-action hold.
e.commands[:]=0;e._resample_commands=lambda ids:None
last=e.arm_command_filter.last.clone();max_slew=0.;resets=0;velmax=0.;poses=[];reset_events=[]
for t in range(100):
 _,_,reward,arm_reward,done,_=e.step(torch.zeros(4,18))
 assert torch.isfinite(e.obs_buf).all() and torch.isfinite(e.dof_state).all()
 assert torch.isfinite(reward).all() and torch.isfinite(arm_reward).all()
 active=~done.bool()
 if done.any():reset_events.append(dict(step=t,reasons={k:v.tolist() for k,v in e.termination_reasons.items()}))
 if active.any():max_slew=max(max_slew,float((e.arm_position_targets[active]-last[active]).abs().max()/e.dt))
 last=e.arm_command_filter.last.clone();resets+=int(done.sum());velmax=max(velmax,float(e.dof_vel[:,12:18].abs().max()))
 if t%25==0:poses.append(e.root_states[:,:3].tolist())
 assert e.obs_buf.shape==(4,755)
assert max_slew<=3.0001
result=dict(passed=True,backend='PhysX CPU',envs=4,steps=100,seconds=100*e.dt,obs=755,action=18,dof=20,optimizer_updates=0,
 max_command_slew_rad_s=max_slew,max_measured_joint_velocity_rad_s=velmax,reset_count=resets,reset_events=reset_events,
 actual_arm_stiffness=np.array([p['stiffness'][12:18] for p in props]).tolist(),actual_arm_damping=np.array([p['damping'][12:18] for p in props]).tolist(),
 motor_strength=s.tolist(),resampler_collision_calls=collision_calls[0],resampler_ik_calls=0,continuous_sample_count=400,bank_attributes_present=False,exhausted_retries_preserve_previous_goal_and_orientation=True,all_finite=True,base_positions=poses,note='Short untrained zero-action hold; no calibrated arm model, locomotion success or deployment claim.')
(O/'physx_verification.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
e.gym.destroy_sim(e.sim)
