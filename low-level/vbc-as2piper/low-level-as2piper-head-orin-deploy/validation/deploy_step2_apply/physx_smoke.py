"""CPU environment integration only, without policy/optimizer/training."""
import isaacgym
from isaacgym import gymapi,gymutil
import torch,numpy as np,json
from pathlib import Path
from legged_gym.envs.manip_loco.as2_head_orin_config import AS2HeadOrinCfg
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.utils.helpers import class_to_dict,set_seed
O=Path(__file__).parent
cfg=AS2HeadOrinCfg();cfg.env.num_envs=4;cfg.terrain.mesh_type='plane';cfg.terrain.curriculum=False;cfg.commands.curriculum=False
# Isolate nominal command plumbing in this test only; production DR untouched.
for key in ['randomize_base_mass','randomize_base_com','randomize_gripper_mass','randomize_friction','push_robots','randomize_motor']:setattr(cfg.domain_rand,key,False)
params=gymapi.SimParams();gymutil.parse_sim_config(class_to_dict(cfg.sim),params);params.use_gpu_pipeline=False;params.physx.use_gpu=False;params.physx.num_threads=2
set_seed(123)
e=ManipLoco(cfg,params,gymapi.SIM_PHYSX,'cpu',True);ids=torch.arange(4);e.reset_idx(ids,start=True)
s=e.feasible_workspace;assert s is not None
props=[e.gym.get_actor_dof_properties(env,actor) for env,actor in zip(e.envs,e.actor_handles)]
assert all(np.allclose(p['stiffness'][12:18],400) and np.allclose(p['damping'][12:18],40) for p in props)
adj=np.zeros((len(s.spheres),)*2,dtype=bool)
for edge in s.bank['edges']:a,b=edge['source'],edge['target'];adj[a,b]=adj[b,a]=True
# Exercise the actual environment resampler, not only its utility class.
def forbidden(*args):raise AssertionError('Unexpected online collision/rejection call')
original_check=e._collision_check;e._collision_check=forbidden
for _ in range(100):
 previous=s.current.clone();old=e.ee_goal_sphere.clone();e._resample_ee_goal(ids)
 assert adj[previous.numpy(),s.current.numpy()].all()
 assert torch.equal(e.ee_start_sphere,old)
 assert torch.equal(e.ee_goal_sphere,s.spheres[s.current])
 assert not e.ee_goal_orn_delta_rpy.any()
e._collision_check=original_check
e.reset_idx(ids,start=True);assert (s.current==0).all();assert torch.equal(e.curr_ee_goal_sphere,e.init_start_ee_sphere.expand(4,-1))
e.commands[:]=0;e._resample_commands=lambda ids:None
max_slew=0.;reset_count=0;sampled=set();last=e.arm_command_filter.last.clone();max_error=0.;reset_reasons={}
for step in range(300):
 _,_,reward,arm_reward,done,_=e.step(torch.zeros(4,18))
 assert torch.isfinite(e.obs_buf).all() and torch.isfinite(e.dof_state).all() and torch.isfinite(e.ee_goal_orn_quat).all()
 assert torch.isfinite(reward).all() and torch.isfinite(arm_reward).all()
 active=~done.bool()
 if active.any():max_slew=max(max_slew,float((e.arm_position_targets[active]-last[active]).abs().max()/e.dt))
 last=e.arm_command_filter.last.clone();reset_count+=int(done.sum());sampled.update(s.current.tolist())
 max_error=max(max_error,float((e.ee_pos-e.curr_ee_goal_cart_world).norm(dim=1).max()))
 if done.any():
  for k,v in e.termination_reasons.items():reset_reasons[k]=reset_reasons.get(k,0)+int(v[done.bool()].sum())
 assert e.obs_buf.shape==(4,755) and e.dof_pos.shape==(4,20)
assert max_slew<3.0001
result=dict(passed=True,backend='PhysX CPU',envs=4,steps=300,seconds=300*e.dt,obs=755,action=18,dof=20,optimizer_updates=0,
 resample_calls_checked=100,online_collision_check_called=False,actual_arm_stiffness=400,actual_arm_damping=40,
 max_command_slew_rad_s=max_slew,reset_count=reset_count,reset_reasons=reset_reasons,visited_goal_ids=sorted(sampled),
 max_ee_error_m=max_error,note='Zero-action untrained integration rollout, NOT locomotion/EE performance or workspace dynamic certification. Production reward/PPO/DR unchanged.')
(O/'physx_verification.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
e.gym.destroy_sim(e.sim)
