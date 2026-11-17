"""Observation/history and applied-action checks without policy learning."""
import argparse,json,random
from pathlib import Path
import isaacgym
from isaacgym import gymapi,gymutil
from isaacgym.torch_utils import quat_apply,quat_rotate_inverse
import torch,numpy as np
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.envs.manip_loco.as2_head_orin_config import AS2HeadOrinCfg
from legged_gym.utils.helpers import class_to_dict

parser=argparse.ArgumentParser();parser.add_argument('--label',default='before');args=parser.parse_args()
root=Path(__file__).resolve().parent;out=root/'validation/observation_ppo'/args.label;out.mkdir(parents=True,exist_ok=True)
random.seed(20261111);np.random.seed(20261111);torch.manual_seed(20261111)
cfg=AS2HeadOrinCfg();cfg.env.num_envs=16;cfg.terrain.num_rows=cfg.terrain.num_cols=4;cfg.terrain.border_size=5;cfg.env.episode_length_s=20
for name in ('randomize_friction','randomize_base_mass','randomize_base_com','randomize_motor','randomize_gripper_mass','push_robots'):setattr(cfg.domain_rand,name,False)
cfg.init_state.init_vel_perturb_range=0
params=gymapi.SimParams();params.dt=cfg.sim.dt;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81);params.use_gpu_pipeline=True;params.physx.use_gpu=True;gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
env=ManipLoco(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True)
env._resample_commands=lambda ids:env.commands.__setitem__(ids,0.)
env.reset()
result={'observation_shape':list(env.obs_buf.shape),'reset_return_history_zero':bool((env.obs_buf[:,84:]==0).all()),'internal_reset_history_filled':bool(torch.allclose(env.obs_history_buf,env.obs_buf[:,:66,None].transpose(1,2).expand(-1,10,-1))), 'normalization':class_to_dict(cfg.normalization),'privileged_order':'added_base_mass, COM xyz delta, gripper added mass, friction, 12 native-order leg motor-strength offsets'}
actions=torch.zeros(16,18,device=env.device);actions[:,12:]=.37;saved=actions.clone();env.step(actions)
result['caller_action_mutated']=not bool(torch.equal(actions,saved));result['applied_arm_actions_max']=float(env.actions[:,12:].abs().max())
rows=[];observations=[];rewards=[];dones=[]
try:
    with torch.no_grad():
        for step in range(200):
            history=env.obs_history_buf.clone();current=env.obs_buf.clone()
            env.commands.zero_();env.step(torch.zeros(16,18,device=env.device))
            assert env.obs_buf.shape==(16,744) and torch.isfinite(env.obs_buf).all()
            expected_priv=torch.cat([env.mass_params_tensor,env.friction_coeffs_tensor,env.motor_strength[:,:12]-1],dim=1)
            assert torch.allclose(env.obs_buf[:,66:84],expected_priv)
            if not env.reset_buf.any():assert torch.allclose(env.obs_buf[:,84:],history.reshape(16,-1))
            expected_q=env._reindex_all(env.dof_pos-env.default_dof_pos)[:,:18]
            expected_dq=env._reindex_all(env.dof_vel*.05)[:,:18]
            assert torch.allclose(env.obs_buf[:,5:23],expected_q)
            assert torch.allclose(env.obs_buf[:,23:41],expected_dq)
            assert torch.allclose(env.obs_buf[:,41:53],env._reindex_all(env.action_history_buf[:,-1])[:,:12])
            assert torch.allclose(env.obs_buf[:,:2],env._get_body_orientation())
            assert torch.allclose(env.obs_buf[:,2:5],env.base_ang_vel*env.obs_scales.ang_vel)
            assert torch.allclose(env.obs_buf[:,57:60],env.commands[:,:3]*env.commands_scale)
            arm_origin=env.base_pos+quat_apply(env.base_quat,env.arm_base_offset)
            expected_goal=quat_rotate_inverse(env.base_quat,env.curr_ee_goal_cart_world-arm_origin)
            assert torch.allclose(env.obs_buf[:,60:63],expected_goal)
            assert not env.obs_buf[:,63:66].any()
            observed=env.obs_buf[:,53:57].bool();filtered=env._reindex_feet(env.contact_filt)
            instant=env._reindex_feet(env.contact_forces[:,env.feet_indices].norm(dim=-1)>2)
            sensor=env._reindex_feet(env.force_sensor_tensor.norm(dim=-1)>1.5)
            rows.append(torch.stack([(observed!=filtered).float().mean(),(sensor!=instant).float().mean(),env.reset_buf.float().sum()]).cpu().numpy())
            observations.append(current.cpu().numpy());rewards.append(torch.stack([env.rew_buf,env.arm_rew_buf],dim=-1).cpu().numpy());dones.append(env.reset_buf.cpu().numpy())
    array=np.stack(rows)
    result.update(steps=200,seconds=4,observation_vs_reward_contact_disagreement=float(array[100:,0].mean()),sensor_vs_instant_contact_disagreement=float(array[100:,1].mean()),resets=int(array[:,2].sum()),history_layout_checked=True,joint_normalization_checked=True,privileged_layout_checked=True)
    # A disabled privileged observation flag must still produce fresh correctly sized data.
    saved_q=env.dof_pos.clone();old_obs=env.obs_buf.clone();cfg.domain_rand.observe_priv=False
    env.dof_pos[:,0]+=.01;env.compute_observations()
    result['observe_priv_false_fresh']=bool((env.obs_buf[:,5:23]-old_obs[:,5:23]).abs().max()>.005)
    result['observe_priv_false_placeholder_zero']=bool((env.obs_buf[:,66:84]==0).all())
    env.dof_pos.copy_(saved_q);cfg.domain_rand.observe_priv=True
    # Exercise the existing zero-delay/one-step-delay schedule and disabled branch.
    original_delay=env.action_delay;original_steps=env.global_steps
    delay_results={}
    for label,delay,global_steps,expected in [('early',original_delay,0,.02),('late',original_delay,240001,0.),('disabled',-1,0,.02)]:
        env.action_delay=delay;env.global_steps=global_steps;env.action_history_buf.zero_()
        probe=torch.zeros(16,18,device=env.device);probe[:,:12]=.02;probe[:,12:]=.37
        retained=probe.clone();env.step(probe)
        assert torch.equal(probe,retained)
        assert torch.allclose(env.actions[:,:12],torch.full_like(env.actions[:,:12],expected))
        assert not env.actions[:,12:].any()
        assert torch.allclose(env._reindex_all(env.action_history_buf[:,-1])[:,:12],torch.full_like(probe[:,:12],.02))
        delay_results[label]={'applied_leg_action':float(env.actions[0,0]),'observed_latest_command':float(env.obs_buf[0,41])}
    env.action_delay=original_delay;env.global_steps=original_steps
    result['action_delay_branches']=delay_results
    result['all_current_observation_fields_checked']=True
    result['dimensions']={'current_proprio':66,'privileged':18,'history':660,'total':744,'deployment':726}
    np.savez_compressed(out/'real_rollout.npz',observations=np.stack(observations),rewards=np.stack(rewards),dones=np.stack(dones),contact_checks=array)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
finally:env.gym.destroy_sim(env.sim)
