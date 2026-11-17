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

parser=argparse.ArgumentParser();parser.add_argument('--label',default='after');args=parser.parse_args()
root=Path(__file__).resolve().parent;out=root/'validation/contact_removal'/args.label;out.mkdir(parents=True,exist_ok=True)
random.seed(20261111);np.random.seed(20261111);torch.manual_seed(20261111)
cfg=AS2HeadOrinCfg();cfg.env.num_envs=16;cfg.terrain.num_rows=cfg.terrain.num_cols=4;cfg.terrain.border_size=5;cfg.env.episode_length_s=20
for name in ('randomize_friction','randomize_base_mass','randomize_base_com','randomize_motor','randomize_gripper_mass','push_robots'):setattr(cfg.domain_rand,name,False)
cfg.init_state.init_vel_perturb_range=0
params=gymapi.SimParams();params.dt=cfg.sim.dt;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81);params.use_gpu_pipeline=True;params.physx.use_gpu=True;gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
env=ManipLoco(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True)
env._resample_commands=lambda ids:env.commands.__setitem__(ids,0.)
env.reset()
result={'observation_shape':list(env.obs_buf.shape),'reset_return_history_zero':bool((env.obs_buf[:,80:]==0).all()),'internal_reset_history_filled':bool(torch.allclose(env.obs_history_buf,env.obs_buf[:,:62,None].transpose(1,2).expand(-1,10,-1))), 'normalization':class_to_dict(cfg.normalization),'privileged_order':'added_base_mass, COM xyz delta, gripper added mass, friction, 12 native-order leg motor-strength offsets'}
actions=torch.zeros(16,18,device=env.device);actions[:,12:]=.37;saved=actions.clone();env.step(actions)
result['caller_action_mutated']=not bool(torch.equal(actions,saved));result['applied_arm_actions_max']=float(env.actions[:,12:].abs().max())
rows=[];observations=[];rewards=[];dones=[]
try:
    with torch.no_grad():
        for step in range(200):
            history=env.obs_history_buf.clone();current=env.obs_buf.clone()
            env.commands.zero_();env.step(torch.zeros(16,18,device=env.device))
            assert env.obs_buf.shape==(16,700) and torch.isfinite(env.obs_buf).all()
            expected_priv=torch.cat([env.mass_params_tensor,env.friction_coeffs_tensor,env.motor_strength[:,:12]-1],dim=1)
            assert torch.allclose(env.obs_buf[:,62:80],expected_priv)
            if not env.reset_buf.any():assert torch.allclose(env.obs_buf[:,80:],history.reshape(16,-1))
            expected_q=env._reindex_all(env.dof_pos-env.default_dof_pos)[:,:18]
            expected_dq=env._reindex_all(env.dof_vel*.05)[:,:18]
            assert torch.allclose(env.obs_buf[:,5:23],expected_q)
            assert torch.allclose(env.obs_buf[:,23:41],expected_dq)
            assert torch.allclose(env.obs_buf[:,41:53],env._reindex_all(env.action_history_buf[:,-1])[:,:12])
            assert torch.allclose(env.obs_buf[:,:2],env._get_body_orientation())
            assert torch.allclose(env.obs_buf[:,2:5],env.base_ang_vel*env.obs_scales.ang_vel)
            assert torch.allclose(env.obs_buf[:,53:56],env.commands[:,:3]*env.commands_scale)
            arm_origin=env.base_pos+quat_apply(env.base_quat,env.arm_base_offset)
            expected_goal=quat_rotate_inverse(env.base_quat,env.curr_ee_goal_cart_world-arm_origin)
            assert torch.allclose(env.obs_buf[:,56:59],expected_goal)
            assert not env.obs_buf[:,59:62].any()
            filtered=env._reindex_feet(env.contact_filt)
            instant=env._reindex_feet(env.contact_forces[:,env.feet_indices].norm(dim=-1)>2)
            sensor=env._reindex_feet(env.force_sensor_tensor.norm(dim=-1)>1.5)
            rows.append(torch.stack([torch.zeros((),device=env.device),(sensor!=instant).float().mean(),env.reset_buf.float().sum()]).cpu().numpy())
            observations.append(current.cpu().numpy());rewards.append(torch.stack([env.rew_buf,env.arm_rew_buf],dim=-1).cpu().numpy());dones.append(env.reset_buf.cpu().numpy())
    array=np.stack(rows)
    result.update(steps=200,seconds=4,contact_inputs_present=False,sensor_vs_instant_contact_disagreement=float(array[100:,1].mean()),resets=int(array[:,2].sum()),history_layout_checked=True,joint_normalization_checked=True,privileged_layout_checked=True)
    # A disabled privileged observation flag must still produce fresh correctly sized data.
    saved_q=env.dof_pos.clone();old_obs=env.obs_buf.clone();cfg.domain_rand.observe_priv=False
    env.dof_pos[:,0]+=.01;env.compute_observations()
    result['observe_priv_false_fresh']=bool((env.obs_buf[:,5:23]-old_obs[:,5:23]).abs().max()>.005)
    result['observe_priv_false_placeholder_zero']=bool((env.obs_buf[:,62:80]==0).all())
    env.dof_pos.copy_(saved_q);cfg.domain_rand.observe_priv=True
    # Prove no contact/sensor values can enter either current input or history.
    reference_history=env.obs_history_buf.clone()
    env.compute_observations();reference=env.obs_buf.clone()
    saved_sensor=env.force_sensor_tensor.clone();saved_bits=env.foot_contacts_from_sensor.clone()
    env.obs_history_buf.copy_(reference_history)
    env.force_sensor_tensor.fill_(float('nan'));env.foot_contacts_from_sensor.logical_not_()
    env.compute_observations()
    assert torch.equal(env.obs_buf,reference)
    env.force_sensor_tensor.copy_(saved_sensor);env.foot_contacts_from_sensor.copy_(saved_bits)
    result['contact_sensor_poison_independence']=True
    # All active contact rewards use simulation contact state, not the six-axis sensor.
    import inspect
    active_names=[n for n in env.reward_names if any(k in n for k in ('contact','feet','collision'))]
    result['active_contact_reward_functions']=active_names
    for n in active_names:
        assert 'force_sensor_tensor' not in inspect.getsource(getattr(env.reward_container,'_reward_'+n))
    result['active_contact_rewards_no_sensor_input']=True
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
    # Instantiate the real runner factory path without calling learn/update.
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner
    from legged_gym.envs.manip_loco.as2_head_orin_config import AS2HeadOrinCfgPPO
    runner=OnPolicyRunner(env,class_to_dict(AS2HeadOrinCfgPPO()),device=env.device)
    policy=runner.get_inference_policy(device=env.device)
    with torch.no_grad():
        actual_action=policy(env.get_observations())
    assert actual_action.shape==(16,18) and torch.isfinite(actual_action).all()
    assert runner.alg.storage.observations.shape[-1]==700
    assert not runner.alg.optimizer.state
    result['actual_runner_storage_shape']=list(runner.alg.storage.observations.shape)
    result['actual_runner_inference_action_shape']=list(actual_action.shape)
    result['actual_runner_optimizer_state_entries']=len(runner.alg.optimizer.state)
    result['dimensions']={'current_proprio':62,'privileged':18,'history':620,'total':700,'deployment':682}
    np.savez_compressed(out/'real_rollout.npz',observations=np.stack(observations),rewards=np.stack(rewards),dones=np.stack(dones),contact_checks=array)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
finally:env.gym.destroy_sim(env.sim)
