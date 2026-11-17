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
root=Path(__file__).resolve().parent;out=root/'validation/gait_clock'/args.label;out.mkdir(parents=True,exist_ok=True)
random.seed(20261111);np.random.seed(20261111);torch.manual_seed(20261111)
cfg=AS2HeadOrinCfg();cfg.env.num_envs=16;cfg.terrain.num_rows=cfg.terrain.num_cols=4;cfg.terrain.border_size=5;cfg.env.episode_length_s=20
for name in ('randomize_friction','randomize_base_mass','randomize_base_com','randomize_motor','randomize_gripper_mass','push_robots'):setattr(cfg.domain_rand,name,False)
cfg.init_state.init_vel_perturb_range=0
params=gymapi.SimParams();params.dt=cfg.sim.dt;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81);params.use_gpu_pipeline=True;params.physx.use_gpu=True;gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
env=ManipLoco(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True)
targets=torch.tensor([[0.,0.,0.],[.3,0.,0.],[0.,0.,.6],[.3,0.,.6]],device=env.device).repeat_interleave(4,dim=0)
env._resample_commands=lambda ids:env.commands.__setitem__(ids,targets[ids])
env.reset()
result={'observation_shape':list(env.obs_buf.shape),'reset_return_history_zero':bool((env.obs_buf[:,85:]==0).all()),'internal_reset_history_filled':bool(torch.allclose(env.obs_history_buf,env.obs_buf[:,:67,None].transpose(1,2).expand(-1,10,-1))), 'normalization':class_to_dict(cfg.normalization),'privileged_order':'added_base_mass, COM xyz delta, gripper added mass, friction, 12 native-order leg motor-strength offsets'}
actions=torch.zeros(16,18,device=env.device);actions[:,12:]=.37;saved=actions.clone();env.step(actions)
result['caller_action_mutated']=not bool(torch.equal(actions,saved));result['applied_arm_actions_max']=float(env.actions[:,12:].abs().max())
# Compare the actual active reward calls with their analytic expressions.
shape_samples={}
for name in ('tracking_contacts_shaped_force','tracking_contacts_shaped_vel'):
    index=env.reward_names.index(name);fn=env.reward_functions[index];shape_samples[name]=[]
    def capture(fn=fn,name=name):
        raw,metric=fn()
        desired=env.desired_contact_states
        if name.endswith('force'):
            value=env.contact_forces[:,env.feet_indices].norm(dim=-1)
            expected=((1-desired)*(1-torch.exp(-value.square()/cfg.rewards.gait_force_sigma))).mean(-1)
        else:
            value=env.rigid_body_state[:,env.feet_indices,7:10].norm(dim=-1)
            expected=(desired*(1-torch.exp(-value.square()/cfg.rewards.gait_vel_sigma))).mean(-1)
        assert torch.allclose(raw,expected,atol=1e-6)
        assert (raw>=0).all() and (raw*env.reward_scales[name]<=0).all()
        shape_samples[name].append(raw.detach().cpu().numpy().copy())
        return raw,metric
    env.reward_functions[index]=capture
rows=[];observations=[];rewards=[];dones=[];phases=[]
try:
    with torch.no_grad():
        for step in range(200):
            history=env.obs_history_buf.clone();current=env.obs_buf.clone()
            env.commands.copy_(targets);env.step(torch.zeros(16,18,device=env.device))
            assert env.obs_buf.shape==(16,755) and torch.isfinite(env.obs_buf).all()
            expected_priv=torch.cat([env.mass_params_tensor,env.friction_coeffs_tensor,env.motor_strength[:,:12]-1],dim=1)
            assert torch.allclose(env.obs_buf[:,67:85],expected_priv)
            if not env.reset_buf.any():assert torch.allclose(env.obs_buf[:,85:],history.reshape(16,-1))
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
            assert torch.equal(env.obs_buf[:,62],env.gait_indices)
            assert torch.equal(env.obs_buf[:,63:67],env.clock_inputs)
            filtered=env._reindex_feet(env.contact_filt)
            instant=env._reindex_feet(env.contact_forces[:,env.feet_indices].norm(dim=-1)>2)
            sensor=env._reindex_feet(env.force_sensor_tensor.norm(dim=-1)>1.5)
            rows.append(torch.stack([torch.zeros((),device=env.device),(sensor!=instant).float().mean(),env.reset_buf.float().sum()]).cpu().numpy())
            phases.append(env.obs_buf[:,62:67].cpu().numpy().copy())
            observations.append(current.cpu().numpy());rewards.append(torch.stack([env.rew_buf,env.arm_rew_buf],dim=-1).cpu().numpy());dones.append(env.reset_buf.cpu().numpy())
    array=np.stack(rows)
    result.update(steps=200,seconds=4,contact_inputs_present=False,sensor_vs_instant_contact_disagreement=float(array[100:,1].mean()),resets=int(array[:,2].sum()),history_layout_checked=True,joint_normalization_checked=True,privileged_layout_checked=True)
    phase_array=np.stack(phases)
    result['clock_changes_in_moving_commands']=bool(np.ptp(phase_array[:,4:,1:],axis=0).max()>.5)
    assert result['clock_changes_in_moving_commands']
    result['shaping_rollout']={}
    for name,samples in shape_samples.items():
        values=np.stack(samples)
        result['shaping_rollout'][name]={'scale':env.reward_scales[name],'conditions':{}}
        assert env.reward_scales[name]==-2.
        for i,label in enumerate(('stand','forward','turn','forward_turn')):
            v=values[:,i*4:(i+1)*4]
            result['shaping_rollout'][name]['conditions'][label]={'raw_mean':float(v.mean()),'weighted_mean':float(v.mean()*env.reward_scales[name]),'nonzero_fraction':float((v>1e-6).mean())}
        assert values.max()>1e-6
    # Isolated live-source and sign tests; do not change the real simulation.
    import copy
    from legged_gym.envs.rewards.maniploco_rewards import ManipLoco_rewards
    shadow=copy.copy(env);shadow.contact_forces=env.contact_forces.clone();shadow.rigid_body_state=env.rigid_body_state.clone()
    shadow.force_sensor_tensor=torch.full_like(env.force_sensor_tensor,float('nan'))
    shadow.foot_velocities=torch.full_like(env.foot_velocities,float('nan'))
    shadow.desired_contact_states=torch.zeros_like(env.desired_contact_states)
    reward=ManipLoco_rewards(shadow)
    shadow.contact_forces[:,env.feet_indices,:]=torch.tensor([0.,0.,50.],device=env.device)
    swing_force=reward._reward_tracking_contacts_shaped_force()[0]
    assert torch.allclose(swing_force,torch.ones_like(swing_force))
    shadow.desired_contact_states.fill_(1.)
    stance_force=reward._reward_tracking_contacts_shaped_force()[0]
    shadow.rigid_body_state[:,env.feet_indices,7:10]=torch.tensor([1.,0.,0.],device=env.device)
    stance_slip=reward._reward_tracking_contacts_shaped_vel()[0]
    assert not stance_force.any() and torch.isfinite(stance_slip).all() and (stance_slip>0).all()
    shadow.rigid_body_state[:,env.feet_indices,7:10]=0.
    no_slip=reward._reward_tracking_contacts_shaped_vel()[0]
    assert not no_slip.any()
    result['shaping_probes']={'swing_50N_weighted':float(swing_force.mean()*-2),'stance_50N_weighted':float(stance_force.mean()*-2),'stance_1m_s_weighted':float(stance_slip.mean()*-2),'stance_0m_s_weighted':float(no_slip.mean()*-2),'sensor_nan_and_stale_velocity_nan_independent':True}
    assert cfg.rewards.scales.foot_contacts_z==0 and 'foot_contacts_z' not in env.reward_names
    result['sensor_only_foot_contacts_z_disabled']=True
    # A disabled privileged observation flag must still produce fresh correctly sized data.
    saved_q=env.dof_pos.clone();old_obs=env.obs_buf.clone();cfg.domain_rand.observe_priv=False
    env.dof_pos[:,0]+=.01;env.compute_observations()
    result['observe_priv_false_fresh']=bool((env.obs_buf[:,5:23]-old_obs[:,5:23]).abs().max()>.005)
    result['observe_priv_false_placeholder_zero']=bool((env.obs_buf[:,67:85]==0).all())
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
    assert runner.alg.storage.observations.shape[-1]==755
    assert not runner.alg.optimizer.state
    result['actual_runner_storage_shape']=list(runner.alg.storage.observations.shape)
    result['actual_runner_inference_action_shape']=list(actual_action.shape)
    result['actual_runner_optimizer_state_entries']=len(runner.alg.optimizer.state)
    result['dimensions']={'current_proprio':67,'privileged':18,'history':670,'total':755,'deployment':737}
    np.savez_compressed(out/'real_rollout.npz',observations=np.stack(observations),rewards=np.stack(rewards),dones=np.stack(dones),contact_checks=array)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
finally:env.gym.destroy_sim(env.sim)
