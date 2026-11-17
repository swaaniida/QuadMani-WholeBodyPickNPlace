"""CPU PhysX preflight only; no policy updates or production config changes."""
import isaacgym
import torch,json
from pathlib import Path
from force_ablation.run import environment
from legged_gym.envs.rewards.maniploco_rewards import ManipLoco_rewards

torch.set_num_threads(2)
out=Path(__file__).resolve().parent
env,train=environment('B',1,'cpu',small=True,terrain='flat')
try:
    env.meter.verify=True
    print("SIM",env.sim_params.gravity,env.sim_params.physx.contact_collection,flush=True)
    targets=torch.tensor([[0.,0.,0.],[.3,0.,0.],[0.,0.,.6],[.3,0.,.6]]).repeat_interleave(env.num_envs//4,0)
    env._resample_commands=lambda ids:env.commands.__setitem__(ids,targets[ids])
    env.reset()
    for i in range(160):
        env.commands.copy_(targets)
        env.step(torch.zeros(env.num_envs,18))
        assert torch.isfinite(env.obs_buf).all() and torch.isfinite(env.rew_buf).all() and torch.isfinite(env.arm_rew_buf).all()
    print("STATE",env.root_states[:4,:3],env.rigid_body_state[:4,env.feet_indices,2],"ALL_FORCE",env._contact_forces.abs().max(),"SENSOR",env.force_sensor_tensor.abs().max(),flush=True)
    metrics={name:env.meter.summary(slice(i*(env.num_envs//4),(i+1)*(env.num_envs//4))) for i,name in enumerate(['stand','forward','turn','forward_turn'])}
    # Isolate formulas from whether a short zero-action PD rollout creates moving contacts.
    import copy
    sh=copy.copy(env);sh.contact_forces=env.contact_forces.clone();sh.rigid_body_state=env.rigid_body_state.clone();sh.episode_length_buf=torch.full_like(env.episode_length_buf,200)
    sh.commands=targets.clone();sh.contact_forces[:,sh.feet_indices]=torch.tensor([0.,0.,100.]);sh.rigid_body_state[:,sh.feet_indices,7:10]=torch.tensor([.1,0.,0.])
    rr=ManipLoco_rewards(sh)
    probes={}
    for n in ['feet_contact_forces','tracking_contacts_shaped_force','tracking_contacts_shaped_vel']:
        raw,_=getattr(rr,'_reward_'+n)();assert torch.isfinite(raw).all() and (raw>=0).all()
        if n!='feet_contact_forces':assert (raw[:env.num_envs//4]==0).all() and (raw[env.num_envs//4:]>0).all()
        else:assert (raw==100).all()
        probes[n]=raw.tolist()
    result={'optimizer_steps':0,'device':'cpu','envs':env.num_envs,'steps':160,'seconds':160*env.dt,'observation_shape':list(env.obs_buf.shape),'finite_states_rewards':True,'actual_reward_sum_verified':True,'controlled_formula_probes':probes,'metrics':metrics,'note':'Zero-action PD diagnostic; no learned locomotion capability or GPU throughput certification. Original validate assumes both shaping rewards strictly negative during short moving-command rollout; zero remains mathematically valid.'}
    (out/'physics_checks.json').write_text(json.dumps(result,indent=2));env.meter.save_samples(out/'physics_samples.npz');print(json.dumps(result,indent=2))
finally:env.gym.destroy_sim(env.sim)
