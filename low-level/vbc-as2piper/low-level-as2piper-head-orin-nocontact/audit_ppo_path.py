"""Replay forward/backward paths while blocking every optimizer step."""
from pathlib import Path
import hashlib,json,tempfile
import isaacgym
import torch,numpy as np
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from legged_gym.envs.manip_loco.as2_head_orin_config import AS2HeadOrinCfgPPO
from legged_gym.utils.helpers import class_to_dict,export_policy_as_jit

root=Path(__file__).resolve().parent;out=root/'validation/gait_clock'
torch.manual_seed(20261111);torch.set_num_threads(2)
data=np.load(out/'after/real_rollout.npz')
policy_cfg=class_to_dict(AS2HeadOrinCfgPPO.policy)
model=ActorCritic(67,67,18,**policy_cfg,num_priv=18,num_hist=10,num_prop=67)
algorithm=PPO(model,device='cpu',**class_to_dict(AS2HeadOrinCfgPPO.algorithm))
algorithm.init_storage(16,24,[755],[None],[18])

def fingerprint():
    return hashlib.sha256(b''.join(value.detach().numpy().tobytes() for value in model.parameters())).hexdigest()

def gradients():
    groups={}
    for name,value in model.named_parameters():
        group='.'.join(name.split('.')[:2])
        groups[group]=groups.get(group,0.)+(0. if value.grad is None else float(value.grad.norm())**2)
    return {key:value**.5 for key,value in groups.items()}

initial=fingerprint();replay=[];timeout_verified=False
with torch.no_grad():
    for i in range(24):
        obs=torch.from_numpy(data['observations'][-24+i].copy())
        action=algorithm.act(obs,obs,hist_encoding=False)
        old_log=algorithm.transition.actions_log_prob.clone()
        assert torch.allclose(old_log,model.get_actions_log_prob(action))
        reward=torch.from_numpy(data['rewards'][-24+i].copy())
        done=torch.from_numpy(data['dones'][-24+i].copy()).bool()
        infos={}
        if i==3:
            infos['time_outs']=torch.zeros(16,dtype=torch.bool);infos['time_outs'][0]=True;done[0]=True
            expected=reward[0]+algorithm.gamma*algorithm.transition.values[0]
        algorithm.process_env_step(reward[:,0],reward[:,1],done,infos)
        if i==3:
            assert torch.allclose(algorithm.storage.rewards[i,0],expected)
            timeout_verified=True
        replay.append(obs)
    algorithm.compute_returns(replay[-1])
assert torch.isfinite(algorithm.storage.advantages).all()
result={'mode':'structural replay and backward only; real PD observations/rewards with sampled diagnostic actions; no on-policy training','optimizer_steps':0,'timeout_bootstrap_verified':timeout_verified,'advantage_global_mean':float(algorithm.storage.advantages.mean()),'advantage_global_std':float(algorithm.storage.advantages.std()),'advantage_branch_means':algorithm.storage.advantages.mean((0,1)).tolist(),'mixing_ratio_at_0':algorithm.get_value_mixing_ratio()}

class StopBeforeStep(Exception):
    pass

def block_ppo_step():
    result['actual_ppo_update_gradients']=gradients()
    raise StopBeforeStep()
algorithm.optimizer.step=block_ppo_step
try:algorithm.update()
except StopBeforeStep:pass
assert fingerprint()==initial
assert result['actual_ppo_update_gradients']['actor.actor_leg_control_head']>0
assert result['actual_ppo_update_gradients']['actor.actor_arm_control_head']>0
assert result['actual_ppo_update_gradients']['critic.critic_leg_control_head']>0
assert result['actual_ppo_update_gradients']['critic.critic_arm_control_head']>0
model.zero_grad(set_to_none=True)
def block_history_step():
    result['actual_dagger_gradients']=gradients()
    raise StopBeforeStep()
algorithm.hist_encoder_optimizer.step=block_history_step
try:algorithm.update_dagger()
except StopBeforeStep:pass
assert fingerprint()==initial
assert result['actual_dagger_gradients']['actor.history_encoder']>0
assert result['actual_dagger_gradients']['actor.actor_backbone']==0

# Prove the two reward/advantage streams and the configured mixing route.
model.zero_grad(set_to_none=True)
obs=replay[-1]
with torch.no_grad():sample=model.act(obs,False).detach()
routing={}
for stream in ('leg','arm'):
    for mix in (0.,1.):
        model.zero_grad(set_to_none=True);model.act(obs,False)
        log_prob=model.get_actions_log_prob(sample)
        ratio=torch.exp(log_prob-log_prob.detach())
        adv=torch.zeros(16,2);adv[:,0 if stream=='leg' else 1]=torch.linspace(-1,1,16)
        mixed=torch.stack([adv[:,0]+mix*adv[:,1],adv[:,1]+mix*adv[:,0]],dim=-1)
        (-mixed*ratio).mean().backward()
        routing[f'{stream}_mix{mix:g}']=gradients()
result['advantage_routing_gradients']=routing
assert routing['leg_mix0']['actor.actor_arm_control_head']==0
assert routing['arm_mix0']['actor.actor_leg_control_head']==0
assert routing['arm_mix1']['actor.actor_leg_control_head']>0

model.zero_grad(set_to_none=True)
with torch.no_grad():
    history=model.act_inference(obs,True)
    perturbed=obs.clone();perturbed[:,67:85]+=10
    native_delta=float((history-model.act_inference(perturbed,True)).abs().max())
    private_delta=float((model.act_inference(obs,False)-model.act_inference(perturbed,False)).abs().max())
    assert native_delta==0 and private_delta>0
    deployment=torch.cat([obs[:,:67],obs[:,85:]],dim=1)
    assert deployment.shape==(16,737)
    with tempfile.TemporaryDirectory() as folder:
        export_policy_as_jit(model,folder)
        scripted=torch.jit.load(str(Path(folder)/'policy_1.pt'))
        exported=scripted(deployment)
        export_error=float((exported-history).abs().max())
        assert export_error<1e-6
        try:
            scripted(torch.zeros(16,682))
        except Exception:
            result['old_682_jit_input_rejected']=True
        else:
            raise AssertionError('Old no-clock deployment input accepted')
        try:
            scripted(torch.zeros(16,726))
        except Exception:
            result['old_726_jit_input_rejected']=True
        else:
            raise AssertionError('Old padded deployment input accepted')
        restored=scripted(torch.cat([perturbed[:,:67],perturbed[:,85:]],dim=1))
        assert torch.equal(exported,restored)
    runner=OnPolicyRunner.__new__(OnPolicyRunner);runner.alg=algorithm
    default_inference=runner.get_inference_policy(device='cpu')
    assert torch.equal(default_inference(obs),history)
for label,fn in [('actor',lambda: model.act_inference(torch.zeros(16,744),True)),('critic',lambda: model.evaluate(torch.zeros(16,744)))]:
    try:fn()
    except (ValueError,RuntimeError):result['old_744_'+label+'_input_rejected']=True
    else:raise AssertionError('Old observation accepted by '+label)
for label,fn in [('actor700',lambda: model.act_inference(torch.zeros(16,700),True)),('critic700',lambda: model.evaluate(torch.zeros(16,700)))]:
    try:fn()
    except (ValueError,RuntimeError):result['old_'+label+'_input_rejected']=True
    else:raise AssertionError('Old no-clock observation accepted by '+label)
# Check actual network widths; removed contact features do not survive as zero slots.
assert model.actor.actor_backbone[0].in_features==87
assert model.critic.critic_backbone[0].in_features==85
assert model.actor.history_encoder.encoder[0].in_features==67
assert algorithm.storage.observations.shape[-1]==755
result['network_input_widths']={'actor_proprio_plus_latent':87,'critic_proprio_plus_priv':85,'history_frame':67,'storage_observation':755,'jit_deployment':737}
result.update(history_actor_privileged_independence_max_error=native_delta,privileged_actor_sensitivity=private_delta,scripted_deployment_max_error=export_error,default_inference_history_verified=True,all_parameters_unchanged=fingerprint()==initial,adam_state_entries=len(algorithm.optimizer.state),history_adam_state_entries=len(algorithm.hist_encoder_optimizer.state))
assert result['all_parameters_unchanged'] and not algorithm.optimizer.state and not algorithm.hist_encoder_optimizer.state
# Quantify the static-policy incentive without changing reward scales.
result['standing_risk_counterfactual']=[{'vx':vx,'velocity_weighted':2*min(vx,.3)/(.3+1e-5),'walking_dof_unchanged':1.464,'two_term_sum':1.464+2*min(vx,.3)/(.3+1e-5)} for vx in (0.,.05,.15,.3)]
(out/'ppo_path.json').write_text(json.dumps(result,indent=2)+'\n')
print('Backward and deployment checks passed; no parameter or optimizer state update',flush=True)
