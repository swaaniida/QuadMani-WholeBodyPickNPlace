"""Measure actual reward calls once per step and audit activation/units."""
import argparse
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace
import isaacgym
from isaacgym import gymapi, gymutil, gymtorch
import numpy as np
import torch
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.envs.manip_loco.as2_head_orin_config import AS2HeadOrinCfg
from legged_gym.envs.rewards.maniploco_rewards import ManipLoco_rewards
from legged_gym.utils.helpers import class_to_dict

parser = argparse.ArgumentParser()
parser.add_argument('--terrain', choices=['plane','trimesh'], default='plane')
parser.add_argument('--label', default='before')
args = parser.parse_args()
root = Path(__file__).resolve().parent
out = root/'validation/reward_review'/args.label/args.terrain
out.mkdir(parents=True, exist_ok=True)
random.seed(20261111)
np.random.seed(20261111)
torch.manual_seed(20261111)
cfg = AS2HeadOrinCfg()
cfg.env.num_envs = 16
cfg.env.episode_length_s = 20
cfg.terrain.mesh_type = args.terrain
cfg.terrain.measure_heights = args.terrain == 'trimesh'
if args.terrain == 'trimesh':
    cfg.terrain.num_rows = cfg.terrain.num_cols = 4
    cfg.terrain.border_size = 5
for flag in ['randomize_friction','randomize_base_mass','randomize_base_com','randomize_motor','randomize_gripper_mass','push_robots']:
    setattr(cfg.domain_rand,flag,False)
cfg.init_state.init_vel_perturb_range = 0.
params = gymapi.SimParams()
params.dt = cfg.sim.dt
params.up_axis = gymapi.UP_AXIS_Z
params.gravity = gymapi.Vec3(0,0,-9.81)
params.use_gpu_pipeline = True
params.physx.use_gpu = True
gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
env = ManipLoco(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True)
targets = torch.tensor([[0,0,0],[.3,0,0],[0,0,.6],[.3,0,.6]],device=env.device).repeat_interleave(4,dim=0)
env._resample_commands = lambda ids: env.commands.__setitem__(ids,targets[ids])
def reset_dofs(ids):
    env.dof_pos[ids] = env.default_dof_pos
    env.dof_vel[ids] = 0
    env.gym.set_dof_state_tensor(env.sim,gymtorch.unwrap_tensor(env.dof_state))
env._reset_dofs = reset_dofs
env.reset()

def ground_at(xy):
    if args.terrain == 'plane': return torch.zeros(xy.shape[:-1],device=env.device)
    pixels = ((xy+cfg.terrain.border_size)/cfg.terrain.horizontal_scale).long()
    x = pixels[...,0].clamp(0,env.height_samples.shape[0]-2)
    y = pixels[...,1].clamp(0,env.height_samples.shape[1]-2)
    height = torch.minimum(env.height_samples[x,y],env.height_samples[x+1,y])
    return torch.minimum(height,env.height_samples[x,y+1])*cfg.terrain.vertical_scale

reward_samples = {}
for branch, functions, names, scales in [('leg',env.reward_functions,env.reward_names,env.reward_scales),('arm',env.arm_reward_functions,env.arm_reward_names,env.arm_reward_scales)]:
    for index,(name,fn) in enumerate(zip(names,functions)):
        key = branch+'/'+name
        reward_samples[key] = []
        def capture(fn=fn,key=key):
            raw, metric = fn()
            raw_array = torch.as_tensor(raw,device=env.device).expand(env.num_envs)
            metric_array = torch.as_tensor(metric,device=env.device).expand(env.num_envs)
            reward_samples[key].append(torch.stack([raw_array,metric_array],dim=-1).cpu().numpy())
            return raw,metric
        functions[index] = capture
physical = []
body_contacts = []
previous_force = None
original_compute = env.compute_reward

def compute():
    global previous_force
    original_compute()
    foot = env.rigid_body_state[:,env.feet_indices]
    ground = ground_at(foot[:,:,:2])
    sensor_force = env.force_sensor_tensor[:,:,:3]
    force_norm = sensor_force.norm(dim=-1)
    sensor_delta = torch.zeros(env.num_envs,device=env.device) if previous_force is None else (sensor_force-previous_force).norm(dim=-1).sum(-1)
    previous_force = sensor_force.clone()
    current_vel = foot[:,:,7:10]
    stale = (env.foot_velocities-current_vel).norm(dim=-1).max(-1).values
    base_ground = ground_at(env.root_states[:,:2])
    physical.append(torch.cat([foot[:,:,2],ground,foot[:,:,2]-ground-.028,force_norm,
        env.root_states[:,2:3],base_ground[:,None],env.base_lin_vel[:,:2],env.base_ang_vel[:,2:3],
        env.commands,stale[:,None],sensor_delta[:,None],env.reset_buf[:,None].float(),
        env.rew_buf[:,None],env.arm_rew_buf[:,None]],dim=1).cpu().numpy())
    body_contacts.append(env.contact_forces.norm(dim=-1).cpu().numpy())
env.compute_reward = compute
try:
    with torch.no_grad():
        for step in range(500):
            env.commands[:] = targets
            env.step(torch.zeros(16,18,device=env.device))
        data = np.stack(physical)
        reward_arrays = {key:np.stack(values) for key,values in reward_samples.items()}
        inventory = {}
        for branch,scales in [('leg',class_to_dict(cfg.rewards.scales)),('arm',class_to_dict(cfg.rewards.arm_scales))]:
            for name,scale in scales.items():
                key = branch+'/'+name
                inventory[key] = {'scale':scale,'called':key in reward_arrays,'gated_by_gait_observation':name.startswith('tracking_contacts_shaped'),'observe_gait_commands':cfg.env.observe_gait_commands}
        result = {'terrain':args.terrain,'seconds':10,'envs':16,'control':'zero leg actions, nominal PD, random arm IK; velocity commands exercise reward gates, no learned walking policy','inventory':inventory,'conditions':{}}
        contacts = np.stack(body_contacts)
        for i,cond in enumerate(('stand','forward','turn','forward_turn')):
            x = data[100:,i*4:(i+1)*4]
            info = {'base_world_mean_m':float(x[:,:,16].mean()),'base_ground_mean_m':float(x[:,:,17].mean()),'base_height_mean_m':float((x[:,:,16]-x[:,:,17]).mean()),'per_foot_force_p95_N':np.quantile(x[:,:,12:16],.95,axis=(0,1)).tolist(),'per_foot_clearance_p95_m':np.quantile(x[:,:,8:12],.95,axis=(0,1)).tolist(),'sensor_force_gt40_fraction':float((x[:,:,12:16]>40).mean()),'stale_foot_velocity_max_error_m_s':float(x[:,:,24].max()),'measured_force_delta_mean_N':float(x[:,:,25].mean()),'resets':int(data[:,i*4:(i+1)*4,26].sum()),'body_contact_peak_N':float(contacts[100:,i*4:(i+1)*4,env.base_body_index].max()),'rewards':{}}
            for key,values in reward_arrays.items():
                array = values[100:,i*4:(i+1)*4]
                info['rewards'][key] = {'raw_mean':float(array[:,:,0].mean()),'metric_mean':float(array[:,:,1].mean()),'weighted_mean_before_div100':float(array[:,:,0].mean()*inventory[key]['scale']),'raw_min':float(array[:,:,0].min()),'raw_max':float(array[:,:,0].max())}
            info['per_foot_sensor_force_mean_N'] = x[:,:,12:16].mean((0,1)).tolist()
            true_force = contacts[100:,i*4:(i+1)*4][:,:,env.feet_indices.cpu().tolist()]
            info['per_foot_contact_force_mean_N'] = true_force.mean((0,1)).tolist()
            info['contact_force_gt40_fraction'] = float((true_force>40).mean())
            info['sensor_contact_disagreement_fraction'] = float(((x[:,:,12:16]>1.5)!=(true_force>2)).mean())
            result['conditions'][cond] = info
        # Independent function probes use a shadow environment with cloned tensors.
        shadow = copy.copy(env)
        shadow.root_states = env.root_states.clone()
        shadow.rigid_body_state = env.rigid_body_state.clone()
        shadow._termination_ground_height = lambda: ground_at(env.root_states[:,:2])
        shadow._reward_foot_ground_heights = lambda: ground_at(env.rigid_body_state[:,env.feet_indices,:2])
        container = ManipLoco_rewards(shadow)
        before_base = container._reward_base_height()[0].clone()
        before_feet = container._reward_feet_height()[0].clone()
        shadow.root_states[:,2] += 1.
        shadow.rigid_body_state[:,:,2] += 1.
        shadow._termination_ground_height = lambda: ground_at(env.root_states[:,:2])+1.
        shadow._reward_foot_ground_heights = lambda: ground_at(env.rigid_body_state[:,env.feet_indices,:2])+1.
        translation = {'base_reward_max_change':float((container._reward_base_height()[0]-before_base).abs().max()),'feet_reward_max_change':float((container._reward_feet_height()[0]-before_feet).abs().max())}
        shadow.cfg = SimpleNamespace(env=SimpleNamespace(observe_gait_commands=True),rewards=cfg.rewards)
        shadow.contact_forces = env.contact_forces.clone()
        shadow.contact_forces[:,env.feet_indices,:] = torch.tensor([0.,0.,50.],device=env.device)
        shadow.desired_contact_states = torch.zeros(16,4,device=env.device)
        force_raw = container._reward_tracking_contacts_shaped_force()[0]
        shadow.desired_contact_states.fill_(1.)
        shadow.foot_velocities = torch.zeros_like(env.foot_velocities)
        shadow.rigid_body_state[:,env.feet_indices,7:10] = torch.tensor([1.,0.,0.],device=env.device)
        vel_raw = container._reward_tracking_contacts_shaped_vel()[0]
        shadow.foot_velocities = shadow.rigid_body_state[:,env.feet_indices,7:10].clone()
        vel_sign_raw = container._reward_tracking_contacts_shaped_vel()[0]
        shadow.force_sensor_tensor = torch.zeros_like(env.force_sensor_tensor)
        shadow.force_sensor_tensor[:,:,2] = 50
        shadow.contact_forces[:,env.feet_indices,:] = torch.tensor([0.,0.,50.],device=env.device)
        shadow.last_contact_forces = torch.zeros_like(env.force_sensor_tensor)
        shadow.episode_length_buf = torch.full_like(env.episode_length_buf,100)
        jerk_raw = container._reward_feet_jerk()[0]
        result['probes'] = {'translation_1m':translation,'swing_force_mismatch_weighted_mean':float(torch.as_tensor(force_raw).float().mean()*env.reward_scales['tracking_contacts_shaped_force']),'stance_velocity_mismatch_weighted_mean':float(torch.as_tensor(vel_raw).float().mean()*env.reward_scales['tracking_contacts_shaped_vel']),'stance_velocity_sign_probe_weighted_mean':float(torch.as_tensor(vel_sign_raw).float().mean()*env.reward_scales['tracking_contacts_shaped_vel']),'force_delta_probe_raw_mean':float(torch.as_tensor(jerk_raw).float().mean())}
        expected_leg = sum(values[:,:,0]*inventory[key]['scale']/100 for key,values in reward_arrays.items() if key.startswith('leg/'))
        expected_arm = sum(values[:,:,0]*inventory[key]['scale']/100 for key,values in reward_arrays.items() if key.startswith('arm/'))
        assert np.allclose(expected_leg,data[:,:,27],atol=1e-6)
        assert np.allclose(expected_arm,data[:,:,28],atol=1e-6)
        assert np.isfinite(data).all()
        result['reward_sum_check_passed'] = True
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        np.savez_compressed(out/'samples.npz',physical=data,body_contacts=contacts,**{key.replace('/','__'):value for key,value in reward_arrays.items()})
        print('saved',out,result['probes'],flush=True)
finally:
    env.gym.destroy_sim(env.sim)
