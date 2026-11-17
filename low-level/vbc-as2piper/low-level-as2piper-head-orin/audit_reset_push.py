"""Paired independent push-strength trials with pre-reset cause logging."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import isaacgym
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz
import numpy as np
import torch
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.envs.manip_loco.as2_head_orin_config import AS2HeadOrinCfg
from legged_gym.utils.helpers import class_to_dict

parser = argparse.ArgumentParser()
parser.add_argument('--multiplier', type=float, required=True)
parser.add_argument('--terrain', choices=['plane', 'trimesh'], default='plane')
args = parser.parse_args()
root = Path(__file__).resolve().parent
out = root/'validation/reset_push'/f'{args.terrain}_mult_{args.multiplier:g}'
out.mkdir(parents=True, exist_ok=True)
random.seed(20261111)
np.random.seed(20261111)
torch.manual_seed(20261111)
cfg = AS2HeadOrinCfg()
cfg.env.num_envs = 32
cfg.env.episode_length_s = 20
cfg.terrain.mesh_type = args.terrain
cfg.terrain.measure_heights = args.terrain == 'trimesh'
if args.terrain == 'trimesh':
    cfg.terrain.num_rows = cfg.terrain.num_cols = 4
    cfg.terrain.border_size = 5
    cfg.terrain.max_init_terrain_level = 3
cfg.domain_rand.push_robots = False  # Trigger exactly once at 8 s.
cfg.domain_rand.standing_push_multiplier = args.multiplier
params = gymapi.SimParams()
params.dt = cfg.sim.dt
params.up_axis = gymapi.UP_AXIS_Z
params.gravity = gymapi.Vec3(0, 0, -9.81)
params.use_gpu_pipeline = True
params.physx.use_gpu = True
gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
env = ManipLoco(cfg, params, gymapi.SIM_PHYSX, 'cuda:0', True)
env._resample_commands = lambda ids: env.commands.__setitem__(ids, 0.)
env.reset()
# Probe the real termination method without advancing or writing simulator state.
saved = {name: getattr(env, name).clone() for name in ('root_states', 'base_quat', 'contact_forces', 'episode_length_buf', 'reset_buf', 'time_out_buf')}
original_ground = env._termination_ground_height
original_indices = env.termination_contact_indices
r_limit, p_limit = cfg.termination.r_threshold, cfg.termination.p_threshold
probes = {}
for name in ('normal', 'roll', 'pitch', 'height', 'timeout', 'contact', 'configured_roll', 'ground_offset'):
    for field in saved:
        getattr(env, field).copy_(saved[field])
    env.contact_forces.zero_()
    env.episode_length_buf.zero_()
    env.base_quat[:] = torch.tensor([0., 0., 0., 1.], device=env.device)
    env._termination_ground_height = lambda: torch.full((env.num_envs,), 10., device=env.device)
    env.root_states[:, 2] = 10.3
    cfg.termination.r_threshold = r_limit
    env.termination_contact_indices = original_indices
    if name in ('roll', 'configured_roll'):
        angle = r_limit + .001 if name == 'roll' else .3
        if name == 'configured_roll': cfg.termination.r_threshold = .2
        a = torch.full((env.num_envs,), angle, device=env.device)
        env.base_quat[:] = quat_from_euler_xyz(a, a*0, a*0)
    if name == 'pitch':
        a = torch.full((env.num_envs,), -p_limit-.001, device=env.device)
        env.base_quat[:] = quat_from_euler_xyz(a*0, a, a*0)
    if name in ('height', 'ground_offset'):
        env.root_states[:, 2] = 10. + cfg.termination.z_threshold - .001
    if name == 'timeout': env.episode_length_buf[:] = env.max_episode_length + 1
    if name == 'contact':
        env.termination_contact_indices = torch.tensor([env.base_body_index], device=env.device)
        env.contact_forces[:, env.base_body_index, 2] = 1.01
    env.check_termination()
    expected = {'roll':'roll', 'pitch':'pitch', 'height':'height', 'ground_offset':'height', 'timeout':'timeout', 'contact':'contact', 'configured_roll':'roll'}.get(name)
    assert bool(env.reset_buf.all()) == (name != 'normal'), name
    if expected: assert env.termination_reasons[expected].all(), name
    probes[name] = {key: bool(value.all()) for key, value in env.termination_reasons.items()}
for field in saved:
    getattr(env, field).copy_(saved[field])
env._termination_ground_height = original_ground
env.termination_contact_indices = original_indices
cfg.termination.r_threshold = r_limit
env.check_termination()

initial_hash = hashlib.sha256(env.root_states.cpu().numpy().tobytes()+env.dof_pos.cpu().numpy().tobytes()+env.mass_params_tensor.cpu().numpy().tobytes()).hexdigest()
traces, events = [], []
original_check = env.check_termination
step = 0
push = None
reset_seen = np.zeros(env.num_envs, dtype=bool)

def check():
    original_check()
    metrics = env.termination_metrics
    reason_names = list(env.termination_reasons)
    reasons = torch.stack(list(env.termination_reasons.values()), dim=1)
    trace = torch.stack([metrics['roll_rad'], metrics['pitch_rad'], metrics['world_z_m'], metrics['ground_z_m'], metrics['height_above_ground_m']], dim=1)
    traces.append(torch.cat([trace, reasons.float()], dim=1).cpu().numpy())
    for index in env.reset_buf.nonzero().flatten().tolist():
        reset_seen[index] = True
        events.append({'time_s':step*env.dt,'env':index,
                       'reasons':[key for key in reason_names if bool(env.termination_reasons[key][index])],
                       **{key:float(value[index]) for key,value in metrics.items()},
                       'foot_force_z_N':env.contact_forces[index,env.feet_indices,2].cpu().tolist(),
                       'base_contact_N':float(env.contact_forces[index,env.base_body_index].norm()),
                       'nonfoot_contacts':[{'body':env.body_names[j], 'force_N':float(env.contact_forces[index,j].norm())}
                                           for j in range(env.num_bodies)
                                           if j not in env.feet_indices.tolist() and float(env.contact_forces[index,j].norm())>1.],
                       'last_push_delta_xy': None if push is None else push['delta_velocity'][index]})
env.check_termination = check
try:
    with torch.no_grad():
        for step in range(round(12/env.dt)):
            env.commands.zero_()
            if step == round(8/env.dt) and args.multiplier > 0:
                torch.manual_seed(20261112)  # Same sampled XY vector for every strength.
                env._push_robots()
                push = {key:(value.cpu().tolist() if isinstance(value,torch.Tensor) else value) for key,value in env.last_push.items()}
            env.step(torch.zeros(env.num_envs,18,device=env.device))
            assert torch.isfinite(env.obs_buf).all()
    array = np.stack(traces)
    result = {'terrain':args.terrain,'envs':32,'seconds':12,'seed':20261111,
              'multiplier':args.multiplier,'base_max_velocity_m_s':cfg.domain_rand.max_push_vel_xy,
              'effective_axis_limit_m_s':args.multiplier*cfg.domain_rand.max_push_vel_xy,
              'reset_start_height_m':cfg.init_state.pos[2],
              'termination_thresholds':{'roll_rad':r_limit,'pitch_rad':p_limit,'height_m':cfg.termination.z_threshold,'height_reference':cfg.termination.height_reference,'contact_N':1.},
              'active_contact_termination_bodies':[env.body_names[i] for i in original_indices.tolist()],
              'initial_state_and_dr_sha256':initial_hash,
              'terrain_sha256': None if args.terrain=='plane' else hashlib.sha256(env.height_samples.cpu().numpy().tobytes()).hexdigest(),
              'reset_count':len(events),
              'failed_environments':int(reset_seen.sum()),'survived_environments':int((~reset_seen).sum()),
              'pre_push_resets':sum(e['time_s']<8 for e in events),'events':events,
              'trace_columns':['roll_rad','pitch_rad','world_z_m','ground_z_m','height_above_ground_m','contact','roll','pitch','height','timeout'],
              'push':push,'threshold_probes':probes,
              'controller':'nominal leg PD, random EE DLS position drive, no learned policy'}
    np.savez_compressed(out/'traces.npz',samples=array)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print({key:result[key] for key in ('multiplier','effective_axis_limit_m_s','reset_count','failed_environments','pre_push_resets')},flush=True)
finally:
    env.gym.destroy_sim(env.sim)
