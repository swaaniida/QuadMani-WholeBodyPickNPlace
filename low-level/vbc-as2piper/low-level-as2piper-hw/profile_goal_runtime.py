"""Scoped diagnostic: attribute rollout time without changing a live run."""
import argparse
import json
import time
import types
from collections import defaultdict
from pathlib import Path

import isaacgym
from isaacgym import gymapi,gymutil
import torch
from legged_gym.envs.manip_loco.as2piper_walk_wbc import AS2PiPERWalkWBC, AS2PiPERWalkWBCNominalCfg
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.envs.manip_loco.as2piper_walk_wbc_vbc import AS2PiPERWalkWBCVBC, AS2PiPERWalkWBCVBCCfg
from legged_gym.envs.manip_loco.as2piper_vbc_port import AS2PiPERVBCPort,AS2PiPERVBCPortCfg
from legged_gym.utils.helpers import class_to_dict,set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['guard', 'vbc', 'screened', 'port'], required=True)
    parser.add_argument('--terrain',choices=['plane','trimesh'])
    parser.add_argument('--dr',choices=['off','full'],default='off')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    set_seed(args.seed)
    cfg = AS2PiPERWalkWBCVBCCfg() if args.mode == 'screened' else AS2PiPERWalkWBCNominalCfg()
    if args.mode=='port':cfg=AS2PiPERVBCPortCfg()
    cfg.env.num_envs = args.num_envs
    if args.dr=='off':
        for key in ['randomize_base_mass','randomize_base_com','randomize_gripper_mass','randomize_friction','randomize_motor','push_robots']:setattr(cfg.domain_rand,key,False)
    if args.terrain:
        cfg.terrain.mesh_type=args.terrain;cfg.terrain.measure_heights=args.terrain=='trimesh'
    params = gymapi.SimParams()
    params.dt = cfg.sim.dt
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = True
    params.physx.use_gpu = True
    for key in ['num_position_iterations', 'num_velocity_iterations', 'contact_offset',
                'rest_offset', 'bounce_threshold_velocity', 'max_depenetration_velocity']:
        setattr(params.physx, key, getattr(cfg.sim.physx, key))
    gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
    cls = AS2PiPERWalkWBCVBC if args.mode == 'screened' else AS2PiPERWalkWBC
    if args.mode=='port':cls=AS2PiPERVBCPort
    env = cls(cfg, params, gymapi.SIM_PHYSX, 'cuda:0', True)
    env.reset()  # Same initialization as OnPolicyRunner; no zero/unset goal bank.
    if args.mode in ['screened','port']:
        assert not hasattr(env, 'collision_guard')
        assert env._collision_check.__func__ is ManipLoco._collision_check
    if args.mode == 'vbc':
        env._collision_check = types.MethodType(ManipLoco._collision_check, env)
        initial = env._resample_ee_goal
        def resample(ids, is_init=False):
            if is_init:
                return initial(ids, is_init=True)
            return ManipLoco._resample_ee_goal(env, ids, is_init=False)
        env._resample_ee_goal = resample
    action = torch.zeros(env.num_envs, 18, device=env.device)
    with torch.no_grad():
        for _ in range(10):
            env.step(action.clone())
    elapsed = defaultdict(float)
    calls = defaultdict(int)
    counts = defaultdict(int)
    def wrap(obj, field, label):
        original = getattr(obj, field)
        def timed(*a, **kw):
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = original(*a, **kw)
            torch.cuda.synchronize()
            elapsed[label] += time.perf_counter() - start
            calls[label] += 1
            if a and isinstance(a[0], torch.Tensor):
                counts[label] += len(a[0])
            return value
        setattr(obj, field, timed)
    wrap(env, '_resample_ee_goal', 'goal_sampling_inclusive')
    wrap(env, '_collision_check', 'candidate_check_inclusive')
    if hasattr(env, 'collision_guard'):
        wrap(env.collision_guard, 'solve', 'iterative_ik')
        wrap(env.collision_guard, 'collides', 'body_obb')
    # Spread requests across rollout steps, as in an asynchronous vector env.
    env.goal_timer[:] = env.traj_total_timesteps - torch.randint(1, 25, (env.num_envs,), device=env.device)
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(args.steps):
            env.step(action.clone())
            if step % 24 == 23:
                print('PROFILE', args.mode, step + 1, round(time.perf_counter() - start, 3), flush=True)
        torch.cuda.synchronize()
        total = time.perf_counter() - start
    result = dict(mode=args.mode, seed=args.seed,terrain=cfg.terrain.mesh_type,dr=args.dr,physics_config=class_to_dict(cfg.sim),num_envs=env.num_envs, steps=args.steps, wall_s=total,
                  equivalent_collection_24steps_s=total * 24 / args.steps,
                  seconds=dict(elapsed), calls=dict(calls), items=dict(counts),
                  non_sampling_s=total-elapsed['goal_sampling_inclusive'],
                  note='CUDA-synchronized nested timers; no PPO; concurrent existing training may affect absolute time.')
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
