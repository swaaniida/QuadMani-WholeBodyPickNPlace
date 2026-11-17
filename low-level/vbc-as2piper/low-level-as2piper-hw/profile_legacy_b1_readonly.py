"""Read-only B1 environment benchmark, output/cache confined to the AS2 worktree."""
import sys
from pathlib import Path
BASE=Path('/data/team/users/suhwan/vbc-legacy')
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(BASE/'low-level'))
sys.path.insert(1,str(BASE/'third_party/rsl_rl'))
import json,time
import isaacgym
from isaacgym import gymapi,gymutil
import torch
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg
from legged_gym.utils.helpers import class_to_dict,set_seed


def main():
    set_seed(1);cfg=B1Z1RoughCfg();cfg.env.observe_gait_commands=True
    cfg.env.stop_update_goal=False;cfg.env.pitch_control=False;cfg.env.num_envs=6144
    params=gymapi.SimParams();params.use_gpu_pipeline=True;params.physx.use_gpu=True
    gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
    env=ManipLoco(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True);env.reset()
    action=torch.zeros(env.num_envs,18,device=env.device)
    with torch.no_grad():
        for _ in range(10):env.step(action.clone())
    timings={'sampling_s':0.,'sampling_calls':0,'sampling_items':0}
    original=env._resample_ee_goal
    def sample(ids,is_init=False):
        torch.cuda.synchronize();start=time.perf_counter();result=original(ids,is_init)
        torch.cuda.synchronize();timings['sampling_s']+=time.perf_counter()-start
        timings['sampling_calls']+=1;timings['sampling_items']+=len(ids);return result
    env._resample_ee_goal=sample
    env.goal_timer[:]=env.traj_total_timesteps-torch.randint(1,25,(env.num_envs,),device=env.device)
    with torch.no_grad():
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(24):env.step(action.clone())
        torch.cuda.synchronize();wall=time.perf_counter()-start
    result=dict(model='B1+Z1 original source/defaults, observe_gait=True',seed=1,num_envs=env.num_envs,
        num_dofs=env.num_dofs,num_bodies=env.num_bodies,num_observations=env.obs_buf.shape[1],steps=24,
        wall_s=wall,non_sampling_s=wall-timings['sampling_s'],timings=timings,
        sim_config=class_to_dict(cfg.sim),terrain_config=class_to_dict(cfg.terrain),
        environment_source=__import__('inspect').getfile(ManipLoco),
        note='No PPO, zero action; concurrent AS2 run. Not a reconstruction of unknown historical W&B overrides.')
    path=ROOT/'audit_hw/vbc_port_audit_20261029/profile_b1_original_seed1.json'
    encoded=json.dumps(result,indent=2,default=lambda value:value.tolist())
    path.write_text(encoded);print(encoded,flush=True);env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
