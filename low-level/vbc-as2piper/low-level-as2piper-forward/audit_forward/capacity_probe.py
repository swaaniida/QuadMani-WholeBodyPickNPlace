"""Bounded full-size capacity probe; disabled W&B, separate run/checkpoints."""
import json
import time
from pathlib import Path
import isaacgym  # Must precede torch.
import torch
from legged_gym.scripts import train_as2
from legged_gym.utils import task_registry

OUT=Path(__file__).resolve().parent
REPORT=OUT/'capacity_probe_measurements.json'
measurements={}
original=task_registry.make_alg_runner


def measured_runner(*args,**kwargs):
    env=kwargs['env']
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free,total=torch.cuda.mem_get_info()
    measurements.update(num_envs=env.num_envs,num_dofs=env.num_dofs,num_actions=env.num_actions,
                        observations=list(env.obs_buf.shape),total_MiB=total/2**20,
                        free_before_forced_resample_MiB=free/2**20)
    # All environments resampling together stresses the new collision guard.
    # This affects this temporary probe only, never the existing run.
    start=time.monotonic()
    env._resample_ee_goal(torch.arange(env.num_envs,device=env.device))
    torch.cuda.synchronize()
    measurements['forced_all_env_resample_s']=time.monotonic()-start
    measurements['forced_resample_fallback_count']=env.pose_fallback_count
    measurements['free_after_forced_resample_MiB']=torch.cuda.mem_get_info()[0]/2**20
    measurements['torch_peak_allocated_after_resample_MiB']=torch.cuda.max_memory_allocated()/2**20
    REPORT.write_text(json.dumps(measurements,indent=2))
    print('CAPACITY_RESAMPLE '+json.dumps(measurements),flush=True)
    return original(*args,**kwargs)


if __name__=='__main__':
    task_registry.make_alg_runner=measured_runner
    start=time.monotonic()
    try:
        train_as2.main()
        measurements['completed']=True
    finally:
        measurements['wall_time_s']=time.monotonic()-start
        if torch.cuda.is_initialized():
            measurements['torch_peak_allocated_MiB']=torch.cuda.max_memory_allocated()/2**20
            measurements['torch_peak_reserved_MiB']=torch.cuda.max_memory_reserved()/2**20
            measurements['free_at_exit_MiB']=torch.cuda.mem_get_info()[0]/2**20
        REPORT.write_text(json.dumps(measurements,indent=2))
