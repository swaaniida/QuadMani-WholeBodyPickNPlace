"""Queue one tracking-weight ablation after our current training slot is free."""
import subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
checkpoint=ROOT/'logs/as2piper-low/gait_fullenv_progress_s7_v1/model_201.pt'
start=time.monotonic()
while not checkpoint.exists():
    if time.monotonic()-start>14400:raise TimeoutError(str(checkpoint))
    time.sleep(10)
print('Starting fresh tracking-gain comparison, no checkpoint loading',flush=True)
cmd=['bash','run_gait.sh','--preset','progress_gain2','--num-envs','6144','--iterations','600',
     '--run','gait_fullenv_progress_gain2_s7_v1','--seed','7',
     '--stop-file','experiments/gait_full_env_v1/progress_gain2/stop_when_valid']
with (ROOT/'experiments/gait_full_env_v1/progress_gain2_train.log').open('w') as log:
    subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
