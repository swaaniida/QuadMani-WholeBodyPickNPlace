"""Complete the tracking-shape x tracking-weight factorial in a freed slot."""
import subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
checkpoint=ROOT/'logs/as2piper-low/gait_fullenv_original_s7_v1/model_600.pt'
start=time.monotonic()
while not checkpoint.exists():
    if time.monotonic()-start>21600:raise TimeoutError(str(checkpoint))
    time.sleep(10)
print('Starting fresh Gaussian-only gain comparison',flush=True)
cmd=['bash','run_gait.sh','--preset','gaussian_gain2','--num-envs','6144','--iterations','201',
     '--run','gait_fullenv_gaussian_gain2_s7_v1','--seed','7']
with (ROOT/'experiments/gait_full_env_v1/gaussian_gain2_train.log').open('w') as log:
    subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
