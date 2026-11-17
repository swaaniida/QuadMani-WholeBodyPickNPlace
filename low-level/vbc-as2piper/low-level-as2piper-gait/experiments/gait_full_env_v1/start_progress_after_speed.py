"""Queue one additional fresh reward arm after our speed trial finishes."""
import os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
end=ROOT/'logs/as2piper-low/gait_fullenv_speed_s7_v1/model_200.pt'
start=time.monotonic()
while not end.exists():
    if time.monotonic()-start>14400:raise TimeoutError(str(end))
    time.sleep(10)
out=ROOT/'experiments/gait_full_env_v1'
cmd=['bash','run_gait.sh','--preset','progress','--num-envs','6144','--iterations','201',
     '--run','gait_fullenv_progress_s7_v1','--seed','7']
print('Starting queued progress-only reward comparison',flush=True)
with (out/'progress_train.log').open('w') as log:
    subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
