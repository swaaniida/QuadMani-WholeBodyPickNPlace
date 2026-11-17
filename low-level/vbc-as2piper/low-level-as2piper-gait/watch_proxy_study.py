"""Refresh bounded study reports; never start, stop, or modify training."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LOGS=ROOT/'experiments/gait_reward_proxy_v2/resume_logs'
# At most four hours; trainers have their own fixed 600-update budgets.
for _ in range(240):
    subprocess.run([sys.executable,'report_proxy_v2.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'plot_proxy_v2.py'],cwd=ROOT,check=True)
    alive=False
    for case in ['stable_proxy','progress_gain2_proxy','stable_proxy_ground']:
        files=sorted(LOGS.glob(case+'_resume1_*.pid'))
        if files:
            proc=Path('/proc')/files[-1].read_text().strip()/'cmdline'
            if proc.exists() and b'train_gait.py' in proc.read_bytes():alive=True
    # Evaluators may need time after the last optimizer update.
    if not alive:
        pending=[]
        for f in LOGS.glob('eval_*resume1_*.pid'):
            proc=Path('/proc')/f.read_text().strip()/'cmdline'
            if proc.exists() and b'monitor_full_env.py' in proc.read_bytes():pending.append(f)
        if not pending:break
    time.sleep(60)
