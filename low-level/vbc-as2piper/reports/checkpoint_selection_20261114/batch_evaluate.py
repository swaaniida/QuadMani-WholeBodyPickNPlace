"""Sequential candidate evaluation; no training, no W&B writes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
clone = root / 'low-level-as2piper-head-orin-nocontact'
run = clone / 'experiments/contact_force_ablation/runs/as2piper_headpiper_orin_200N_0.18m'
env = os.environ.copy()
env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
           PYTHONPATH=':'.join(map(str, [clone / '.wandb_deps', clone, clone / '.deps',
                                       clone / 'rsl_rl_local', root / 'third_party/isaacgym/python'])),
           TORCH_EXTENSIONS_DIR=str(root / 'low-level-as2piper-head-orin/.torch_extensions'),
           MAX_JOBS='2')
os.umask(0o002)
status = []
parser = argparse.ArgumentParser()
parser.add_argument('--iterations', nargs='+', type=int,
                    default=[18000, 20000, 24000, 28000, 30000, 34000, 40600, 43400, 45000])
parser.add_argument('--status-name', default='batch_status.json')
args = parser.parse_args()
for it in args.iterations:
    for terrain in ['flat', 'rough']:
        destination = out / 'evaluations' / ('%d_%s' % (it, terrain))
        if (destination / 'result.json').exists():
            status.append({'iteration': it, 'terrain': terrain, 'existing_complete': True})
            continue
        assert not destination.exists(), 'Incomplete output must be inspected before retry: ' + str(destination)
        cmd = [sys.executable, str(out / 'evaluate_candidate.py'), '--checkpoint', str(run / ('model_%d.pt' % it)),
               '--terrain', terrain, '--output', str(destination)]
        start = time.time()
        with (out / ('eval_%d_%s.log' % (it, terrain))).open('w') as f:
            result = subprocess.run(cmd, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT)
        entry = {'iteration': it, 'terrain': terrain, 'returncode': result.returncode, 'wall_seconds': time.time()-start}
        status.append(entry)
        (out / args.status_name).write_text(json.dumps(status, indent=2) + '\n')
        print(json.dumps(entry), flush=True)
        if result.returncode:
            sys.exit(result.returncode)
