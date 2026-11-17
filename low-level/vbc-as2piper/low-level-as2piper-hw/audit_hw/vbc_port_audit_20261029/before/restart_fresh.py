"""Delete the explicitly superseded resumed run and start fresh WBC training."""
import os
import argparse
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD_ID = 'as2piper_hw_gaitfix_ee_transfer520_20261029_124502'


def requested_processes(old_id=OLD_ID):
    matches = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
            if b'legged_gym/scripts/train_hw.py' not in args:
                continue
            if b'--exptid' in args and args[args.index(b'--exptid') + 1] == old_id.encode():
                matches.append(int(entry.name))
        except (FileNotFoundError, PermissionError, IndexError):
            continue
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-run-id', default=OLD_ID)
    parser.add_argument('--task', default='as2piper_plate_walk_wbc_vbc')
    parser.add_argument('--run-prefix', default='as2piper_hw_gaitfix_vbcsampler_scratch')
    options = parser.parse_args()
    old_id = options.old_run_id
    if not old_id.startswith('as2piper_hw_gaitfix_') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in old_id):
        raise SystemExit('Only an exact AS2 HW gait-fix run ID is allowed.')
    old_run = 'quadruped_capstone/as2piper-low/' + old_id
    os.umask(0o002)
    fresh_marker = ROOT / 'launch_logs' / ('fresh_restart_' + old_id + '.done')
    legacy_marker = ROOT / 'launch_logs' / 'fresh_restart_20261029.done'
    if old_id == OLD_ID and legacy_marker.exists():
        raise SystemExit('Original restart already completed: ' + legacy_marker.read_text().strip())
    if fresh_marker.exists():
        raise SystemExit('Fresh restart already completed: ' + fresh_marker.read_text().strip())

    # Match the full run ID, never stop another experiment by a broad pattern.
    old_pids = set(requested_processes(old_id))
    launcher_files = []
    for path in (ROOT/'launch_logs').glob('*.pid'):
        try:
            pid = int(path.read_text().strip())
            if pid in old_pids:
                launcher_files.extend([path, path.with_suffix('.log')])
        except (ValueError, OSError):
            continue
    for pid in old_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            raise SystemExit('Run this script as vbc or root; cannot stop PID ' + str(pid))
        print('Stopping superseded run, PID', pid, flush=True)
    deadline = time.monotonic() + 30
    while requested_processes(old_id):
        if time.monotonic() > deadline:
            raise SystemExit('Old training has not exited; no data deleted and no new training started.')
        time.sleep(0.5)

    # Wait for the old W&B writer to finish before deleting its remote run.
    for _ in range(20):
        writer_alive = False
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                args = (entry / 'cmdline').read_bytes().split(b'\0')
                writer_alive |= (b'--pid' in args and
                                 args[args.index(b'--pid') + 1] in {str(p).encode() for p in old_pids} and
                                 any(b'wandb-core' in arg for arg in args))
            except (FileNotFoundError, PermissionError, IndexError):
                continue
        if not writer_alive:
            break
        time.sleep(0.5)
    if writer_alive:
        raise SystemExit('Old W&B writer is still exiting; rerun after it finishes.')

    os.environ['WANDB_BASE_URL'] = 'https://api.wandb.ai'
    import wandb
    api = wandb.Api(timeout=30)
    # Query the exact ID; fail before local deletion if the API is unavailable.
    old = next((run for run in api.runs('quadruped_capstone/as2piper-low',
                                      filters={'name': old_id}) if run.id == old_id), None)
    if old is not None:
        old.delete(delete_artifacts=False)
        print('Deleted W&B run:', old_run, flush=True)
    else:
        print('W&B run already absent:', old_run, flush=True)

    old_dir = ROOT / 'logs' / 'as2piper-low' / old_id
    if old_dir.is_symlink():
        raise RuntimeError('Refusing unexpected symlink: ' + str(old_dir))
    if old_dir.exists():
        shutil.rmtree(str(old_dir))
        print('Deleted local run:', old_dir, flush=True)
    # These launcher files belong only to the superseded run.
    for path in launcher_files:
        if path.exists():
            path.unlink()

    new_id = options.run_prefix + '_' + datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    env = os.environ.copy()
    env.update(AS2_TASK=options.task, WANDB_MODE='online',
               WANDB_BASE_URL='https://api.wandb.ai', WANDB_RUN_ID=new_id,
               WANDB_NAME=new_id, WANDB_RESUME='never')
    # No --resume, --resumeid, or --checkpoint: model/Adam/counters start fresh.
    subprocess.run(['bash', str(ROOT / 'launch_hw.sh'), '--max_iterations', '45000'],
                   cwd=str(ROOT), env=env, check=True)
    fresh_marker.write_text(new_id + '\n')
    print('Fresh run:', new_id, flush=True)
    print('Task: ' + options.task + '; iteration 0 -> 45000', flush=True)


if __name__ == '__main__':
    main()
