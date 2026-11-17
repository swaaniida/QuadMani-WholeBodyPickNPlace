"""Diagnostic entry: unchanged training with a cooperative checkpointed stop."""
from pathlib import Path
import isaacgym
import wandb
from legged_gym.scripts import train_hw
from legged_gym.utils.hw_runner import HardwareRunner

ROOT=Path(__file__).resolve().parent
STOP=ROOT/'audit_hw/vbc_walk_verify_20261030/stop_after_update'


class VerificationComplete(Exception):
    pass


original_log=HardwareRunner.log
def log_with_stop(self,locs,*args,**kwargs):
    original_log(self,locs,*args,**kwargs)
    if STOP.exists():
        counter=self.alg.counter
        self.save(str(Path(self.log_dir)/('model_%d.pt'%counter)),counter,
                  infos={'diagnostic_stop':'Requested after a complete policy/history update'})
        print('Verification stopped after complete update',counter,flush=True)
        raise VerificationComplete()


if __name__=='__main__':
    # Diagnostic-only checkpoint frequency; no reward/optimizer changes.
    train_hw.task_registry.train_cfgs['as2piper_plate_vbc_port'].runner.save_interval=50
    HardwareRunner.log=log_with_stop
    try:train_hw.main()
    except VerificationComplete:wandb.finish()
