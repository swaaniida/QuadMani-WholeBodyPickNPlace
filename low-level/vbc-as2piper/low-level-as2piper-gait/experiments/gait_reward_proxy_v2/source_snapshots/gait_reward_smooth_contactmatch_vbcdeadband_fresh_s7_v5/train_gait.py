"""Original PPO/runner, isolated gait reward preset, explicit matched controls."""
import argparse, hashlib, json, os
from pathlib import Path
import isaacgym
from isaacgym import gymapi, gymutil
import torch, wandb
from legged_gym.envs.manip_loco.as2piper_gait_study import (
    AS2PiPERGaitStudy, AS2PiPERGaitStudyCfg, AS2PiPERGaitStudyCfgPPO, PRESETS, configure)
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.utils.hw_runner import HardwareRunner

ROOT=Path(__file__).resolve().parent


class StudyComplete(Exception):
    pass


class StoppableStudyRunner(HardwareRunner):
    def log(self, locs, *args, **kwargs):
        super().log(locs,*args,**kwargs)
        if self.stop_file.exists():
            counter=self.alg.counter
            self.save(str(Path(self.log_dir)/('model_%d.pt'%counter)),counter,
                      infos={'study_stop':'Complete optimizer/history update; checkpoint preserved'})
            print('Study stopped after complete update',counter,flush=True)
            raise StudyComplete()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--preset',choices=PRESETS,default='stable')
    p.add_argument('--controlled',action='store_true')
    p.add_argument('--num-envs',type=int,default=1024)
    p.add_argument('--iterations',type=int,default=150)
    p.add_argument('--checkpoint')
    p.add_argument('--run',required=True)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--save-interval',type=int,default=100)
    p.add_argument('--stop-file',help='Optional cooperative stop marker, checked only after a complete update')
    a=p.parse_args();out=ROOT/'logs/as2piper-low'/a.run
    if out.exists():raise FileExistsError('Refuse to overwrite an existing experiment: '+str(out))
    out.mkdir(parents=True)
    set_seed(a.seed)
    cfg=configure(AS2PiPERGaitStudyCfg(),a.preset,a.controlled);cfg.env.num_envs=a.num_envs
    cfg.seed=a.seed;ppo=AS2PiPERGaitStudyCfgPPO();ppo.runner.save_interval=a.save_interval
    params=gymapi.SimParams();params.use_gpu_pipeline=True;params.physx.use_gpu=True
    gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
    wandb.init(project='as2piper-low',name=a.run,id=os.environ.get('WANDB_RUN_ID',a.run),
               mode=os.environ.get('WANDB_MODE','disabled'),dir=str(out),resume='never')
    env=AS2PiPERGaitStudy(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True)
    runner_class=StoppableStudyRunner if a.stop_file else HardwareRunner
    runner=runner_class(env,class_to_dict(ppo),str(out),'cuda:0')
    if a.stop_file:runner.stop_file=Path(a.stop_file)
    if a.checkpoint:runner.load(a.checkpoint)
    set_seed(a.seed)  # Identical holdout seed after loading checkpoint RNG.
    env._resample_commands(torch.arange(env.num_envs,device=env.device))
    snapshot=dict(arguments=vars(a),environment=class_to_dict(cfg),ppo=class_to_dict(ppo),
                  initial_iteration=runner.current_learning_iteration,
                  asset_sha256=hashlib.sha256(Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(ROOT))).read_bytes()).hexdigest(),
                  source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                     for folder in ['legged_gym','rsl_rl_local'] for f in (ROOT/folder).rglob('*.py')})
    snapshot['environment_realization'] = dict(
        terrain_sha256=hashlib.sha256(env.height_samples.cpu().numpy().tobytes()).hexdigest() if getattr(env,'height_samples',None) is not None else None,
        initial_dof_sha256=hashlib.sha256(env.dof_pos.detach().cpu().numpy().tobytes()).hexdigest(),
        env_origins_sha256=hashlib.sha256(env.env_origins.detach().cpu().numpy().tobytes()).hexdigest())
    (out/'runtime_snapshot.json').write_text(json.dumps(snapshot,indent=2,default=lambda v:v.tolist()))
    stopped=False
    try:runner.learn(a.iterations,init_at_random_ep_len=True)
    except StudyComplete:stopped=True
    (out/'study_finished.json').write_text(json.dumps(dict(completed_updates=runner.alg.counter,cooperative_stop=stopped),indent=2)+'\n')
    wandb.finish();env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
