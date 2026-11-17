"""Hardware task entry, runtime snapshots and extended checkpoint restoration."""
import os,json,hashlib,importlib
from pathlib import Path
import isaacgym
import torch,wandb
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.hw_runner import HardwareRunner


def main():
    args=get_args();root=Path(LEGGED_GYM_ROOT_DIR)
    out=root/'logs'/args.proj_name/args.exptid;out.mkdir(parents=True,exist_ok=True)
    wandb.init(project=args.proj_name,name=args.exptid,id=os.environ.get('WANDB_RUN_ID'),
        resume=os.environ.get('WANDB_RESUME','never'),mode=os.environ.get('WANDB_MODE','online'),dir=str(out))
    cfg,train_cfg=task_registry.get_cfgs(args.task)
    env,cfg=task_registry.make_env(name=args.task,args=args,env_cfg=cfg)
    assert env.num_dofs==20 and env.num_actions==18 and env.obs_buf.shape==(env.num_envs,799)
    assert env.ee_j_eef.shape==(env.num_envs,6,6)
    importlib.import_module('legged_gym.utils.task_registry').OnPolicyRunner=HardwareRunner
    runner,train_cfg,_=task_registry.make_alg_runner(log_root=str(out),env=env,name=args.task,args=args)
    # Registry's historical set_it(checkpoint filename) must not undo next_iteration.
    if args.resumeid:
        runner.set_it(runner.alg.counter)
        env._resample_commands(torch.arange(env.num_envs,device=env.device))
    asset=Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(root)))
    snapshot=dict(environment=class_to_dict(cfg),ppo=class_to_dict(train_cfg),
        hardware=json.loads((root/getattr(cfg.asset,'hardware_profile','resources/commands/hardware.json')).read_text()),
        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['legged_gym','rsl_rl_local/rsl_rl'] for p in (root/folder).rglob('*.py')},
        initial_iteration=runner.current_learning_iteration)
    (out/'runtime_snapshot.json').write_text(json.dumps(snapshot,indent=2,default=lambda v:v.tolist()))
    wandb.config.update({'hardware_task':args.task,'observations':799,'num_envs':env.num_envs,
        'leg_kp':cfg.control.stiffness,'leg_kd':cfg.control.damping,'gait_hz':cfg.env.frequencies,
        'terrain':cfg.terrain.mesh_type,'asset_sha256':snapshot['asset_sha256'],
        'ee_sampling_mode':cfg.goal_ee.sampling_mode,
        'ee_command_ranges':class_to_dict(cfg.goal_ee.ranges)})
    target=args.max_iterations or train_cfg.runner.max_iterations
    remaining=target-runner.current_learning_iteration
    if remaining<=0:raise ValueError('Target iteration is not after the checkpoint')
    runner.learn(num_learning_iterations=remaining,init_at_random_ep_len=True)
    print('AS2 hardware training completed',flush=True);wandb.finish()


if __name__=='__main__':main()
