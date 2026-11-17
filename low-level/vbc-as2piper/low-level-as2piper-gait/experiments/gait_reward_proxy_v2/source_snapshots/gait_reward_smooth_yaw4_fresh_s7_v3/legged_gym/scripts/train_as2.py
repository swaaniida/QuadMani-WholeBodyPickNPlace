"""Isolated AS2 plate-only train entry; smoke mode also checks tensor contracts."""
import os
import json
from pathlib import Path
import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
import torch
import wandb


def main():
    args=get_args()
    path=Path(LEGGED_GYM_ROOT_DIR)/'logs'/args.proj_name/args.exptid
    path.mkdir(parents=True,exist_ok=True)
    wandb.init(project=args.proj_name,name=args.exptid,id=os.environ.get('WANDB_RUN_ID'),
               resume=os.environ.get('WANDB_RESUME','never'),mode=os.environ.get('WANDB_MODE','online'),dir=str(path))
    cfg,train_cfg=task_registry.get_cfgs(args.task)
    if os.environ.get('AS2_SMOKE')=='1':
        cfg.env.num_envs=8
        cfg.terrain.num_rows=2;cfg.terrain.num_cols=2;cfg.terrain.max_init_terrain_level=1
    env,cfg=task_registry.make_env(name=args.task,args=args,env_cfg=cfg)
    env.reset()
    assert env.num_dofs==20 and env.num_actions==18
    assert env.obs_buf.shape==(env.num_envs,cfg.env.num_observations)
    assert env.ee_j_eef.shape==(env.num_envs,6,6)
    assert torch.isfinite(env.ee_j_eef).all()
    assert torch.isfinite(env.obs_buf).all()
    if os.environ.get('AS2_SMOKE')=='1':
        # Cross-engine FK check uses actual PhysX joint and root states.
        from isaacgym.torch_utils import quat_apply
        from legged_gym.utils.urdf_chain import multiply
        fk_pos,fk_quat=env.arm_fk(env.dof_pos[:,12:18])
        predicted=env._get_ee_goal_spherical_center()+quat_apply(env.root_states[:,3:7],fk_pos)
        fk_error=float((predicted-env.ee_pos).norm(dim=-1).max())
        assert fk_error<.002, fk_error
        predicted_quat=multiply(env.root_states[:,3:7],fk_quat)
        fk_rotation_error=float((1-(predicted_quat*env.ee_orn).sum(dim=-1).abs()).abs().max())
        assert fk_rotation_error<1e-5, fk_rotation_error
        max_gripper_error=0.
        upright_gripper_error=0.
        total_reward=0.
        env._resample_ee_goal(torch.arange(env.num_envs,device=env.device))
        for step in range(350):
            sampled=torch.zeros(env.num_envs,18,device=env.device)
            sampled[:,12:]=.25
            original=sampled.clone()
            result=env.step(sampled)
            assert torch.equal(sampled,original), 'PPO action was modified by environment'
            assert torch.isfinite(result[0]).all(),step
            assert torch.isfinite(result[2]).all() and torch.isfinite(result[3]).all(),step
            max_gripper_error=max(max_gripper_error,float((env.dof_pos[:,-2:]-env.default_dof_pos[-2:]).abs().max()))
            upright=(env._get_body_orientation().abs().max(dim=-1).values<.4)&((env.root_states[:,2]-env.env_origins[:,2])>.22)
            if upright.any():
                upright_gripper_error=max(upright_gripper_error,float((env.dof_pos[upright,-2:]-env.default_dof_pos[-2:]).abs().max()))
            total_reward+=float(result[2].abs().sum())
        # Untrained zero-action robots can fall on the fingers. Contact-driven
        # compression is recorded separately from upright open-hold accuracy.
        assert upright_gripper_error < .005, upright_gripper_error
        assert total_reward>0
        ids=torch.arange(env.num_envs,device=env.device)
        env.reset_idx(ids)
        print('AS2 forward rollout contracts PASS: 350 steps + explicit reset',flush=True)
        (path/'smoke_contracts.json').write_text(json.dumps(dict(
            num_envs=env.num_envs,num_dofs=env.num_dofs,num_actions=env.num_actions,
            observations=list(env.obs_buf.shape),jacobian=list(env.ee_j_eef.shape),
            dof_names=env.dof_names,body_names=env.body_names,gripper_max_error_m=max_gripper_error,
            upright_gripper_max_error_m=upright_gripper_error,
            sum_abs_rollout_reward=total_reward,asset=cfg.asset.file,
            fk_position_error_m=fk_error,fk_quaternion_dot_error=fk_rotation_error,
            sampler='B1 spherical',collision_rejection='EE AABB + arm/body OBB + IK',
            collision_retry_fallbacks=env.pose_fallback_count),indent=2))
    runner,_,_=task_registry.make_alg_runner(log_root=str(path),env=env,name=args.task,args=args)
    runner.learn(num_learning_iterations=runner.max_iterations if hasattr(runner,'max_iterations') else (args.max_iterations or train_cfg.runner.max_iterations),init_at_random_ep_len=True)
    print('AS2 training completed',flush=True)
    wandb.finish()


if __name__=='__main__':main()
