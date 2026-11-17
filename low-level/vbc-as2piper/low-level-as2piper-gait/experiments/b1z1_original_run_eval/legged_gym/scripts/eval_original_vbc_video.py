"""Evaluate and save original B1Z1 VBC policy states without touching source run."""
import os, json
from pathlib import Path
import isaacgym
import numpy as np
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

ROOT=Path(LEGGED_GYM_ROOT_DIR)
OUT=ROOT/'videos';OUT.mkdir(parents=True,exist_ok=True)

def phases(mode):
    if mode=='ee':
        return [('stand + sampled EE poses',12.0,[0.,0.,0.])]
    return [
        ('standing',4.,[0.,0.,0.]),
        ('forward',5.,[.5,0.,0.]),
        ('reverse',5.,[-.3,0.,0.]),
        ('turn',5.,[0.,0.,.6]),
        ('forward + turn',5.,[.35,0.,.5]),
    ]

def main():
    args=get_args()
    run=ROOT/'logs'/args.proj_name/args.exptid
    env_cfg,train_cfg=task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs=1
    env_cfg.terrain.num_rows=6;env_cfg.terrain.num_cols=3
    env_cfg.domain_rand.push_robots=False
    env_cfg.domain_rand.randomize_base_mass=True
    env_cfg.domain_rand.randomize_base_com=False
    env_cfg.env.episode_length_s=3600.
    if args.flat_terrain:env_cfg.terrain.height=[0.,0.]
    env, _=task_registry.make_env(name=args.task,args=args,env_cfg=env_cfg)
    train_cfg.runner.resume=True
    runner,train_cfg,checkpoint,_=task_registry.make_alg_runner(log_root=str(run),env=env,name=args.task,args=args,train_cfg=train_cfg,return_log_dir=True)
    policy=runner.get_inference_policy(device=env.device,stochastic=False)
    mode=os.environ.get('B1_EVAL_MODE','all');ps=phases(mode)
    duration=float(os.environ.get('B1_EVAL_SECONDS','0'))
    if duration>0:ps=[(ps[0][0],duration,ps[0][2])]
    stem='b1z1_original_run01_45000_'+mode+('_flat' if args.flat_terrain else '_rough')
    dt=float(env.dt);obs=env.get_observations();samples=[];phase_ranges=[];step=0
    feet=env.feet_indices
    for phase_id,(label,seconds,command) in enumerate(ps):
        start=len(samples);n=int(seconds/dt)
        for _ in range(n):
            env.commands[:]=torch.tensor(command,device=env.device).view(1,-1)
            with torch.no_grad():actions=policy(obs.detach(),hist_encoding=True)
            obs,_,rew,arm_rew,dones,infos=env.step(actions.detach())
            if env.global_steps%2==0:
                origin=env.env_origins[0].detach().cpu().numpy()
                root=env.root_states[0,:7].detach().cpu().numpy().copy();root[:3]-=origin
                target=(env.curr_ee_goal_cart_world[0].detach().cpu().numpy()-origin)
                actual=(env.ee_pos[0].detach().cpu().numpy()-origin)
                q=env.dof_pos[0].detach().cpu().numpy().copy()
                gv=env.projected_gravity[0].detach().cpu().numpy().copy()
                feet_fz=env.contact_forces[0,feet,2].detach().cpu().numpy().copy()
                cmd=env.commands[0,:3].detach().cpu().numpy().copy()
                vel=env.base_lin_vel[0,:3].detach().cpu().numpy().copy()
                omg=env.base_ang_vel[0,:3].detach().cpu().numpy().copy()
                ee_err=float(np.linalg.norm(actual-target))
                samples.append((step*dt,phase_id,root,q,target,actual,cmd,vel,omg,gv,feet_fz,ee_err,int(dones[0].item())))
            step+=1
        phase_ranges.append({'name':label,'seconds':seconds,'command_vx_vy_yaw':command,'first_sample':start,'last_sample_exclusive':len(samples)})
    arrays={
      'time_s':np.array([r[0] for r in samples],dtype=np.float32),
      'phase_id':np.array([r[1] for r in samples],dtype=np.int32),
      'root_xyzw':np.stack([r[2] for r in samples]),'dof_pos':np.stack([r[3] for r in samples]),
      'ee_target_world':np.stack([r[4] for r in samples]),'ee_actual_world':np.stack([r[5] for r in samples]),
      'command_vxy_yaw':np.stack([r[6] for r in samples]),'base_lin_vel':np.stack([r[7] for r in samples]),
      'base_ang_vel':np.stack([r[8] for r in samples]),'projected_gravity':np.stack([r[9] for r in samples]),
      'feet_fz':np.stack([r[10] for r in samples]),'ee_position_error':np.array([r[11] for r in samples],dtype=np.float32),
      'done':np.array([r[12] for r in samples],dtype=np.int8)}
    np.savez_compressed(OUT/(stem+'_states.npz'),**arrays)
    terrain_obj=None
    if not args.flat_terrain:
        vertices=env.terrain.vertices.copy();triangles=env.terrain.triangles
        vertices[:,:2]-=env.terrain.cfg.border_size
        origin=env.env_origins[0].detach().cpu().numpy();local_vertices=vertices-origin
        xy=arrays['root_xyzw'][:,:2]
        lower=xy.min(0)-2.5;upper=xy.max(0)+2.5
        inside=((local_vertices[:,:2]>=lower)&(local_vertices[:,:2]<=upper)).all(-1)
        selected=triangles[inside[triangles].any(axis=1)]
        used,indices=np.unique(selected,return_inverse=True)
        terrain_obj=OUT/(stem+'_terrain.obj')
        with terrain_obj.open('w') as f:
            np.savetxt(f,local_vertices[used],fmt='v %.6f %.6f %.6f')
            np.savetxt(f,indices.reshape(-1,3)+1,fmt='f %d %d %d')
    summary=[]
    for p in phase_ranges:
        a,b=p['first_sample'],p['last_sample_exclusive'];lo=a+int(.2*max(1,b-a));hi=b
        s=arrays['projected_gravity'][lo:hi];
        summary.append({**{k:v for k,v in p.items() if not k.startswith('first_') and not k.startswith('last_')},
          'mean_base_vx_m_s':float(arrays['base_lin_vel'][lo:hi,0].mean()),
          'mean_base_vy_m_s':float(arrays['base_lin_vel'][lo:hi,1].mean()),
          'mean_yaw_rate_rad_s':float(arrays['base_ang_vel'][lo:hi,2].mean()),
          'mean_ee_error_m':float(arrays['ee_position_error'][lo:hi].mean()),
          'ee_error_p95_m':float(np.quantile(arrays['ee_position_error'][lo:hi],.95)),
          'upright_fraction':float((np.linalg.norm(s[:,:2],axis=-1)<.5).mean()),
          'mean_loaded_feet':float((arrays['feet_fz'][lo:hi]>5).sum(-1).mean()),
          'done_samples':int(arrays['done'][a:b].sum())})
    meta={'run':'b1z1-low/original-run01','checkpoint':str(run/'model_45000.pt'),'task':'B1Z1 original ManipLoco task copied read-only from vbc-legacy','source_config':'original run W&B config has --observe_gait_commands, seed=1','evaluation':'deterministic policy; one environment; original play.py test overrides (6x3 terrain grid, push off, base mass rand on, base CoM rand off); episode reset suppressed for continuous clips','terrain':'flat height range' if args.flat_terrain else 'original procedural terrain','terrain_obj':str(terrain_obj.resolve()) if terrain_obj else None,'dt_s':dt,'sample_rate_hz':25,'dof_names':env.dof_names,'phases':phase_ranges,'summary':summary}
    (OUT/(stem+'_metrics.json')).write_text(json.dumps(meta,indent=2))
    print(json.dumps({'states':str(OUT/(stem+'_states.npz')),'metrics':str(OUT/(stem+'_metrics.json')),'summary':summary},indent=2))
    env.gym.destroy_sim(env.sim)
if __name__=='__main__':main()
