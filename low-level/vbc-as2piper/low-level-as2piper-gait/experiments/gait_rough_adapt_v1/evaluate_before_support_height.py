"""Measure learned velocity tracking; standing alone is not gait validation."""
import argparse,json
from pathlib import Path
import isaacgym
from isaacgym import gymapi,gymutil,gymtorch
import torch,numpy as np
from legged_gym.envs.manip_loco.as2piper_hw import AS2PiPERHardware
from legged_gym.envs.manip_loco.as2piper_hw_config import AS2PiPERHardwareCfg,AS2PiPERHardwareCfgPPO
from legged_gym.utils.helpers import class_to_dict,set_seed
from legged_gym.utils.hw_runner import HardwareRunner

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint')
    parser.add_argument('--output');parser.add_argument('--encoder',choices=['history','privileged'],default='history')
    parser.add_argument('--vx',type=float);parser.add_argument('--yaw',type=float)
    parser.add_argument('--seconds',type=float,default=10.)
    parser.add_argument('--arm-goals',action='store_true')
    parser.add_argument('--domain-rand',action='store_true')
    parser.add_argument('--vbc-sampler', action='store_true')
    parser.add_argument('--vbc-port', action='store_true')
    parser.add_argument('--gait-preset',choices=['original','velocity','jerk','stop','data_stop','speed','stable'])
    parser.add_argument('--terrain',choices=['plane','trimesh'],default='plane')
    parser.add_argument('--seed',type=int,default=20261030)
    args=parser.parse_args()
    if args.vx is None:args.vx=.25 if args.vbc_port else .15
    if args.yaw is None:args.yaw=.6 if args.vbc_port else .25
    if args.vbc_sampler and args.vbc_port:parser.error('Choose one task variant')
    set_seed(args.seed)
    env_class=AS2PiPERHardware
    cfg=AS2PiPERHardwareCfg();ppo_cfg=AS2PiPERHardwareCfgPPO()
    if args.vbc_sampler:
        from legged_gym.envs.manip_loco.as2piper_walk_wbc_vbc import AS2PiPERWalkWBCVBC,AS2PiPERWalkWBCVBCCfg
        env_class=AS2PiPERWalkWBCVBC;cfg=AS2PiPERWalkWBCVBCCfg()
    if args.vbc_port:
        from legged_gym.envs.manip_loco.as2piper_vbc_port import AS2PiPERVBCPort,AS2PiPERVBCPortCfg,AS2PiPERVBCPortCfgPPO
        env_class=AS2PiPERVBCPort;cfg=AS2PiPERVBCPortCfg();ppo_cfg=AS2PiPERVBCPortCfgPPO()
    if args.gait_preset:
        from legged_gym.envs.manip_loco.as2piper_gait_study import AS2PiPERGaitStudy,AS2PiPERGaitStudyCfg,AS2PiPERGaitStudyCfgPPO,configure
        env_class=AS2PiPERGaitStudy;cfg=configure(AS2PiPERGaitStudyCfg(),args.gait_preset)
        ppo_cfg=AS2PiPERGaitStudyCfgPPO()
    cfg.env.num_envs=32
    cfg.terrain.mesh_type=args.terrain;cfg.terrain.measure_heights=args.terrain=='trimesh'
    if not args.domain_rand:
        cfg.init_state.leg_reset_fraction=0;cfg.init_state.init_vel_perturb_range=0
        for k in ['randomize_friction','randomize_base_mass','randomize_base_com','randomize_motor','randomize_gripper_mass','push_robots']:setattr(cfg.domain_rand,k,False)
    params=gymapi.SimParams();params.dt=cfg.sim.dt;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81);params.use_gpu_pipeline=True;params.physx.use_gpu=True
    for k in ['num_position_iterations','num_velocity_iterations','contact_offset','rest_offset','bounce_threshold_velocity','max_depenetration_velocity']:setattr(params.physx,k,getattr(cfg.sim.physx,k))
    gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
    e=env_class(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True)
    if (args.vbc_port or args.gait_preset) and not args.domain_rand:
        # Evaluation-only nominal reset; leave the production +/-20% reset intact.
        def nominal_reset(ids):
            e.dof_pos[ids]=e.default_dof_pos;e.dof_vel[ids]=0
            e.gym.set_dof_state_tensor(e.sim,gymtorch.unwrap_tensor(e.dof_state))
        e._reset_dofs=nominal_reset
    r=HardwareRunner(e,class_to_dict(ppo_cfg),None,'cuda:0');r.load(args.checkpoint);r.alg.actor_critic.eval()
    set_seed(args.seed)  # Loading a checkpoint restores training RNG; use a holdout sequence.
    targets=torch.tensor([[0,0,0],[args.vx,0,0],[0,0,args.yaw],[args.vx,0,args.yaw]],device=e.device).repeat_interleave(8,0)
    e._resample_commands=lambda ids:e.commands.__setitem__(ids,targets[ids])
    e.commands[:]=targets;e.compute_observations();rows=[];frames=[];original=e.check_termination
    arm_bodies=[i for i,name in enumerate(e.body_names) if name.startswith('piper_') and name not in ('piper_base_link','piper_link1')]
    def check():
        original();rp=e._get_body_orientation();height=e.root_states[:,2]-e.env_origins[:,2]
        foot_state=e.rigid_body_state[:,e.feet_indices]
        offset=torch.zeros_like(foot_state[:,:,7:10]);offset[:,:,2]=-.028
        point_speed=(foot_state[:,:,7:10]+torch.cross(foot_state[:,:,10:13],offset,dim=-1))[:,:,:2].norm(dim=-1)
        rows.append(torch.cat((e.base_lin_vel[:,0:1],e.base_ang_vel[:,2:3],e.reset_buf[:,None],e.time_out_buf[:,None],
            ((rp.abs().max(-1).values<.35)&(height>.22))[:,None],e.contact_forces[:,e.feet_indices,2],
            e.rigid_body_state[:,e.feet_indices,2]-e.env_origins[:,2:3]-.028,
            e.rigid_body_state[:,e.feet_indices,7:9].norm(dim=-1),e.desired_contact_states,
            e.torques[:,:12],(e.ee_pos-applied_goal).norm(dim=-1)[:,None],
            torch.rad2deg(2*torch.acos((e.ee_orn*applied_orientation).sum(-1).abs().clamp(0,1)))[:,None],point_speed,
            e.contact_forces[:,arm_bodies].norm(dim=-1).max(-1).values[:,None]),dim=-1).cpu().numpy())
        if len(rows)%2==0:
            ids=torch.tensor([0,8,16,24],device=e.device)
            frames.append(dict(root=(e.root_states[ids,:7]-torch.cat((e.env_origins[ids],torch.zeros(4,4,device=e.device)),dim=-1)).cpu().numpy(),q=e.dof_pos[ids].cpu().numpy(),
                box=(e.box_root_state[ids,:7]-torch.cat((e.env_origins[ids],torch.zeros(4,4,device=e.device)),dim=-1)).cpu().numpy(),
                target=(applied_goal[ids]-e.env_origins[ids]).cpu().numpy(),ee=(e.ee_pos[ids]-e.env_origins[ids]).cpu().numpy(),
                step=len(rows),iteration=r.current_learning_iteration,random=applied_random[ids].cpu().numpy(),
                upright=((rp[ids].abs().max(-1).values<.35)&(height[ids]>.22)).cpu().numpy(),
                position_error=(e.ee_pos[ids]-applied_goal[ids]).norm(dim=-1).cpu().numpy(),
                vx=e.base_lin_vel[ids,0].cpu().numpy(),yaw_rate=e.base_ang_vel[ids,2].cpu().numpy(),commands=e.commands[ids].cpu().numpy()))
    e.check_termination=check
    with torch.no_grad():
        for _ in range(int(args.seconds/e.dt)):
            if not args.arm_goals:e.goal_timer.zero_()
            e.commands[:]=targets
            applied_goal=e.curr_ee_goal_cart_world.clone();applied_orientation=e.ee_goal_orn_quat.clone()
            applied_random=(e.ee_goal_sphere-e.init_end_ee_sphere).norm(dim=-1)>1e-4
            action=r.alg.actor_critic.act_inference(e.obs_buf,hist_encoding=args.encoder=='history')
            e.step(action)
    x=np.stack(rows);result={}
    for i,name in enumerate(['stand','forward','turn','forward_turn']):
        v=x[100:,i*8:(i+1)*8];t=targets[i*8].cpu().numpy()
        result[name]=dict(command_vx_m_s=float(t[0]),command_yaw_rad_s=float(t[2]),
            command_above_walking_threshold=bool(abs(float(t[0]))>cfg.commands.lin_vel_x_clip or abs(float(t[2]))>cfg.commands.ang_vel_yaw_clip),
            mean_vx_m_s=float(v[:,:,0].mean()),mean_yaw_rad_s=float(v[:,:,1].mean()),
            vx_abs_error_m_s=float(abs(v[:,:,0]-t[0]).mean()),yaw_abs_error_rad_s=float(abs(v[:,:,1]-t[2]).mean()),
            upright_fraction=float(v[:,:,4].mean()),fall_resets=int(((v[:,:,2]>.5)&(v[:,:,3]<.5)).sum()),
            per_foot_loaded_fraction=(v[:,:,5:9]>5).mean((0,1)).tolist(),
            foot_clearance_p95_m=np.quantile(v[:,:,9:13],.95,axis=(0,1)).tolist(),
            foot_clearance_p99_m=np.quantile(v[:,:,9:13],.99,axis=(0,1)).tolist(),
            per_foot_unloaded_fraction=(v[:,:,5:9]<5).mean((0,1)).tolist(),
            applied_leg_torque_abs_max_Nm=float(abs(v[:,:,21:33]).max()),
            ee_position_mean_m=float(v[:,:,33].mean()),ee_position_p95_m=float(np.quantile(v[:,:,33],.95)),
            ee_orientation_mean_deg=float(v[:,:,34].mean()),
            arm_contact_gt_1N_fraction=float((v[:,:,39]>1).mean()),
            arm_contact_max_N=float(v[:,:,39].max()),
            stance_center_speed_mean_m_s=float(v[:,:,13:17][v[:,:,5:9]>5].mean()) if (v[:,:,5:9]>5).any() else 0.,
            stance_slip_mean_m_s=float(v[:,:,35:39][v[:,:,5:9]>5].mean()) if (v[:,:,5:9]>5).any() else 0.,
            midstance_slip_mean_m_s=float(v[:,:,35:39][(v[:,:,5:9]>20)&(v[:,:,17:21]>.9)].mean()) if ((v[:,:,5:9]>20)&(v[:,:,17:21]>.9)).any() else 0.,
            phase_contact_agreement=float(((v[:,:,5:9]>5)==(v[:,:,17:21]>.5)).mean()))
    result['conditions']=f'Terrain={args.terrain}, DR={args.domain_rand}, deterministic {args.encoder} inference, random arm goals={args.arm_goals}, 32 envs, {args.seconds}s; 8 envs per command.'
    result['task_variant']='vbc_port' if args.vbc_port else ('restricted_vbc_sampler' if args.vbc_sampler else 'hardware')
    if args.gait_preset:result['task_variant']='gait_study_'+args.gait_preset
    result['checkpoint']=str(Path(args.checkpoint).resolve())
    result['walking_thresholds']=dict(lin_vel_x=cfg.commands.lin_vel_x_clip,ang_vel_yaw=cfg.commands.ang_vel_yaw_clip)
    result['clearance_reference']='Foot height relative to environment origin, accurate on plane; not local terrain clearance on trimesh.'
    result['slip_definition']='Estimated spherical-foot bottom point world velocity v_com + omega cross [0,0,-0.028], loaded when Fz>5N; not raw foot-center velocity.'
    out=Path(args.output) if args.output else ROOT/'audit_hw';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(str(out/'policy_evaluation.npz'),samples=x)
    if frames:
        np.savez_compressed(str(out/'trajectories.npz'),**{k:np.stack([v[k] for v in frames]) for k in frames[0]})
        (out/'metadata.json').write_text(json.dumps(dict(video_ids=[0,8,16,24],dof_names=e.dof_names,encoder=args.encoder,box_size_m=cfg.box.box_size)))
    (out/'policy_evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    e.gym.destroy_sim(e.sim)


if __name__=='__main__':main()
