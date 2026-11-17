"""Measure learned velocity tracking; standing alone is not gait validation."""
import argparse,json,hashlib,fcntl,os
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
    parser.add_argument('--gait-preset',choices=['original','velocity','jerk','stop','data_stop','speed','stable','stable_grid','stable_contact','progress','progress_gain2','gaussian_gain2','original_proxy','stable_proxy','progress_gain2_proxy','stable_proxy_ground','progress_gain2_proxy_ground','progress_gain2_proxy_yaw4','progress_gain2_proxy_ground_floor','progress_gain2_proxy_smooth_yaw4','progress_gain2_proxy_smooth_ground_floor_yaw4','progress_gain2_proxy_smooth_yaw4_contactmatch','progress_gain2_proxy_smooth_yaw4_footground','progress_gain2_proxy_smooth_yaw4_footground_deadband','progress_gain2_proxy_smooth_yaw4_contactmatch_deadband'])
    parser.add_argument('--terrain',choices=['plane','trimesh'],default='plane')
    parser.add_argument('--seed',type=int,default=20261030)
    parser.add_argument('--envs-per-command',type=int,default=8)
    parser.add_argument('--export-terrain',action='store_true')
    args=parser.parse_args()
    evaluation_lock=None
    if args.output and any(part in str(args.output) for part in ['gait_full_env_v1','gait_reward_proxy_v2']):
        evaluation_lock=(ROOT/'experiments/gait_full_env_v1/gpu_evaluation.lock').open('a')
        fcntl.flock(evaluation_lock,fcntl.LOCK_EX)
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
    group=args.envs_per_command
    if group<1:parser.error("--envs-per-command must be positive")
    cfg.env.num_envs=4*group
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
    targets=torch.tensor([[0,0,0],[args.vx,0,0],[0,0,args.yaw],[args.vx,0,args.yaw]],device=e.device).repeat_interleave(group,0)
    e._resample_commands=lambda ids:e.commands.__setitem__(ids,targets[ids])
    e.commands[:]=targets;e.compute_observations();rows=[];frames=[];original=e.check_termination
    arm_bodies=[i for i,name in enumerate(e.body_names) if name.startswith('piper_') and name not in ('piper_base_link','piper_link1')]
    shadow_support=(e._support_ground.clone() if hasattr(e,'_support_ground') else
        torch.where(e.foot_contacts_from_sensor,e.rigid_body_state[:,e.feet_indices,2]-.028,e.env_origins[:,2:3].repeat(1,4)))
    def check():
        original();rp=e._get_body_orientation();height=e.root_states[:,2]-e.env_origins[:,2]
        foot_state=e.rigid_body_state[:,e.feet_indices]
        offset=torch.zeros_like(foot_state[:,:,7:10]);offset[:,:,2]=-.028
        point_speed=(foot_state[:,:,7:10]+torch.cross(foot_state[:,:,10:13],offset,dim=-1))[:,:,:2].norm(dim=-1)
        foot_z=e.rigid_body_state[:,e.feet_indices,2]
        shadow_support.copy_(torch.where(e.foot_contacts_from_sensor,foot_z-.028,shadow_support))
        if args.terrain=='trimesh':
            ij=((e.rigid_body_state[:,e.feet_indices,:2]+e.terrain.cfg.border_size)/e.terrain.cfg.horizontal_scale).long()
            ix=ij[...,0].clamp(0,e.height_samples.shape[0]-2);iy=ij[...,1].clamp(0,e.height_samples.shape[1]-2)
            grid_ground=torch.minimum(torch.minimum(e.height_samples[ix,iy],e.height_samples[ix+1,iy]),e.height_samples[ix,iy+1])*e.terrain.cfg.vertical_scale
            support_ground=shadow_support
        else:grid_ground=torch.zeros_like(foot_z);support_ground=grid_ground
        target_lift=.04*torch.sin(2*torch.pi*(e.foot_indices-.5)).clamp(min=0) if hasattr(e,'foot_indices') else torch.zeros_like(foot_z)
        grid_height_reward=torch.exp(-(((foot_z-grid_ground-.028-target_lift)/.025)**2).mean(-1))*e._get_walking_cmd_mask()
        support_height_reward=torch.exp(-(((foot_z-support_ground-.028-target_lift)/.025)**2).mean(-1))*e._get_walking_cmd_mask()
        rows.append(torch.cat((e.base_lin_vel[:,0:1],e.base_ang_vel[:,2:3],e.reset_buf[:,None],e.time_out_buf[:,None],
            ((rp.abs().max(-1).values<.35)&(height>.22))[:,None],e.contact_forces[:,e.feet_indices,2],
            e.rigid_body_state[:,e.feet_indices,2]-e.env_origins[:,2:3]-.028,
            e.rigid_body_state[:,e.feet_indices,7:9].norm(dim=-1),e.desired_contact_states,
            e.torques[:,:12],(e.ee_pos-applied_goal).norm(dim=-1)[:,None],
            torch.rad2deg(2*torch.acos((e.ee_orn*applied_orientation).sum(-1).abs().clamp(0,1)))[:,None],point_speed,
            e.contact_forces[:,arm_bodies].norm(dim=-1).max(-1).values[:,None],
            rp,e.root_states[:,2:3],height[:,None],
            e.contact_forces[:,e.body_names.index('base_link')].norm(dim=-1)[:,None],e.foot_contacts_from_sensor.float(),e.rigid_body_state[:,e.feet_indices,:3].reshape(e.num_envs,-1),grid_ground,support_ground,target_lift,grid_height_reward[:,None],support_height_reward[:,None],e.box_root_state[:,:7],e.base_lin_vel[:,1:2],e.root_states[:,:7],foot_state[:,:,7:13].reshape(e.num_envs,-1),e.box_root_state[:,7:13]),dim=-1).cpu().numpy())
        if len(rows)%2==0:
            ids=torch.tensor([0,group,2*group,3*group],device=e.device)
            frames.append(dict(root=(e.root_states[ids,:7]-torch.cat((e.env_origins[ids],torch.zeros(4,4,device=e.device)),dim=-1)).cpu().numpy(),q=e.dof_pos[ids].cpu().numpy(),
                box=(e.box_root_state[ids,:7]-torch.cat((e.env_origins[ids],torch.zeros(4,4,device=e.device)),dim=-1)).cpu().numpy(),
                target=(applied_goal[ids]-e.env_origins[ids]).cpu().numpy(),ee=(e.ee_pos[ids]-e.env_origins[ids]).cpu().numpy(),
                step=len(rows),iteration=r.current_learning_iteration,random=applied_random[ids].cpu().numpy(),
                upright=((rp[ids].abs().max(-1).values<.35)&(height[ids]>.22)).cpu().numpy(),
                position_error=(e.ee_pos[ids]-applied_goal[ids]).norm(dim=-1).cpu().numpy(),
                vx=e.base_lin_vel[ids,0].cpu().numpy(),yaw_rate=e.base_ang_vel[ids,2].cpu().numpy(),commands=e.commands[ids].cpu().numpy()))
    e.check_termination=check
    reward_rows=[];reward_values={};reward_reconstruction_error=[0.]
    if args.output and any(part in str(args.output) for part in ['gait_full_env_v1','gait_reward_proxy_v2']):
        for index,name in enumerate(e.reward_names):
            fn=e.reward_functions[index]
            def captured(fn=fn,name=name):
                result=fn();value=result[0]
                if not torch.is_tensor(value):value=torch.full((e.num_envs,),float(value),device=e.device)
                reward_values[name]=value.detach()*e.reward_scales[name]
                return result
            e.reward_functions[index]=captured
        compute=e.compute_reward
        def measured_reward():
            compute()
            batch=torch.stack([reward_values[name] for name in e.reward_names]+[e.rew_buf],dim=-1).cpu().numpy()
            reward_reconstruction_error[0]=max(reward_reconstruction_error[0],float(np.abs(batch[:,:-1].sum(-1)/100.-batch[:,-1]).max()))
            reward_rows.append(batch[:,:-1])
        e.compute_reward=measured_reward
    with torch.no_grad():
        for evaluation_step in range(int(args.seconds/e.dt)):
            if evaluation_step%250==0:print('Evaluation simulation seconds %.1f / %.1f'%(evaluation_step*e.dt,args.seconds),flush=True)
            if not args.arm_goals:e.goal_timer.zero_()
            e.commands[:]=targets
            applied_goal=e.curr_ee_goal_cart_world.clone();applied_orientation=e.ee_goal_orn_quat.clone()
            applied_random=(e.ee_goal_sphere-e.init_end_ee_sphere).norm(dim=-1)>1e-4
            action=r.alg.actor_critic.act_inference(e.obs_buf,hist_encoding=args.encoder=='history')
            e.step(action)
            reset_ids=e.reset_buf.nonzero(as_tuple=False).flatten()
            shadow_support[reset_ids]=e.env_origins[reset_ids,2:3]
    x=np.stack(rows);result={}
    for i,name in enumerate(['stand','forward','turn','forward_turn']):
        v=x[100:,i*group:(i+1)*group];t=targets[i*group].cpu().numpy()
        result[name]=dict(command_vx_m_s=float(t[0]),command_yaw_rad_s=float(t[2]),
            command_above_walking_threshold=bool(abs(float(t[0]))>cfg.commands.lin_vel_x_clip or abs(float(t[2]))>cfg.commands.ang_vel_yaw_clip),
            mean_vx_m_s=float(v[:,:,0].mean()),mean_yaw_rad_s=float(v[:,:,1].mean()),
            mean_vy_m_s=float(v[:,:,82].mean()),vy_abs_error_m_s=float(abs(v[:,:,82]).mean()),
            vx_abs_error_m_s=float(abs(v[:,:,0]-t[0]).mean()),yaw_abs_error_rad_s=float(abs(v[:,:,1]-t[2]).mean()),
            upright_fraction=float(v[:,:,4].mean()),fall_resets=int(((v[:,:,2]>.5)&(v[:,:,3]<.5)).sum()),
            per_foot_loaded_fraction=(v[:,:,5:9]>5).mean((0,1)).tolist(),
            foot_clearance_p95_m=np.quantile(v[:,:,9:13],.95,axis=(0,1)).tolist(),
            foot_clearance_p99_m=np.quantile(v[:,:,9:13],.99,axis=(0,1)).tolist(),
            per_foot_unloaded_fraction=(v[:,:,5:9]<5).mean((0,1)).tolist(),
            applied_leg_torque_abs_max_Nm=float(abs(v[:,:,21:33]).max()),
            leg_torque_above_90pct_fraction=float((abs(v[:,:,21:33])>.9*e.torque_limits[:12].cpu().numpy()).mean()),
            ee_position_mean_m=float(v[:,:,33].mean()),ee_position_p95_m=float(np.quantile(v[:,:,33],.95)),
            ee_orientation_mean_deg=float(v[:,:,34].mean()),
            arm_contact_gt_1N_fraction=float((v[:,:,39]>1).mean()),
            arm_contact_max_N=float(v[:,:,39].max()),
            stance_center_speed_mean_m_s=float(v[:,:,13:17][v[:,:,5:9]>5].mean()) if (v[:,:,5:9]>5).any() else 0.,
            stance_slip_mean_m_s=float(v[:,:,35:39][v[:,:,5:9]>5].mean()) if (v[:,:,5:9]>5).any() else 0.,
            midstance_slip_mean_m_s=float(v[:,:,35:39][(v[:,:,5:9]>20)&(v[:,:,17:21]>.9)].mean()) if ((v[:,:,5:9]>20)&(v[:,:,17:21]>.9)).any() else 0.,
            phase_contact_agreement=float(((v[:,:,5:9]>5)==(v[:,:,17:21]>.5)).mean()))
    result['conditions']=f'Terrain={args.terrain}, DR={args.domain_rand}, deterministic {args.encoder} inference, random arm goals={args.arm_goals}, {cfg.env.num_envs} envs, {args.seconds}s; {group} envs per command.'
    result['task_variant']='vbc_port' if args.vbc_port else ('restricted_vbc_sampler' if args.vbc_sampler else 'hardware')
    if args.gait_preset:result['task_variant']='gait_study_'+args.gait_preset
    result['envs_per_command']=group
    result['environment_realization']=dict(terrain_sha256=hashlib.sha256(e.height_samples.cpu().numpy().tobytes()).hexdigest() if getattr(e,'height_samples',None) is not None else None, env_origins_sha256=hashlib.sha256(e.env_origins.cpu().numpy().tobytes()).hexdigest())
    result['environment_realization']['dr_sha256']=hashlib.sha256(b''.join(getattr(e,k).detach().cpu().numpy().tobytes() for k in ['mass_params_tensor','friction_coeffs_tensor','motor_strength'])).hexdigest()
    if args.terrain=='trimesh':
        h=hashlib.sha256();h.update(e.terrain.vertices.tobytes());h.update(e.terrain.triangles.tobytes());result['environment_realization']['triangle_mesh_sha256']=h.hexdigest()
    result['sample_columns_extra']=dict(contact_presence_proxy=[45,49],foot_world_positions=[49,61],grid_ground_z=[61,65],last_support_ground_z=[65,69],target_lift=[69,73],counterfactual_grid_height_reward=73,counterfactual_support_height_reward=74,box_world_pose=[75,82],base_vy=82,root_world_pose=[83,90],foot_world_linear_angular_velocity=[90,114],box_world_linear_angular_velocity=[114,120])
    result['leg_torque_limits_Nm']=e.torque_limits[:12].cpu().numpy().tolist()
    result['sample_schema_width']=int(x.shape[-1])
    result['completed_updates']=r.current_learning_iteration
    result['checkpoint']=str(Path(args.checkpoint).resolve())
    result['walking_thresholds']=dict(lin_vel_x=cfg.commands.lin_vel_x_clip,ang_vel_yaw=cfg.commands.ang_vel_yaw_clip)
    result['clearance_reference']='Foot height relative to environment origin, accurate on plane; not local terrain clearance on trimesh.'
    result['force_reporting_note']='Raw net contact forces can be missed on triangle-mesh terrain. Existing foot sensor contact bits are presence proxies, not calibrated ground reaction forces.'
    result['slip_definition']='Estimated spherical-foot bottom point world velocity v_com + omega cross [0,0,-0.028], loaded when Fz>5N; not raw foot-center velocity.'
    out=Path(args.output) if args.output else ROOT/'audit_hw';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(str(out/'policy_evaluation.npz'),samples=x)
    if reward_rows:
        result['reward_breakdown_reconstruction_max_abs_error']=reward_reconstruction_error[0]
        values=np.stack(reward_rows)
        np.savez_compressed(str(out/'reward_breakdown.npz'),weighted_rewards=values,names=np.array(e.reward_names))
        breakdown={name:{reward:float(values[100:,i*group:(i+1)*group,j].mean()) for j,reward in enumerate(e.reward_names)}
                   for i,name in enumerate(['stand','forward','turn','forward_turn'])}
        (out/'reward_breakdown.json').write_text(json.dumps(breakdown,indent=2))
    np.savez_compressed(str(out/'evaluation_dr.npz'),mass_params=e.mass_params_tensor.cpu().numpy(),friction=e.friction_coeffs_tensor.cpu().numpy(),motor_strength=e.motor_strength.cpu().numpy(),env_origins=e.env_origins.cpu().numpy())
    if frames:
        np.savez_compressed(str(out/'trajectories.npz'),**{k:np.stack([v[k] for v in frames]) for k in frames[0]})
        (out/'metadata.json').write_text(json.dumps(dict(video_ids=[0,group,2*group,3*group],dof_names=e.dof_names,encoder=args.encoder,box_size_m=cfg.box.box_size)))
    if args.output and 'gait_reward_proxy_v2' in str(args.output) and args.terrain=='trimesh':
        cache=ROOT/'experiments/gait_reward_proxy_v2/terrain_cache'/(result['environment_realization']['triangle_mesh_sha256']+'.npz')
        cache.parent.mkdir(parents=True,exist_ok=True)
        if not cache.exists():
            temporary=cache.with_suffix('.tmp')
            with temporary.open('wb') as file:np.savez_compressed(file,vertices=e.terrain.vertices,triangles=e.terrain.triangles,border_size=np.array(e.terrain.cfg.border_size))
            os.replace(str(temporary),str(cache))
        result['actual_terrain_cache']=str(cache)
    if args.export_terrain and args.terrain=='trimesh':
        vertices=e.terrain.vertices.copy();vertices[:,:2]-=e.terrain.cfg.border_size
        triangles=e.terrain.triangles
        for panel,env_id in enumerate([0,group,2*group,3*group]):
            origin=e.env_origins[env_id].cpu().numpy();v=vertices-origin
            xy=np.stack([frame['root'][panel,:2] for frame in frames])
            lower=xy.min(0)-2.5;upper=xy.max(0)+2.5
            inside=((v[:,:2]>=lower)&(v[:,:2]<=upper)).all(-1)
            selected=triangles[inside[triangles].any(axis=1)]
            used,indices=np.unique(selected,return_inverse=True)
            mesh=out/('terrain_panel_%d.obj'%panel)
            with mesh.open('w') as f:
                np.savetxt(f,v[used],fmt='v %.7f %.7f %.7f')
                np.savetxt(f,indices.reshape(-1,3)+1,fmt='f %d %d %d')
        result['terrain_replay']='Actual simulator triangles, cropped around the entire recorded trajectory plus 2.5m margin, relative to initial environment origin.'
    (out/'policy_evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    e.gym.destroy_sim(e.sim)
    if reward_rows:assert reward_reconstruction_error[0]<1e-4,'Reward instrumentation failed to reconstruct the original reward buffer'


if __name__=='__main__':main()
