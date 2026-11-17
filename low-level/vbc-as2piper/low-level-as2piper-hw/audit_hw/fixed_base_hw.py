"""PhysX control test: same arm sampler/IK/drives, fixed torso, no PPO/DR."""
import json,os
from pathlib import Path
import isaacgym
from isaacgym import gymapi,gymtorch,gymutil
from isaacgym.torch_utils import to_torch,orientation_error,quat_apply
import torch
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.manip_loco.as2piper_hw import AS2PiPERHardware as AS2PiPERForward,RefinedGuard as ArmCollisionGuard
from legged_gym.envs.manip_loco.as2piper_hw_config import AS2PiPERHardwareCfg as AS2PiPERPlateCfg
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.urdf_chain import URDFChain

ROOT=Path(LEGGED_GYM_ROOT_DIR)
OUT=Path(os.environ.get('EE_AUDIT_DIR',str(ROOT/'audit_hw/fixed_base')))
OUT.mkdir(parents=True,exist_ok=True)


def main():
    torch.manual_seed(20261029);np.random.seed(20261029)
    cfg=AS2PiPERPlateCfg();n=32;gym=gymapi.acquire_gym()
    params=gymapi.SimParams();params.dt=.005;params.substeps=1;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81)
    params.use_gpu_pipeline=True;params.physx.use_gpu=True
    for name in ['num_position_iterations','num_velocity_iterations','contact_offset','rest_offset','bounce_threshold_velocity','max_depenetration_velocity']:
        setattr(params.physx,name,getattr(cfg.sim.physx,name))
    gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
    sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,params);assert sim is not None
    plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1);gym.add_ground(sim,plane)
    asset_path=Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(ROOT)))
    options=gymapi.AssetOptions();options.fix_base_link=True;options.collapse_fixed_joints=True
    options.flip_visual_attachments=False;options.default_dof_drive_mode=gymapi.DOF_MODE_POS
    for name in ['replace_cylinder_with_capsule','armature','thickness','density','angular_damping','linear_damping','max_angular_velocity','max_linear_velocity','disable_gravity']:
        setattr(options,name,getattr(cfg.asset,name))
    asset=gym.load_asset(sim,str(asset_path.parent),asset_path.name,options)
    names=gym.get_asset_dof_names(asset);bodies=gym.get_asset_rigid_body_names(asset);tcp=bodies.index(cfg.asset.gripper_name)
    props=gym.get_asset_dof_properties(asset)
    props['driveMode'][:]=gymapi.DOF_MODE_POS;props['stiffness'][:12]=80;props['damping'][:12]=2
    props['stiffness'][12:18]=400;props['damping'][12:18]=40
    props['stiffness'][-2:]=1800;props['damping'][-2:]=35
    home=np.array([cfg.init_state.default_joint_angles[name] for name in names],dtype=np.float32)
    envs=[];actors=[]
    for i in range(n):
        env=gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,2),8)
        pose=gymapi.Transform();pose.p=gymapi.Vec3(0,0,.3244069024246581)
        actor=gym.create_actor(env,asset,pose,'robot_dog',i,0)
        gym.set_actor_dof_properties(env,actor,props);gym.enable_actor_dof_force_sensors(env,actor)
        state=np.zeros(20,dtype=gymapi.DofState.dtype);state['pos']=home
        gym.set_actor_dof_states(env,actor,state,gymapi.STATE_ALL)
        envs.append(env);actors.append(actor)
    gym.prepare_sim(sim)
    root=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(n,13)
    state=gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(n,20,2)
    rigid=gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).view(n,len(bodies),13)
    jac=gymtorch.wrap_tensor(gym.acquire_jacobian_tensor(sim,'robot_dog'))
    force=gymtorch.wrap_tensor(gym.acquire_dof_force_tensor(sim)).view(n,20)
    contact=gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(n,len(bodies),3)
    target=to_torch(home,device='cuda:0').repeat(n,1)
    def refresh():
        gym.refresh_dof_state_tensor(sim);gym.refresh_actor_root_state_tensor(sim);gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim);gym.refresh_dof_force_tensor(sim);gym.refresh_net_contact_force_tensor(sim)
    gym.set_dof_position_target_tensor(sim,gymtorch.unwrap_tensor(target))
    gym.simulate(sim);gym.fetch_results(sim,True);refresh()
    # A minimal state object reuses the actual AS2/B1 methods, without changing
    # the training task or its settings.
    e=object.__new__(AS2PiPERForward);e.cfg=cfg;e.device='cuda:0';e.num_envs=n;e.dt=.02
    e.root_states=root;e.dof_pos=state[:,:,0];e.ee_pos=rigid[:,tcp,:3];e.ee_orn=rigid[:,tcp,3:7]
    e.env_origins=root[:,:3].clone();e.env_origins[:,2]=0
    e.arm_base_offset=to_torch(cfg.asset.arm_base_offset,device=e.device).repeat(n,1)
    e.base_yaw_quat=torch.zeros(n,4,device=e.device);e.base_yaw_quat[:,3]=1
    e.ee_j_eef=jac[:,tcp-1,:,12:18]  # Fixed-base Jacobian excludes root body.
    e.collision_guard=ArmCollisionGuard(asset_path,ROOT/'resources/commands/arm_collision.json',e.device)
    e.goal_ee_ranges=class_to_dict(cfg.goal_ee.ranges)
    e.init_start_ee_sphere=to_torch(cfg.goal_ee.ranges.init_pos_start,device=e.device)[None,:]
    e.init_end_ee_sphere=to_torch(cfg.goal_ee.ranges.init_pos_end,device=e.device)[None,:]
    for field in ['ee_start_sphere','ee_goal_sphere','ee_goal_cart','curr_ee_goal_cart','curr_ee_goal_cart_world','curr_ee_goal_sphere','ee_goal_orn_delta_rpy']:
        setattr(e,field,torch.zeros(n,3,device=e.device))
    e.ee_goal_orn_quat=torch.zeros(n,4,device=e.device);e.ee_goal_orn_quat[:,3]=1
    e.traj_timesteps=torch.empty(n,device=e.device).uniform_(1,3)/e.dt
    e.traj_total_timesteps=e.traj_timesteps+torch.empty(n,device=e.device).uniform_(.5,2)/e.dt
    e.goal_timer=torch.zeros(n,device=e.device);e.commands=torch.zeros(n,3,device=e.device);e.stop_update_goal=False
    e.collision_lower_limits=to_torch(cfg.goal_ee.collision_lower_limits,device=e.device)
    e.collision_upper_limits=to_torch(cfg.goal_ee.collision_upper_limits,device=e.device)
    e.underground_limit=cfg.goal_ee.underground_limit;e.num_collision_check_samples=10
    e.collision_check_t=torch.linspace(0,1,10,device=e.device)[None,None,:]
    e.pose_requests=0;e.pose_accepted=0;e.pose_fallback_count=0;e._last_rejected_ids=torch.empty(0,device=e.device,dtype=torch.long)
    all_ids=torch.arange(n,device=e.device);e._resample_ee_goal(all_ids,is_init=True)
    rows=[];traces=[];arm_body=[i for i,b in enumerate(bodies) if b.startswith('piper_')]
    for step in range(750):
        if step==150:e._resample_ee_goal(all_ids)  # Three-second home settling.
        applied=e.curr_ee_goal_cart_world.clone();orientation=e.ee_goal_orn_quat.clone()
        pose_error=torch.cat((applied-e.ee_pos,orientation_error(orientation,e.ee_orn)),dim=-1)[:,:,None]
        target[:,12:18]=torch.clamp(e._control_ik(pose_error)+state[:,12:18,0],to_torch(props['lower'][12:18],device=e.device),to_torch(props['upper'][12:18],device=e.device))
        for _ in range(4):
            gym.set_dof_position_target_tensor(sim,gymtorch.unwrap_tensor(target));gym.simulate(sim);gym.fetch_results(sim,True);refresh()
        position=(e.ee_pos-applied).norm(dim=-1)
        angle=torch.rad2deg(2*torch.acos((orientation*e.ee_orn).sum(-1).abs().clamp(0,1)))
        hold=e.goal_timer>=e.traj_timesteps
        contacts=contact[:,arm_body].norm(dim=-1).max(dim=-1).values
        rows.append(torch.stack((position,angle,hold.float(),contacts,force[:,12:18].abs().max(dim=-1).values),dim=-1).cpu().numpy())
        if step%2==0:
            traces.append(dict(root=root[:4,:7].cpu().numpy(),q=state[:4,:,0].cpu().numpy(),target=applied[:4].cpu().numpy(),ee=e.ee_pos[:4].cpu().numpy(),step=step))
        e._update_curr_ee_goal()
        if step%250==0:print('FIXED_BASE',step,float(position.mean()),float(angle.mean()),flush=True)
    a=np.stack(rows);np.savez_compressed(str(OUT/'tracking.npz'),samples=a)
    np.savez_compressed(str(OUT/'trajectories.npz'),**{k:np.stack([r[k] for r in traces]) for k in traces[0]})
    summary={}
    # Some B1 timers expire before 3 seconds; this early window is mixed,
    # not a pure home-only measurement. Random windows below exclude it.
    for label,selection in [('early_mixed',a[100:150]),('random_all',a[200:]),('random_hold',a[200:][a[200:,:,2]>.5])]:
        v=selection.reshape(-1,5)
        summary[label]=dict(count=len(v),position_mean_m=float(v[:,0].mean()),position_p95_m=float(np.quantile(v[:,0],.95)),
            orientation_mean_deg=float(v[:,1].mean()),orientation_p95_deg=float(np.quantile(v[:,1],.95)),
            contact_gt_1N_fraction=float((v[:,3]>1).mean()),force_abs_max_p95_Nm=float(np.quantile(v[:,4],.95)))
    summary.update(pose_requests=e.pose_requests,pose_accepted=e.pose_accepted,pose_fallback_count=e.pose_fallback_count,conditions='Fixed torso, nominal mass/COM, gravity on; same URDF, spherical sampler, DLS IK, 400/40 arm and 1800/35 gripper drives. Not locomotion.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    gym.destroy_sim(sim)


if __name__=='__main__':main()
