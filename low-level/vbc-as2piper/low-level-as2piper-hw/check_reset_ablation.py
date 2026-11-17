"""One-factor reset comparisons under the same zero-action plane dynamics."""
import argparse,json,time
from pathlib import Path
import isaacgym
from isaacgym import gymapi,gymutil
import torch,numpy as np
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.envs.manip_loco.as2piper_walk_wbc_vbc import AS2PiPERWalkWBCVBC,AS2PiPERWalkWBCVBCCfg
from legged_gym.envs.manip_loco.as2piper_vbc_port import AS2PiPERVBCPort,AS2PiPERVBCPortCfg
from legged_gym.utils.helpers import class_to_dict


class JointReset(AS2PiPERWalkWBCVBC):
    _reset_dofs=ManipLoco._reset_dofs


class FixedReset(JointReset):
    _reset_root_states=ManipLoco._reset_root_states


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['current','joint20','velocity','fixed_height','port'],required=True)
    parser.add_argument('--num-envs',type=int,default=128);parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--noise',choices=['none','original','similarity'],default='none')
    args=parser.parse_args();torch.manual_seed(20261029);np.random.seed(20261029)
    cls={'current':AS2PiPERWalkWBCVBC,'joint20':JointReset,'velocity':JointReset,'fixed_height':FixedReset,'port':AS2PiPERVBCPort}[args.case]
    cfg=AS2PiPERVBCPortCfg() if args.case=='port' else AS2PiPERWalkWBCVBCCfg()
    cfg.env.num_envs=args.num_envs;cfg.terrain.mesh_type='plane';cfg.terrain.measure_heights=False
    if args.case in ['velocity','fixed_height']:cfg.init_state.init_vel_perturb_range=.1
    for k in ['randomize_base_mass','randomize_base_com','randomize_gripper_mass','randomize_friction','randomize_motor','push_robots']:setattr(cfg.domain_rand,k,False)
    params=gymapi.SimParams();params.dt=.005;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81)
    params.use_gpu_pipeline=True;params.physx.use_gpu=True
    for key in ['num_position_iterations','num_velocity_iterations','contact_offset','rest_offset','bounce_threshold_velocity','max_depenetration_velocity']:setattr(params.physx,key,getattr(cfg.sim.physx,key))
    gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
    env=cls(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True)
    env.reset()
    values=[];original=env.check_termination
    def capture():
        original();rp=env._get_body_orientation()
        values.append(torch.stack((env.reset_buf.float(),env.time_out_buf.float(),rp.abs().max(-1).values,
            env.contact_forces[:,env.feet_indices,2].max(-1).values,env.torques[:,:12].abs().max(-1).values,
            (env.ee_pos-applied).norm(dim=-1),env.root_states[:,2]),-1).cpu().numpy())
    env.check_termination=capture
    noise_std=torch.tensor([.8,1.,1.]*4,device=env.device)
    if args.noise=='similarity':
        from legged_gym.envs.manip_loco.as2piper_vbc_port import MASS,LENGTH,HW
        ratio=torch.tensor([80*MASS*LENGTH/k for k in HW['leg_kp']]*4,device=env.device)
        noise_std*=ratio
    with torch.no_grad():
        start=time.monotonic()
        for step in range(args.steps):
            env.goal_timer.zero_();env.commands.zero_();applied=env.curr_ee_goal_cart_world.clone()
            action=torch.zeros(env.num_envs,18,device=env.device)
            if args.noise!='none':action[:,:12]=torch.randn(env.num_envs,12,device=env.device)*noise_std
            env.step(action)
    data=np.stack(values);v=data[5:]
    result=dict(case=args.case,noise=args.noise,physics_config=class_to_dict(cfg.sim),noise_std=noise_std[:3].tolist() if args.noise!='none' else None,num_envs=env.num_envs,seconds=args.steps*.02,
        resets_after_initial5steps=int(((v[:,:,0]>.5)&(v[:,:,1]<.5)).sum()),
        roll_pitch_p95_rad=float(np.quantile(v[:,:,2],.95)),
        max_foot_force_N=float(v[:,:,3].max()),max_foot_force_including_initial_N=float(data[:,:,3].max()),
        max_applied_leg_torque_Nm=float(v[:,:,4].max()),ee_position_mean_m=float(v[:,:,5].mean()),
        mean_root_z_m=float(v[:,:,6].mean()),wall_s=time.monotonic()-start,
        conditions='No trained policy/DR; '+('zero actions' if args.noise=='none' else 'Gaussian actions')+'; first five samples excluded from aggregate.')
    path=Path(__file__).resolve().parent/'audit_hw/vbc_port_audit_20261029'/('reset_'+args.case+('' if args.noise=='none' else '_'+args.noise)+'.json')
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
