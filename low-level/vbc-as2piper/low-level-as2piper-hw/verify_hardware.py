"""Run nominal/DR controls before authorizing a long PPO run; no W&B."""
import argparse,json
from pathlib import Path
import isaacgym
from isaacgym import gymapi
import torch,numpy as np
from legged_gym.envs.manip_loco.as2piper_hw import AS2PiPERHardware
from legged_gym.envs.manip_loco.as2piper_hw_config import AS2PiPERHardwareCfg

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['nominal','dr','explore'],default='nominal');a=p.parse_args()
    torch.manual_seed(1);np.random.seed(1);cfg=AS2PiPERHardwareCfg();cfg.env.num_envs=32
    if a.case=='nominal':
        for k in ['randomize_friction','randomize_base_mass','randomize_base_com','randomize_motor','randomize_gripper_mass','push_robots']:setattr(cfg.domain_rand,k,False)
        cfg.init_state.leg_reset_fraction=0;cfg.init_state.init_vel_perturb_range=0
    params=gymapi.SimParams();params.dt=cfg.sim.dt;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81);params.use_gpu_pipeline=True;params.physx.use_gpu=True
    for k in ['num_position_iterations','num_velocity_iterations','contact_offset','rest_offset','bounce_threshold_velocity','max_depenetration_velocity']:setattr(params.physx,k,getattr(cfg.sim.physx,k))
    e=AS2PiPERHardware(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True);e.reset();rows=[];traces=[]
    armature=e.gym.get_actor_dof_properties(e.envs[0],e.actor_handles[0])['armature'][:12]
    assert np.allclose(armature,cfg.control.leg_armature*4)
    original=e.check_termination
    def check():
        original();angles=e._get_body_orientation();h=e.root_states[:,2]-e.env_origins[:,2]
        up=(angles.abs().max(-1).values<.35)&(h>.22)
        rows.append(torch.cat((e.reset_buf[:,None].float(),up[:,None].float(),h[:,None],angles,
            e.contact_forces[:,e.feet_indices,2],e.torques[:,:12].abs().max(-1).values[:,None],
            (e.ee_pos-e.curr_ee_goal_cart_world).norm(dim=-1)[:,None]),dim=-1).cpu().numpy())
        traces.append(dict(root=(e.root_states[:4,:7]-torch.cat((e.env_origins[:4],torch.zeros(4,4,device=e.device)),dim=-1)).cpu().numpy(),q=e.dof_pos[:4].cpu().numpy()))
    e.check_termination=check
    # Verify mismatch shaping cannot produce a positive weighted reward.
    for name,fn in zip(e.reward_names,e.reward_functions):
        if name.startswith('tracking_contacts_shaped'):
            raw,_=fn();assert (raw*e.reward_scales[name]<=1e-6).all(),name
    for i in range(500):
        e.goal_timer.zero_();e.commands.zero_()
        action=torch.randn(e.num_envs,18,device=e.device)*.3 if a.case=='explore' else torch.zeros(e.num_envs,18,device=e.device)
        e.step(action)
    x=np.stack(rows);out=ROOT/'audit_hw';out.mkdir(exist_ok=True)
    np.savez_compressed(str(out/(a.case+'_control.npz')),samples=x,**{k:np.stack([v[k] for v in traces]) for k in traces[0]})
    result=dict(case=a.case,envs=32,seconds=10,resets=int(x[:,:,0].sum()),upright_fraction=float(x[:,:,1].mean()),
        resets_first_7s=int(x[:350,:,0].sum()),upright_first_7s=float(x[:350,:,1].mean()),leg_armature=armature.tolist(),
        height_mean_m=float(x[100:,:,2].mean()),roll_pitch_abs_p95_deg=float(np.rad2deg(np.quantile(abs(x[:,:,3:5]),.95))),
        mean_foot_normal_force_N=x[100:,:,5:9].mean((0,1)).tolist(),
        loaded_fraction_per_foot=(x[100:,:,5:9]>5).mean((0,1)).tolist(),leg_torque_abs_max_Nm=float(x[:,:,9].max()),
        ee_position_mean_m=float(x[100:,:,10].mean()))
    (out/(a.case+'_control.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    e.gym.destroy_sim(e.sim)


if __name__=='__main__':main()
