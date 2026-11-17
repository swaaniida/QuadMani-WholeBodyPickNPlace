"""Isolate free-base home holding: flat ground, nominal mass/COM, zero actions."""
import json
from pathlib import Path
import isaacgym
from isaacgym import gymapi,gymtorch
import torch
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.manip_loco.as2piper_forward import AS2PiPERForward
from legged_gym.envs.manip_loco.as2piper_config import AS2PiPERPlateCfg

ROOT=Path(LEGGED_GYM_ROOT_DIR);OUT=ROOT/'audit_forward/ee_nominal_stand_20261029';OUT.mkdir(exist_ok=True)


class FlatNominal(AS2PiPERForward):
    def _create_trimesh(self):self._create_ground_plane()
    def _get_env_origins(self):
        self.custom_origins=False;self.env_origins=torch.zeros(self.num_envs,3,device=self.device)
        self.env_origins[:,0]=torch.arange(self.num_envs,device=self.device)*3
    def _reset_dofs(self,ids):
        self.dof_pos[ids]=self.default_dof_pos;self.dof_vel[ids]=0
        self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(self.dof_state))


def main():
    torch.manual_seed(1);np.random.seed(1);cfg=AS2PiPERPlateCfg();cfg.env.num_envs=16
    cfg.env.stop_update_goal=False;cfg.env.pitch_control=False
    cfg.terrain.num_rows=2;cfg.terrain.num_cols=2;cfg.terrain.max_init_terrain_level=1
    cfg.terrain.measure_heights=False;cfg.terrain.curriculum=False
    for n in ['randomize_friction','randomize_base_mass','randomize_base_com','randomize_motor','randomize_gripper_mass','push_robots']:
        setattr(cfg.domain_rand,n,False)
    cfg.init_state.rand_yaw_range=0;cfg.init_state.origin_perturb_range=0;cfg.init_state.init_vel_perturb_range=0
    params=gymapi.SimParams();params.dt=cfg.sim.dt;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81)
    params.use_gpu_pipeline=True;params.physx.use_gpu=True
    for name in ['num_position_iterations','num_velocity_iterations','contact_offset','rest_offset','bounce_threshold_velocity','max_depenetration_velocity']:
        setattr(params.physx,name,getattr(cfg.sim.physx,name))
    e=FlatNominal(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True);e.reset()
    rows=[];traces=[];original=e.check_termination
    def check():
        original()
        angles=e._get_body_orientation();height=e.root_states[:,2]-e.env_origins[:,2]
        error=(e.ee_pos-e.curr_ee_goal_cart_world).norm(dim=-1)
        up=(angles.abs().max(-1).values<.35)&(height>.22)
        rows.append(torch.stack((error,up.float(),e.reset_buf.float(),height,angles[:,0],angles[:,1]),dim=-1).cpu().numpy())
        traces.append(dict(root=(e.root_states[:2,:7]-torch.cat((e.env_origins[:2],torch.zeros(2,4,device=e.device)),dim=-1)).cpu().numpy(),q=e.dof_pos[:2].cpu().numpy()))
    e.check_termination=check
    for i in range(500):
        e.goal_timer.zero_();e.commands.zero_()  # Hold the initial EE command.
        e.step(torch.zeros(e.num_envs,18,device=e.device))
        if i%100==0:print('NOMINAL',i,float(e.root_states[:,2].mean()),flush=True)
    a=np.stack(rows);np.savez_compressed(str(OUT/'tracking.npz'),samples=a,**{k:np.stack([f[k] for f in traces]) for k in traces[0]})
    result=dict(conditions='Free root, nominal mass/COM/motors, flat ground, exact neutral joint reset, zero policy actions, held home EE; not PPO and not the original domain-randomized task.',
        envs=16,seconds=10,reset_count=int(a[:,:,2].sum()),upright_fraction=float(a[:,:,1].mean()),
        late_position_mean_m=float(a[100:,:,0].mean()),late_position_p95_m=float(np.quantile(a[100:,:,0],.95)),
        base_height_mean_m=float(a[100:,:,3].mean()),roll_p95_deg=float(np.rad2deg(np.quantile(np.abs(a[:,:,4]),.95))))
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    e.gym.destroy_sim(e.sim)


if __name__=='__main__':main()
