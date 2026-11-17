"""Locomotion curriculum: learn a measured gait before simultaneous arm goals."""
import torch
from isaacgym.torch_utils import torch_rand_float
from .as2piper_hw import AS2PiPERHardware
from .as2piper_hw_config import AS2PiPERHardwareCfg,AS2PiPERHardwareCfgPPO


class AS2PiPERGaitCfg(AS2PiPERHardwareCfg):
    class domain_rand(AS2PiPERHardwareCfg.domain_rand):
        randomize_base_mass=False
        randomize_base_com=False
        randomize_gripper_mass=False
        push_robots=False
    class goal_ee(AS2PiPERHardwareCfg.goal_ee):
        class ranges(AS2PiPERHardwareCfg.goal_ee.ranges):
            init_pos_end=AS2PiPERHardwareCfg.goal_ee.ranges.init_pos_start
    class rewards(AS2PiPERHardwareCfg.rewards):
        tracking_sigma=.04
        swing_clearance=.04
        class scales(AS2PiPERHardwareCfg.rewards.scales):
            walking_dof=0.
            tracking_lin_vel_max=0.
            tracking_lin_vel=3.
            tracking_ang_vel=1.
            feet_air_time=.5
    class commands(AS2PiPERHardwareCfg.commands):
        class ranges:
            lin_vel_x=[-.4,.4]
            ang_vel_yaw=[-.5,.5]


class AS2PiPERGaitCfgPPO(AS2PiPERHardwareCfgPPO):
    class policy(AS2PiPERHardwareCfgPPO.policy):
        init_std=[[.6,.6,.6]*4+[1.]*6]
    class runner(AS2PiPERHardwareCfgPPO.runner):
        experiment_name='as2piper_gait_curriculum'
        save_interval=100


class AS2PiPERGait(AS2PiPERHardware):
    def _resample_ee_goal(self,ids,is_init=False):
        # Arm pose and finger geometry stay physically present, with normal IK.
        # No costly random manipulation goals while learning to place the feet.
        return super()._resample_ee_goal(ids,is_init=True)

    def _resample_commands(self,ids):
        if self.cfg.env.teleop_mode or not len(ids):return
        self.commands[ids,0]=torch_rand_float(.15,.4,(len(ids),1),device=self.device).flatten()
        self.commands[ids,1]=0
        self.commands[ids,2]=torch_rand_float(-.5,.5,(len(ids),1),device=self.device).flatten()
        kind=torch.rand(len(ids),device=self.device)
        self.commands[ids[kind<.15]]=0
        self.commands[ids[(kind>=.15)&(kind<.35)],0]=0
        self.commands[ids[kind>.65],2]=0

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        replacements={'feet_air_time':self._air_time,'feet_height':self._phase_height,
                      'tracking_ang_vel':self._yaw_tracking}
        for i,name in enumerate(self.reward_names):
            if name in replacements:self.reward_functions[i]=replacements[name]

    def _air_time(self):
        contact=self.foot_contacts_from_sensor
        first=(self.feet_air_time>0)&contact
        self.feet_air_time+=self.dt
        target=.5/self.cfg.env.frequencies
        value=((self.feet_air_time-target)*first).sum(-1)*self._get_walking_cmd_mask()
        self.feet_air_time*=~contact
        return value,value

    def _phase_height(self):
        phase=self.foot_indices
        swing=torch.clamp(torch.sin(2*torch.pi*(phase-.5)),min=0)
        desired=self.cfg.rewards.swing_clearance*swing
        actual=self.rigid_body_state[:,self.feet_indices,2]-self.env_origins[:,2:3]-.028
        error=((actual-desired)/.025).square().mean(-1)
        value=torch.exp(-error)
        value*=self._get_walking_cmd_mask()
        return value,error

    def _yaw_tracking(self):
        error=(self.commands[:,2]-self.base_ang_vel[:,2]).square()
        return torch.exp(-error/.0625),torch.sqrt(error)
