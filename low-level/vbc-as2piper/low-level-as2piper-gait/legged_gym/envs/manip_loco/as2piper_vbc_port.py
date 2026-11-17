"""VBC task semantics with AS2/PiPER geometry and dimensional hardware values."""
import json
from pathlib import Path
import torch
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, euler_from_quat
from .manip_loco import ManipLoco
from .as2piper_config import AS2PiPERPlateCfg, AS2PiPERPlateCfgPPO

ROOT = Path(__file__).resolve().parents[3]
DATA = json.loads((ROOT/'resources/commands/hardware.json').read_text())
HW = DATA['derived']
MASS = DATA['ratios']['mass']
LENGTH = DATA['ratios']['length']
NOISE_RATIO = [80.*MASS*LENGTH/k for k in HW['leg_kp']]


class AS2PiPERVBCPortCfg(AS2PiPERPlateCfg):
    class env(AS2PiPERPlateCfg.env):
        stop_update_goal = False
        pitch_control = False
        # Keep the original 2Hz clock; no extra gait curriculum/task override.
    class control(AS2PiPERPlateCfg.control):
        stiffness = {**{j+'_joint':v for j,v in zip(('hip','thigh','calf'), HW['leg_kp'])}, 'piper_joint':5.}
        damping = {**{j+'_joint':v for j,v in zip(('hip','thigh','calf'), HW['leg_kd'])}, 'piper_joint':.5}
        leg_armature = HW['leg_armature']
        # Joint angles are not scaled by mass; original action scales restored.
    class asset(AS2PiPERPlateCfg.asset):
        hardware_profile = 'resources/commands/hardware_vbc_port.json'
    class goal_ee(AS2PiPERPlateCfg.goal_ee):
        sampling_mode = 'original_vbc_sphere'
        reference_frame = 'vbc_yaw_center_world_z_invariant'
        validation_mode = 'original_vbc_ee_box_floor'
        underground_limit = -(.3244069024246581+.08202847821788371)
        class sphere_center:
            x_offset = .005
            y_offset = 0.
            z_invariant_offset = .3244069024246581+.08202847821788371
        class ranges(AS2PiPERPlateCfg.goal_ee.ranges):
            # Expanded ranges; independently reviewed with actual collision shapes.
            pos_l = [.40,.50]
            pos_p = [.10,.55]
            pos_y = [-.30,.30]
            delta_orn_r = [-.10,.10]
            delta_orn_p = [-.10,.10]
            delta_orn_y = [-.10,.10]
            init_pos_end = [.46,.25,0.]
    class commands(AS2PiPERPlateCfg.commands):
        lin_vel_x_clip = .2 * LENGTH**.5
        class ranges(AS2PiPERPlateCfg.commands.ranges):
            lin_vel_x = [-.8 * LENGTH**.5, .8 * LENGTH**.5]
    class domain_rand(AS2PiPERPlateCfg.domain_rand):
        added_mass_range = [0.,15.*MASS]
        added_com_range_x = [-.15*LENGTH,.15*LENGTH]
        added_com_range_y = [-.15*LENGTH,.15*LENGTH]
        added_com_range_z = [-.15*LENGTH,.15*LENGTH]
        max_push_vel_xy = .5*LENGTH**.5
    class rewards(AS2PiPERPlateCfg.rewards):
        max_contact_force = 40.*MASS
        feet_height_target = .3*LENGTH
        gait_force_sigma = .5*MASS**2
        gait_vel_sigma = .5*LENGTH


class AS2PiPERVBCPortCfgPPO(AS2PiPERPlateCfgPPO):
    class policy(AS2PiPERPlateCfgPPO.policy):
        # Match the B1 initial PD-torque noise after dimensional torque scaling.
        # Unlike the old action-scale cap, this does not restrict policy means.
        init_std = [[s*r for s,r in zip([.8,1.,1.],NOISE_RATIO)]*4+[1.]*6]
    class algorithm(AS2PiPERPlateCfgPPO.algorithm):
        min_policy_std = [[s*r for s,r in zip([.15,.25,.25],NOISE_RATIO)]*4+[.2]*3+[.05]*3]
    class runner(AS2PiPERPlateCfgPPO.runner):
        experiment_name = 'as2piper_vbc_hardware_port'


class AS2PiPERVBCPort(ManipLoco):
    # No AS2PiPERHardware/Gait/WalkWBC inheritance: original command, reset,
    # observation, reward functions and contact targets are the active defaults.
    def step(self, actions):
        return super().step(actions.clone())

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        props['armature'][:12] = self.cfg.control.leg_armature*4
        return props

    def _create_trimesh(self):
        if self.cfg.terrain.mesh_type == 'plane':
            return self._create_ground_plane()
        return super()._create_trimesh()

    def _get_env_origins(self):
        if self.cfg.terrain.mesh_type != 'plane':
            return super()._get_env_origins()
        self.custom_origins=False
        idx=torch.arange(self.num_envs,device=self.device);n=int(self.num_envs**.5)+1
        self.env_origins=torch.zeros(self.num_envs,3,device=self.device)
        self.env_origins[:,0]=torch.div(idx,n,rounding_mode='floor')*3
        self.env_origins[:,1]=(idx%n)*3

    def _reset_root_states(self, ids):
        super()._reset_root_states(ids)
        # Base orientation cache otherwise still refers to the previous episode.
        r,p,y=euler_from_quat(self.root_states[ids,3:7])
        self.base_yaw_quat[ids]=quat_from_euler_xyz(r*0,p*0,y)

    def _resample_ee_goal(self, ids, is_init=False):
        super()._resample_ee_goal(ids,is_init)
        if not is_init or not len(ids):
            return
        # Publish the new episode's goal immediately; do not execute one step
        # toward a stale goal from the previous episode.
        self.curr_ee_goal_sphere[ids]=self.init_start_ee_sphere
        from isaacgym.torch_utils import sphere2cart
        cart=sphere2cart(self.curr_ee_goal_sphere[ids])
        self.curr_ee_goal_cart[ids]=cart
        self.curr_ee_goal_cart_world[ids]=self._get_ee_goal_spherical_center()[ids]+quat_apply(self.base_yaw_quat[ids],cart)
        yaw=euler_from_quat(self.root_states[ids,3:7])[2];zero=yaw*0
        self.ee_goal_orn_quat[ids]=quat_from_euler_xyz(zero+self.cfg.goal_ee.orientation_roll_offset,
            -self.curr_ee_goal_sphere[ids,1]+self.cfg.goal_ee.arm_induced_pitch,
            yaw+self.curr_ee_goal_sphere[ids,2])

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        # Retain one explicitly demonstrated sign correction. All reward weights
        # and other reward functions are restored to the original VBC settings.
        for i,name in enumerate(self.reward_names):
            if name in ('tracking_contacts_shaped_force','tracking_contacts_shaped_vel'):
                original=self.reward_functions[i]
                def penalty(original=original):
                    raw,metric=original()
                    return -raw,metric
                self.reward_functions[i]=penalty
