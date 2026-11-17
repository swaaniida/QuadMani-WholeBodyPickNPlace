"""Isolated reward ablations; the preserved hardware/VBC port is never edited."""
import torch
from .as2piper_vbc_port import AS2PiPERVBCPort, AS2PiPERVBCPortCfg, AS2PiPERVBCPortCfgPPO

PRESETS = {
    'original': (False, False, False, False, False),
    'velocity': (True, False, False, False, False),
    'jerk': (False, True, False, False, False),
    'stop': (False, False, True, False, False),
    'data_stop': (True, True, True, False, False),
    'speed': (True, True, True, True, False),
    'stable': (True, True, True, True, True),
}


class AS2PiPERGaitStudyCfg(AS2PiPERVBCPortCfg):
    class rewards(AS2PiPERVBCPortCfg.rewards):
        class scales(AS2PiPERVBCPortCfg.rewards.scales):
            pass
    class study:
        preset = 'stable'
        velocity_fix = True
        jerk_fix = True
        stop_fix = True
        speed_fix = True
        phase_fix = True
        velocity_sigma_m_s = .20
        yaw_sigma_rad_s = .35
        swing_clearance_m = .04
        clearance_sigma_m = .025
        foot_radius_m = .028


class AS2PiPERGaitStudyCfgPPO(AS2PiPERVBCPortCfgPPO):
    class runner(AS2PiPERVBCPortCfgPPO.runner):
        experiment_name = 'as2piper_gait_study'
        save_interval = 100


def configure(cfg, preset='stable', controlled=False):
    """Every paired ablation has the same environment, commands and PPO."""
    flags = PRESETS[preset]
    cfg.study.preset = preset
    for name, value in zip(('velocity_fix','jerk_fix','stop_fix','speed_fix','phase_fix'), flags):
        setattr(cfg.study, name, value)
    if flags[3]:
        cfg.rewards.scales.tracking_ang_vel = 1.0
    if flags[4]:
        cfg.rewards.scales.walking_dof = .5
        cfg.rewards.scales.feet_air_time = .5
    if controlled:
        cfg.terrain.mesh_type = 'plane'
        cfg.terrain.measure_heights = False
        cfg.init_state.init_vel_perturb_range = 0.
        for key in ('randomize_friction','randomize_base_mass','randomize_base_com',
                    'randomize_motor','randomize_gripper_mass','push_robots'):
            setattr(cfg.domain_rand, key, False)
    return cfg


class AS2PiPERGaitStudy(AS2PiPERVBCPort):
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.cfg.study.velocity_fix:
            # Tensor-index selection is a copy: gather AFTER every PhysX refresh.
            self.foot_velocities = self.rigid_body_state[:, self.feet_indices, 7:10]
            self.foot_positions = self.rigid_body_state[:, self.feet_indices, :3]

    def _step_contact_targets(self):
        super()._step_contact_targets()
        if self.cfg.study.stop_fix:
            self.desired_contact_states[~self._get_walking_cmd_mask()] = 1.

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        replacements = {}
        if self.cfg.study.jerk_fix:
            replacements['feet_jerk'] = self._contact_change
        if self.cfg.study.speed_fix:
            replacements.update(tracking_lin_vel_max=self._speed_tracking,
                                tracking_ang_vel=self._yaw_tracking)
        if self.cfg.study.phase_fix:
            replacements.update(feet_height=self._phase_clearance,
                                feet_air_time=self._phase_air_time,
                                base_height=self._terrain_relative_base_height)
        for i, name in enumerate(self.reward_names):
            if name in replacements:
                self.reward_functions[i] = replacements[name]

    def _contact_change(self):
        forces = self.force_sensor_tensor
        if not hasattr(self, '_previous_foot_wrench'):
            value = torch.zeros(self.num_envs, device=self.device)
        else:
            value = (forces-self._previous_foot_wrench).norm(dim=-1).sum(-1)
        self._previous_foot_wrench = forces.clone()
        value[self.episode_length_buf < 50] = 0.
        return value, value

    def _speed_tracking(self):
        error = self.base_lin_vel[:, :2]-self.commands[:, :2]
        value = torch.exp(-error.square().sum(-1)/self.cfg.study.velocity_sigma_m_s**2)
        return value, error.norm(dim=-1)

    def _yaw_tracking(self):
        error = self.base_ang_vel[:, 2]-self.commands[:, 2]
        return torch.exp(-error.square()/self.cfg.study.yaw_sigma_rad_s**2), error.abs()

    def _foot_ground_height(self):
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, 4, device=self.device)
        # Same conservative three-neighbour grid estimator used by legged_gym;
        # an approximation to the simplified trimesh, not a mesh ray cast.
        xy = self.rigid_body_state[:, self.feet_indices, :2]
        ij = ((xy+self.terrain.cfg.border_size)/self.terrain.cfg.horizontal_scale).long()
        ix = ij[..., 0].clamp(0, self.height_samples.shape[0]-2)
        iy = ij[..., 1].clamp(0, self.height_samples.shape[1]-2)
        heights = torch.minimum(self.height_samples[ix, iy], self.height_samples[ix+1, iy])
        heights = torch.minimum(heights, self.height_samples[ix, iy+1])
        return heights*self.terrain.cfg.vertical_scale

    def _phase_clearance(self):
        swing = torch.sin(2*torch.pi*(self.foot_indices-.5)).clamp(min=0.)
        desired = self.cfg.study.swing_clearance_m*swing
        clearance = (self.rigid_body_state[:, self.feet_indices, 2]
                     - self._foot_ground_height()-self.cfg.study.foot_radius_m)
        error = ((clearance-desired)/self.cfg.study.clearance_sigma_m).square().mean(-1)
        value = torch.exp(-error)*self._get_walking_cmd_mask()
        return value, (clearance-desired).abs().mean(-1)

    def _terrain_relative_base_height(self):
        # On a plane this is exactly the inherited world-z expression. On
        # terrain, changing ground elevation must not change the height error.
        height=self.root_states[:, 2]-self._foot_ground_height().mean(-1)
        return (height-self.cfg.rewards.base_height_target).abs(), height

    def _phase_air_time(self):
        contact = self.foot_contacts_from_sensor
        first = (self.feet_air_time > 0.) & contact
        self.feet_air_time += self.dt
        target = .5/self.cfg.env.frequencies
        value = ((self.feet_air_time.clamp(max=target+.10)-target)*first).sum(-1)
        value *= self._get_walking_cmd_mask()
        self.feet_air_time *= ~contact
        return value, value


class AS2PiPERStableGaitCfg(AS2PiPERGaitStudyCfg):
    """Public task defaults: selected reward, original rough terrain/full DR."""
    class rewards(AS2PiPERGaitStudyCfg.rewards):
        class scales(AS2PiPERGaitStudyCfg.rewards.scales):
            tracking_ang_vel=1.0
            walking_dof=.5
            feet_air_time=.5
