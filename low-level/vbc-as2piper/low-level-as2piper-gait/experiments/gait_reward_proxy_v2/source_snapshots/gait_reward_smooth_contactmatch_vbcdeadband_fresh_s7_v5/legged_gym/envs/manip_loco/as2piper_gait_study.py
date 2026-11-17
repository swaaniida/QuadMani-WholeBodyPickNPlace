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
    'stable_grid': (True, True, True, True, True),
    'stable_contact': (True, True, True, True, True),
    'progress': (True, True, True, True, True),
    'progress_gain2': (True, True, True, True, True),
    'gaussian_gain2': (True, True, True, True, True),
    'original_proxy': (False, False, False, False, False),
    'stable_proxy': (True, True, True, True, True),
    'progress_gain2_proxy': (True, True, True, True, True),
    'stable_proxy_ground': (True, True, True, True, True),
    'progress_gain2_proxy_ground': (True, True, True, True, True),
    'progress_gain2_proxy_yaw4': (True, True, True, True, True),
    'progress_gain2_proxy_ground_floor': (True, True, True, True, True),
    'progress_gain2_proxy_smooth_yaw4': (True, True, True, True, True),
    'progress_gain2_proxy_smooth_ground_floor_yaw4': (True, True, True, True, True),
    'progress_gain2_proxy_smooth_yaw4_contactmatch': (True, True, True, True, True),
    'progress_gain2_proxy_smooth_yaw4_footground': (True, True, True, True, True),
    'progress_gain2_proxy_smooth_yaw4_footground_deadband': (True, True, True, True, True),
    'progress_gain2_proxy_smooth_yaw4_contactmatch_deadband': (True, True, True, True, True),
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
        terrain_height_mode = 'grid'
        ground_ema = .2
        progress_mix = 0.
        contact_force_proxy = False


class AS2PiPERGaitStudyCfgPPO(AS2PiPERVBCPortCfgPPO):
    class runner(AS2PiPERVBCPortCfgPPO.runner):
        experiment_name = 'as2piper_gait_study'
        save_interval = 100


def configure(cfg, preset='stable', controlled=False):
    """Every paired ablation has the same environment, commands and PPO."""
    flags = PRESETS[preset]
    cfg.study.preset = preset
    cfg.study.contact_force_proxy = preset.endswith('_proxy') or preset in ('stable_proxy_ground','progress_gain2_proxy_ground','progress_gain2_proxy_yaw4','progress_gain2_proxy_ground_floor','progress_gain2_proxy_smooth_yaw4','progress_gain2_proxy_smooth_ground_floor_yaw4','progress_gain2_proxy_smooth_yaw4_contactmatch','progress_gain2_proxy_smooth_yaw4_footground','progress_gain2_proxy_smooth_yaw4_footground_deadband','progress_gain2_proxy_smooth_yaw4_contactmatch_deadband')
    cfg.study.progress_mix = .5 if preset in ('progress','progress_gain2','progress_gain2_proxy','progress_gain2_proxy_ground','progress_gain2_proxy_yaw4','progress_gain2_proxy_ground_floor','progress_gain2_proxy_smooth_yaw4','progress_gain2_proxy_smooth_ground_floor_yaw4','progress_gain2_proxy_smooth_yaw4_contactmatch','progress_gain2_proxy_smooth_yaw4_footground','progress_gain2_proxy_smooth_yaw4_footground_deadband','progress_gain2_proxy_smooth_yaw4_contactmatch_deadband') else 0.
    cfg.study.terrain_height_mode = 'proxy' if preset in ('stable_proxy_ground','progress_gain2_proxy_ground','progress_gain2_proxy_ground_floor','progress_gain2_proxy_smooth_ground_floor_yaw4') else ('contact' if preset == 'stable_contact' else 'grid')
    for name, value in zip(('velocity_fix','jerk_fix','stop_fix','speed_fix','phase_fix'), flags):
        setattr(cfg.study, name, value)
    if flags[3]:
        cfg.rewards.scales.tracking_ang_vel = 1.0
    if flags[4]:
        cfg.rewards.scales.walking_dof = .5
        cfg.rewards.scales.feet_air_time = .5
    if preset in ('progress_gain2','gaussian_gain2','progress_gain2_proxy','progress_gain2_proxy_ground','progress_gain2_proxy_yaw4','progress_gain2_proxy_ground_floor','progress_gain2_proxy_smooth_yaw4','progress_gain2_proxy_smooth_ground_floor_yaw4','progress_gain2_proxy_smooth_yaw4_contactmatch','progress_gain2_proxy_smooth_yaw4_footground','progress_gain2_proxy_smooth_yaw4_footground_deadband','progress_gain2_proxy_smooth_yaw4_contactmatch_deadband'):
        cfg.rewards.scales.tracking_lin_vel_max = 4.0
        cfg.rewards.scales.tracking_ang_vel = 2.0
    if preset in ('progress_gain2_proxy_yaw4','progress_gain2_proxy_smooth_yaw4','progress_gain2_proxy_smooth_ground_floor_yaw4','progress_gain2_proxy_smooth_yaw4_contactmatch','progress_gain2_proxy_smooth_yaw4_footground','progress_gain2_proxy_smooth_yaw4_footground_deadband','progress_gain2_proxy_smooth_yaw4_contactmatch_deadband'):
        cfg.rewards.scales.tracking_ang_vel = 4.0
    if preset in ('progress_gain2_proxy_ground_floor','progress_gain2_proxy_smooth_ground_floor_yaw4'):
        cfg.study.penalize_excess_swing = False
    if preset in ('progress_gain2_proxy_smooth_yaw4','progress_gain2_proxy_smooth_ground_floor_yaw4','progress_gain2_proxy_smooth_yaw4_contactmatch','progress_gain2_proxy_smooth_yaw4_footground','progress_gain2_proxy_smooth_yaw4_footground_deadband','progress_gain2_proxy_smooth_yaw4_contactmatch_deadband'):
        cfg.study.smooth_progress_mix = True
    if preset in ('progress_gain2_proxy_smooth_yaw4_contactmatch','progress_gain2_proxy_smooth_yaw4_contactmatch_deadband'):
        cfg.study.symmetric_contact = True
    if preset in ('progress_gain2_proxy_smooth_yaw4_footground','progress_gain2_proxy_smooth_yaw4_footground_deadband'):
        cfg.study.foot_height_reference = 'proxy'
    if preset in ('progress_gain2_proxy_smooth_yaw4_footground_deadband','progress_gain2_proxy_smooth_yaw4_contactmatch_deadband'):
        cfg.study.vbc_linear_deadband = True
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
        if (self.cfg.study.phase_fix and self.cfg.terrain.mesh_type != 'plane'
                and (self.cfg.study.terrain_height_mode in ('contact','proxy')
                     or getattr(self.cfg.study,'foot_height_reference',None)=='proxy')):
            self._update_support_ground()

    def _reset_root_states(self, ids):
        super()._reset_root_states(ids)
        if hasattr(self, '_support_ground'):
            self._support_ground[ids] = self.env_origins[ids, 2:3]
            self._body_ground[ids] = self.env_origins[ids, 2]

    def _update_support_ground(self):
        if not hasattr(self, '_support_ground'):
            self._support_ground = self.env_origins[:, 2:3].repeat(1, 4)
            self._body_ground = self.env_origins[:, 2].clone()
        use_proxy=(self.cfg.study.terrain_height_mode == 'proxy'
                   or getattr(self.cfg.study,'foot_height_reference',None)=='proxy')
        loaded = self.foot_contacts_from_sensor if use_proxy else (self.contact_forces[:, self.feet_indices, 2] > 5.)
        measured = self.rigid_body_state[:, self.feet_indices, 2]-self.cfg.study.foot_radius_m
        self._support_ground = torch.where(loaded, measured, self._support_ground)
        self._body_ground += self.cfg.study.ground_ema*(self._support_ground.mean(-1)-self._body_ground)

    def _step_contact_targets(self):
        super()._step_contact_targets()
        if self.cfg.study.stop_fix:
            self.desired_contact_states[~self._get_walking_cmd_mask()] = 1.

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        replacements = {}
        if self.cfg.study.contact_force_proxy:
            replacements['tracking_contacts_shaped_force']=self._proxy_contact_force
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

    def _proxy_contact_force(self):
        # Existing VBC presence bit, not a calibrated normal force.
        if not self.cfg.env.observe_gait_commands:return 0.,0.
        presence=self.foot_contacts_from_sensor.float()
        cost=(1-self.desired_contact_states)*presence
        if getattr(self.cfg.study,'symmetric_contact',False):
            # Missing stance contact must not make the force cost disappear.
            cost=cost+self.desired_contact_states*(1-presence)
        cost=cost.mean(-1)
        return cost,-cost

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
        command=self.commands[:,:2]
        if getattr(self.cfg.study,'vbc_linear_deadband',False):
            # Preserve VBC's in-place linear target below its existing clip.
            # Reward-only copy: command observations/sampler remain unchanged.
            command=command.clone()
            command[:,0]=torch.where(command[:,0].abs()<self.cfg.commands.lin_vel_x_clip,
                                     torch.zeros_like(command[:,0]),command[:,0])
        error = self.base_lin_vel[:, :2]-command
        value = torch.exp(-error.square().sum(-1)/self.cfg.study.velocity_sigma_m_s**2)
        if self.cfg.study.progress_mix:
            norm2=command.square().sum(-1)
            mix=self.cfg.study.progress_mix
            if getattr(self.cfg.study,'smooth_progress_mix',False):
                # Bounded directional credit; continuous at zero command.
                # Large commands keep the original exploration incentive.
                mix=mix*(norm2.sqrt()/self.cfg.study.velocity_sigma_m_s).clamp(max=1.)
                progress=((self.base_lin_vel[:,:2]*command).sum(-1)/norm2.clamp(min=1e-12)).clamp(-1.,1.)
            else:
                progress=((self.base_lin_vel[:,:2]*command).sum(-1)/norm2.clamp(min=1e-6)).clamp(-1.,1.)
                progress=torch.where(norm2>1e-6,progress,value)
            value=(1-mix)*value+mix*progress
        return value, error.norm(dim=-1)

    def _yaw_tracking(self):
        error = self.base_ang_vel[:, 2]-self.commands[:, 2]
        value=torch.exp(-error.square()/self.cfg.study.yaw_sigma_rad_s**2)
        if self.cfg.study.progress_mix:
            command=self.commands[:,2];mix=self.cfg.study.progress_mix
            if getattr(self.cfg.study,'smooth_progress_mix',False):
                mix=mix*(command.abs()/self.cfg.study.yaw_sigma_rad_s).clamp(max=1.)
                progress=(self.base_ang_vel[:,2]*command/command.square().clamp(min=1e-12)).clamp(-1.,1.)
            else:
                progress=(self.base_ang_vel[:,2]*command/command.square().clamp(min=1e-6)).clamp(-1.,1.)
                progress=torch.where(command.abs()>1e-3,progress,value)
            value=(1-mix)*value+mix*progress
        return value,error.abs()

    def _foot_ground_height(self,mode=None):
        mode=self.cfg.study.terrain_height_mode if mode is None else mode
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, 4, device=self.device)
        if mode in ('contact','proxy'):
            if not hasattr(self, '_support_ground'):self._update_support_ground()
            return self._support_ground
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
                     - self._foot_ground_height(mode=getattr(self.cfg.study,'foot_height_reference',None))
                     - self.cfg.study.foot_radius_m)
        delta = clearance-desired
        if self.cfg.terrain.mesh_type != 'plane' and self.cfg.study.terrain_height_mode == 'contact':
            # On unobserved uneven ground, a higher viable step is allowed.
            # Stance motion/contact penalties still use actual current physics.
            delta = delta.clamp(max=0.)
        if (self.cfg.terrain.mesh_type != 'plane'
                and not getattr(self.cfg.study, 'penalize_excess_swing', True)):
            # Permit higher swing over rough ground, retain a zero-height
            # stance target. This does not relax physical joint/collision limits.
            delta = torch.where(swing > 0., delta.clamp(max=0.), delta)
        error = (delta/self.cfg.study.clearance_sigma_m).square().mean(-1)
        value = torch.exp(-error)*self._get_walking_cmd_mask()
        return value, (clearance-desired).abs().mean(-1)

    def _terrain_relative_base_height(self):
        # On a plane this is exactly the inherited world-z expression. On
        # terrain, changing ground elevation must not change the height error.
        ground=self._foot_ground_height().mean(-1)
        if self.cfg.terrain.mesh_type != 'plane' and self.cfg.study.terrain_height_mode in ('contact','proxy'):
            ground=self._body_ground
        height=self.root_states[:, 2]-ground
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
    """Nominal-plane validated reward; full rough-terrain/DR selection still pending."""
    class rewards(AS2PiPERGaitStudyCfg.rewards):
        class scales(AS2PiPERGaitStudyCfg.rewards.scales):
            tracking_ang_vel=1.0
            walking_dof=.5
            feet_air_time=.5
