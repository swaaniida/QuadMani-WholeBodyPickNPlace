"""Hardware-calibrated task with pose-aware retries and an explicit cold start."""
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float,sphere2cart
from .as2piper_forward import AS2PiPERForward
from legged_gym.utils.arm_collision import ArmCollisionGuard


class RefinedGuard(ArmCollisionGuard):
    def solve(self,target,orientation,seed,iterations=12):
        q,failed=super().solve(target,orientation,seed,iterations)
        if failed.any():
            ids=failed.nonzero(as_tuple=False).flatten()
            q1,f1=super().solve(target[ids],orientation[ids],q[ids],52)
            q[ids]=q1;failed[ids]=f1
        return q,failed


class AS2PiPERHardware(AS2PiPERForward):
    def _process_dof_props(self,props,env_id):
        props=super()._process_dof_props(props,env_id)
        props['armature'][:12]=self.cfg.control.leg_armature*4
        return props

    def _create_trimesh(self):
        if self.cfg.terrain.mesh_type=='plane':return self._create_ground_plane()
        return super()._create_trimesh()

    def _init_buffers(self):
        super()._init_buffers()
        self.collision_guard.__class__=RefinedGuard
        self.pose_requests=0;self.pose_accepted=0

    def _get_env_origins(self):
        if self.cfg.terrain.mesh_type!='plane':return super()._get_env_origins()
        self.custom_origins=False
        n=int(self.num_envs**.5)+1;idx=torch.arange(self.num_envs,device=self.device)
        self.env_origins=torch.zeros(self.num_envs,3,device=self.device)
        self.env_origins[:,0]=torch.div(idx,n,rounding_mode='floor')*3
        self.env_origins[:,1]=(idx%n)*3

    def _reset_dofs(self,ids):
        self.dof_pos[ids]=self.default_dof_pos
        noise=self.cfg.init_state.leg_reset_fraction
        self.dof_pos[ids,:12]*=torch_rand_float(1-noise,1+noise,(len(ids),12),device=self.device)
        self.dof_pos[ids]=torch.maximum(torch.minimum(self.dof_pos[ids],self.dof_pos_limits[:,1]),self.dof_pos_limits[:,0])
        self.dof_vel[ids]=0
        self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(self.dof_state))

    def _resample_ee_goal(self,env_ids,is_init=False):
        if is_init or self.cfg.env.teleop_mode:
            return super()._resample_ee_goal(env_ids,is_init)
        if not len(env_ids):return
        previous=self.ee_goal_sphere[env_ids].clone();orn=self.ee_goal_orn_delta_rpy[env_ids].clone()
        self.ee_start_sphere[env_ids]=previous
        pending=env_ids.clone();self.pose_requests+=len(env_ids)
        for _ in range(10):
            # Same independent uniform proposals; retry the complete pose.
            self._resample_ee_goal_orn_once(pending)
            self._resample_ee_goal_sphere_once(pending)
            pending=pending[self._collision_check(pending)]
            if not len(pending):break
        self.pose_accepted+=len(env_ids)-len(pending)
        if len(pending):
            # env_ids are normally sorted; support arbitrary caller order too.
            sorted_ids,order=env_ids.sort();where=torch.searchsorted(sorted_ids,pending)
            self.ee_goal_sphere[pending]=previous[order[where]]
            self.ee_goal_orn_delta_rpy[pending]=orn[order[where]]
            self.pose_fallback_count+=len(pending)
        self.ee_goal_cart[env_ids]=sphere2cart(self.ee_goal_sphere[env_ids]);self.goal_timer[env_ids]=0

    def _resample_commands(self,ids):
        if self.cfg.env.teleop_mode or not len(ids):return
        # Start with attainable slow commands; restore the full scaled envelope.
        progress=min(self.global_steps/(24*self.cfg.env.command_warmup_iterations),1.)
        factor=.25+.75*progress
        vmax=self.command_ranges['lin_vel_x'][1]*factor
        wmax=self.command_ranges['ang_vel_yaw'][1]*factor
        low=0. if self.global_steps<5000*24 else -vmax
        self.commands[ids,0]=torch_rand_float(low,vmax,(len(ids),1),device=self.device).flatten()
        self.commands[ids,1]=0
        self.commands[ids,2]=torch_rand_float(-wmax,wmax,(len(ids),1),device=self.device).flatten()
        # Explicitly include stand commands rather than relying only on clipping.
        stop=torch.rand(len(ids),device=self.device)<.2
        self.commands[ids[stop]]=0
        keep=(self.commands[ids,0].abs()>.05)|(self.commands[ids,2].abs()>.1)
        self.commands[ids]*=keep[:,None]

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        # The inherited raw gait terms are <=0 with negative scales, creating
        # positive reward for mismatch. Correct only this double-negative bug;
        # retain every configured reward weight and the other formulas.
        for index,name in enumerate(self.reward_names):
            if name in ['tracking_contacts_shaped_force','tracking_contacts_shaped_vel']:
                original=self.reward_functions[index]
                def penalty(original=original):
                    raw,metric=original()
                    return -raw,metric
                self.reward_functions[index]=penalty
