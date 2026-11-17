"""AS2-specific, arm-base-relative and reachability-screened EE commands."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import cart2sphere,euler_from_quat,quat_apply,quat_rotate_inverse
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.urdf_chain import URDFChain,multiply,inverse
from .manip_loco import ManipLoco


class AS2PiPERForward(ManipLoco):
    def _init_buffers(self):
        super()._init_buffers()
        root=Path(LEGGED_GYM_ROOT_DIR)
        asset=Path(self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(root)))
        directory=root/'resources/commands'
        meta=json.loads((directory/'arm_commands.json').read_text())
        if hashlib.sha256(asset.read_bytes()).hexdigest()!=meta['asset_sha256']:
            raise RuntimeError('Command bank does not match URDF: rerun build_command_bank.py')
        with np.load(str(directory/'arm_commands.npz')) as bank:
            self.arm_command_bank=torch.tensor(bank['joints'],device=self.device,dtype=torch.float)
            self.arm_command_home=torch.tensor(bank['home'],device=self.device,dtype=torch.float)
        self.arm_fk=URDFChain(asset,'piper_base_link','piper_umi_nominal_tcp',meta['joint_names'],self.device)
        self.home_local_quat=self.arm_fk(self.arm_command_home[None,:])[1]
        self.leg_fk=[]
        for leg in ('FL','FR','RL','RR'):
            names=[leg+'_'+s+'_joint' for s in ('hip','thigh','calf')]
            self.leg_fk.append(URDFChain(asset,'base_link',leg+'_foot',names,self.device))
        self.command_start_q=self.arm_command_home.repeat(self.num_envs,1)
        self.command_end_q=self.command_start_q.clone()
        self.command_return_home=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.command_local_quat=torch.zeros(self.num_envs,4,device=self.device);self.command_local_quat[:,3]=1
        self.world_goal_latched=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._write_arm_goals(torch.arange(self.num_envs,device=self.device),self.command_start_q)

    def step(self,actions):
        # The inherited environment masks arm actions in-place. Protect PPO's
        # original sampled action/log-prob pair without changing the algorithm.
        return super().step(actions.clone())

    def _get_ee_goal_spherical_center(self):
        return self.root_states[:,:3]+quat_apply(self.root_states[:,3:7],self.arm_base_offset)

    def _get_ee_goal_observation(self):
        qbody=self.root_states[:,3:7]
        local=quat_rotate_inverse(qbody,self.curr_ee_goal_cart_world-self._get_ee_goal_spherical_center())
        rotation=multiply(inverse(qbody),self.ee_goal_orn_quat)
        # Spatial rotation vector relative to home, expressed in arm-base axes.
        # Unlike absolute Euler angles this stays continuous near home's pi roll.
        delta=multiply(rotation,inverse(self.home_local_quat).expand(self.num_envs,-1))
        delta=torch.where(delta[:,3:]<0,-delta,delta)
        n=delta[:,:3].norm(dim=-1,keepdim=True)
        angle=2*torch.atan2(n,delta[:,3:])
        rotvec=delta[:,:3]*torch.where(n>1e-6,angle/n.clamp_min(1e-6),torch.full_like(n,2.))
        return local,rotvec

    def _reset_dofs(self,env_ids):
        super()._reset_dofs(env_ids)
        # Original leg reset distribution stays intact. Arm starts exactly at
        # the screened home, so its position AND orientation match the goal.
        self.dof_pos[env_ids,12:18]=self.default_dof_pos[12:18]
        self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(self.dof_state))

    def _reset_root_states(self,env_ids):
        super()._reset_root_states(env_ids)
        # Place the lowest real foot sphere 2 mm above terrain origin height,
        # using the randomized leg configuration rather than B1's z=0.5 m.
        bottom=[]
        for i,fk in enumerate(self.leg_fk):
            p,_=fk(self.dof_pos[env_ids,i*3:i*3+3]);bottom.append(p[:,2]-.028)
        low=torch.stack(bottom,dim=-1).min(dim=-1).values
        self.root_states[env_ids,2]=self.env_origins[env_ids,2]-low+.002
        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self._root_states))
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def _write_arm_goals(self,env_ids,q):
        p,rotation=self.arm_fk(q)
        body=self.root_states[env_ids,3:7]
        self.curr_ee_goal_cart[env_ids]=p
        self.curr_ee_goal_sphere[env_ids]=cart2sphere(p)
        self.command_local_quat[env_ids]=rotation
        self.curr_ee_goal_cart_world[env_ids]=self._get_ee_goal_spherical_center()[env_ids]+quat_apply(body,p)
        self.ee_goal_orn_quat[env_ids]=multiply(body,rotation)
        self.ee_goal_orn_euler[env_ids]=torch.stack(euler_from_quat(self.ee_goal_orn_quat[env_ids]),dim=-1)

    def _resample_ee_goal(self,env_ids,is_init=False):
        if not len(env_ids):return
        self.world_goal_latched[env_ids]=False
        if is_init:
            q=self.dof_pos[env_ids,12:18].clone()
            self.command_start_q[env_ids]=q
            self.command_end_q[env_ids]=self.arm_command_home
            self.command_return_home[env_ids]=False
        else:
            returning=self.command_return_home[env_ids]
            self.command_start_q[env_ids]=self.command_end_q[env_ids].clone()
            index=torch.randint(len(self.arm_command_bank),(len(env_ids),),device=self.device)
            chosen=self.arm_command_bank[index]
            self.command_end_q[env_ids]=torch.where(returning[:,None],self.arm_command_home[None,:],chosen)
            self.command_return_home[env_ids]=~returning
        self.goal_timer[env_ids]=0
        n=len(env_ids)
        self.traj_timesteps[env_ids]=torch.empty(n,device=self.device).uniform_(*self.cfg.goal_ee.traj_time)/self.dt
        self.traj_total_timesteps[env_ids]=self.traj_timesteps[env_ids]+torch.empty(n,device=self.device).uniform_(*self.cfg.goal_ee.hold_time)/self.dt
        self.ee_start_sphere[env_ids]=cart2sphere(self.arm_fk(self.command_start_q[env_ids])[0])
        self.ee_goal_cart[env_ids]=self.arm_fk(self.command_end_q[env_ids])[0]
        self.ee_goal_sphere[env_ids]=cart2sphere(self.ee_goal_cart[env_ids])
        if is_init:self._write_arm_goals(env_ids,self.command_start_q[env_ids])

    def _update_curr_ee_goal(self):
        env_ids=(~self.world_goal_latched).nonzero(as_tuple=False).flatten()
        t=(self.goal_timer[env_ids]/self.traj_timesteps[env_ids]).clamp(0,1)
        smooth=t*t*(3-2*t)
        q=torch.lerp(self.command_start_q[env_ids],self.command_end_q[env_ids],smooth[:,None])
        self._write_arm_goals(env_ids,q)
        self.goal_timer[env_ids]+=1
        finished=env_ids[self.goal_timer[env_ids]>self.traj_total_timesteps[env_ids]]
        if self.stop_update_goal:self.commands[finished,0]=0;self.commands[finished,2]=0
        self._resample_ee_goal(finished)

    def set_world_ee_goal(self,env_ids,position,quaternion_xyzw):
        """Latch an external world target for future whole-body manipulation.

        This bypasses the locomotion command bank; the caller must check reach,
        collision and terrain. Robot tilt must not move a world-fixed object.
        """
        self.curr_ee_goal_cart_world[env_ids]=position
        self.ee_goal_orn_quat[env_ids]=quaternion_xyzw/quaternion_xyzw.norm(dim=-1,keepdim=True)
        self.ee_goal_orn_euler[env_ids]=torch.stack(euler_from_quat(self.ee_goal_orn_quat[env_ids]),dim=-1)
        self.world_goal_latched[env_ids]=True

    def _draw_ee_goal_traj(self):
        # Inherited drawing interpolates spherical coordinates; our trajectory
        # is joint-space FK. Keep current-goal axes from _draw_ee_goal_curr only.
        return

    def _draw_collision_bbox(self):
        # B1's yaw/world-height keep-out box does not describe this sampler.
        return
