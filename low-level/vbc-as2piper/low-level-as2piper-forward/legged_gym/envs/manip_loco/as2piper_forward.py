"""B1 spherical pose sampling with AS2 dimensions and arm/body rejection."""
import hashlib,json
from pathlib import Path
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import sphere2cart,quat_apply,quat_rotate_inverse,quat_from_euler_xyz,euler_from_quat
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.urdf_chain import URDFChain,rotate
from legged_gym.utils.arm_collision import ArmCollisionGuard
from .manip_loco import ManipLoco


class AS2PiPERForward(ManipLoco):
    def _init_buffers(self):
        super()._init_buffers()
        root=Path(LEGGED_GYM_ROOT_DIR);asset=Path(self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(root)))
        geometry=root/'resources/commands/arm_collision.json'
        if json.loads(geometry.read_text())['asset_sha256']!=hashlib.sha256(asset.read_bytes()).hexdigest():
            raise RuntimeError('Rebuild collision geometry for this URDF with build_collision_guard.py')
        names=['piper_joint'+str(i) for i in range(1,7)]
        self.arm_fk=URDFChain(asset,'piper_base_link','piper_umi_nominal_tcp',names,self.device)
        self.collision_guard=ArmCollisionGuard(asset,geometry,self.device)
        self.leg_fk=[URDFChain(asset,'base_link',leg+'_foot',[leg+'_'+s+'_joint' for s in ('hip','thigh','calf')],self.device)
                     for leg in ('FL','FR','RL','RR')]
        self._last_rejected_ids=torch.empty(0,device=self.device,dtype=torch.long)
        self.pose_fallback_count=0

    def step(self,actions):
        return super().step(actions.clone())  # Protect PPO's stored action.

    def _get_ee_goal_spherical_center(self):
        # Actual arm origin. Sampling axes remain yaw/gravity aligned, as B1.
        return self.root_states[:,:3]+quat_apply(self.root_states[:,3:7],self.arm_base_offset)

    def _get_ee_goal_observation(self):
        local=quat_rotate_inverse(self.root_states[:,3:7],self.curr_ee_goal_cart_world-self._get_ee_goal_spherical_center())
        return local,0*self.curr_ee_goal_sphere  # Original B1 observation contract.

    def _reset_dofs(self,env_ids):
        super()._reset_dofs(env_ids)
        self.dof_pos[env_ids,12:18]=self.default_dof_pos[12:18]
        self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(self.dof_state))

    def _reset_root_states(self,env_ids):
        super()._reset_root_states(env_ids)
        bottom=[fk(self.dof_pos[env_ids,i*3:i*3+3])[0][:,2]-.028 for i,fk in enumerate(self.leg_fk)]
        low=torch.stack(bottom,dim=-1).min(dim=-1).values
        self.root_states[env_ids,2]=self.env_origins[env_ids,2]-low+.002
        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self._root_states))
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def _resample_ee_goal(self,env_ids,is_init=False):
        # B1 still draws radius/pitch/yaw independently, draws orientation
        # deltas independently, and retries the position at most ten times.
        if not len(env_ids):return
        previous=self.ee_goal_sphere.clone();previous_orn=self.ee_goal_orn_delta_rpy.clone()
        self._last_rejected_ids=torch.empty(0,device=self.device,dtype=torch.long)
        super()._resample_ee_goal(env_ids,is_init=is_init)
        failed=self._last_rejected_ids
        if not is_init and len(failed):
            # Original B1 leaves the final rejected pose after ten failures.
            # Holding the previous pose is the collision-specific exception.
            self.ee_goal_sphere[failed]=previous[failed]
            self.ee_goal_orn_delta_rpy[failed]=previous_orn[failed]
            self.ee_goal_cart[failed]=sphere2cart(self.ee_goal_sphere[failed])
            self.pose_fallback_count+=len(failed)
        if is_init:
            # Initialize immediately; do not use the previous episode's goal
            # for the first IK action after reset.
            self.curr_ee_goal_sphere[env_ids]=self.init_start_ee_sphere
            local=sphere2cart(self.curr_ee_goal_sphere[env_ids]);self.curr_ee_goal_cart[env_ids]=local
            yaw=euler_from_quat(self.root_states[env_ids,3:7])[2]
            zero=torch.zeros_like(yaw);qy=quat_from_euler_xyz(zero,zero,yaw)
            v=quat_apply(qy,local)
            self.curr_ee_goal_cart_world[env_ids]=self._get_ee_goal_spherical_center()[env_ids]+v
            self.ee_goal_orn_quat[env_ids]=quat_from_euler_xyz(zero+self.cfg.goal_ee.orientation_roll_offset,
                -self.curr_ee_goal_sphere[env_ids,1]+self.cfg.goal_ee.arm_induced_pitch,yaw+self.curr_ee_goal_sphere[env_ids,2])

    def _collision_check(self,env_ids):
        rejected=super()._collision_check(env_ids)
        q=self.dof_pos[env_ids,12:18].clone()
        body=self.root_states[env_ids,3:7];yaw=self.base_yaw_quat[env_ids]
        origin=self._get_ee_goal_spherical_center()[env_ids]
        for t in (0.,.5,1.):
            sphere=torch.lerp(self.ee_start_sphere[env_ids],self.ee_goal_sphere[env_ids],t)
            v=quat_apply(yaw,sphere2cart(sphere));world=origin+v
            target=quat_rotate_inverse(body,v)
            delta=self.ee_goal_orn_delta_rpy[env_ids]
            orn=quat_from_euler_xyz(delta[:,0]+self.cfg.goal_ee.orientation_roll_offset,
                -sphere[:,1]+self.cfg.goal_ee.arm_induced_pitch+delta[:,1],
                torch.atan2(v[:,1],v[:,0])+delta[:,2])
            # Transform target orientation from world into actual arm axes.
            basis=torch.eye(3,device=self.device).expand(len(env_ids),3,3)
            world_axes=rotate(orn[:,None,:].expand(-1,3,-1),basis)
            local_axes=quat_rotate_inverse(body[:,None,:].expand(-1,3,-1).reshape(-1,4),world_axes.reshape(-1,3)).reshape(-1,3,3).transpose(1,2)
            q,failed=self.collision_guard.solve(target,local_axes,q)
            rejected|=failed|self.collision_guard.collides(q,self.arm_base_offset[env_ids,None,:])
            rejected|=world[:,2]<self.env_origins[env_ids,2]+.03
        self._last_rejected_ids=env_ids[rejected]
        return rejected
