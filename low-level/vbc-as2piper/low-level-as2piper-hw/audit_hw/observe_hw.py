"""Observe the real PPO run before goal updates/resets can hide EE errors."""
import hashlib,json,os,time,types
from pathlib import Path
import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import euler_from_quat
import torch
import numpy as np
from legged_gym.scripts import train_hw as train_as2
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.manip_loco.as2piper_hw import AS2PiPERHardware as AS2PiPERForward

ROOT=Path(__file__).resolve().parent.parent
OUT=Path(os.environ.get('EE_AUDIT_DIR',str(ROOT/'audit_hw/ppo20')))
OUT.mkdir(parents=True,exist_ok=True)
GROUPS=['all','upright_age_gt_0p5','upright_random_age_gt_0p5','fallen','upright_random_hold']
METRICS=['count','position_sum_m','position_sq_sum_m2','orientation_sum_deg','position_lt_2cm','position_lt_5cm','position_lt_10cm','orientation_lt_5deg','orientation_lt_10deg','pose_lt_5cm_10deg','arm_contact_gt_1N','arm_force_abs_max_sum_Nm']
RESETS=['done','roll','pitch','height','timeout','roll_exclusive','pitch_exclusive','height_exclusive','timeout_exclusive','episode_steps_sum']
SAMPLING=['requested','accepted','fallback','candidate_count','candidate_rejected','candidate_ik_rejected','candidate_body_rejected']
observer=None


class Observer:
    def __init__(self,env):
        self.e=env;self.active=False;self.step_count=0;self.iteration=0;self.rows=[]
        dev=env.device;n=env.num_envs
        self.phase=torch.zeros(n,device=dev,dtype=torch.bool)
        self.metrics=torch.zeros(len(GROUPS),len(METRICS),device=dev)
        self.resets=torch.zeros(len(RESETS),device=dev);self.sampling=torch.zeros(len(SAMPLING),device=dev)
        self.sample_ids=torch.linspace(0,n-1,min(512,n),device=dev).long()
        self.video_ids=torch.linspace(0,n-1,min(8,n),device=dev).long()
        mass=env.mass_params_tensor.detach().cpu().numpy()
        least=int(np.argmin((mass[:,0]/3.)**2+np.sum((mass[:,1:4]/.03)**2,axis=1)))
        self.video_ids[0]=least
        self.samples=[];self.frames=[];self.scratch=None
        self.arm_bodies=[i for i,name in enumerate(env.body_names) if name.startswith('piper_') and name!='piper_base_link']
        self.forces=gymtorch.wrap_tensor(env.gym.acquire_dof_force_tensor(env.sim)).view(n,env.num_dofs)
        self.applied=None
        self.install()
        (OUT/'metadata.json').write_text(json.dumps(dict(num_envs=n,num_dofs=env.num_dofs,device=str(dev),groups=GROUPS,metrics=METRICS,resets=RESETS,sampling=SAMPLING,
            sample_ids=self.sample_ids.tolist(),video_ids=self.video_ids.tolist(),body_names=env.body_names,dof_names=env.dof_names,
            video_mass_params=mass[self.video_ids.cpu().numpy()].tolist(),least_perturbed_env=least,
            goal_config=class_to_dict(env.cfg.goal_ee),domain_randomization=class_to_dict(env.cfg.domain_rand),
            sample_columns=['applied_position_error_m','applied_orientation_error_deg','episode_age_s','upright','random_command','hold','done','arm_contact_N','arm_dof_force_abs_max_Nm'],
            measurement='Actual post-physics EE against target applied at step start; captured before reset. Sensors enabled only in this audit process.'),indent=2))

    def install(self):
        e=self.e;old_step=e.step;old_check=e.check_termination;old_resample=e._resample_ee_goal;old_collision=e._collision_check
        old_solve=e.collision_guard.solve;old_collides=e.collision_guard.collides
        def step(actions):
            if self.active:
                self.applied=(e.curr_ee_goal_cart_world.clone(),e.ee_goal_orn_quat.clone(),self.phase.clone(),(e.goal_timer>=e.traj_timesteps).clone())
            return old_step(actions)
        def check():
            old_check()
            if self.active and self.applied is not None:self.capture()
        def resample(ids,is_init=False):
            previous=self.phase.clone()
            old_resample(ids,is_init=is_init)
            if is_init:self.phase[ids]=False
            else:
                self.phase[ids]=True;failed=e._last_rejected_ids;self.phase[failed]=previous[failed]
                if self.active:
                    self.sampling[0]+=len(ids);self.sampling[1]+=len(ids)-len(failed);self.sampling[2]+=len(failed)
        def collision(ids):
            if not self.active:return old_collision(ids)
            self.scratch=[torch.zeros(len(ids),device=e.device,dtype=torch.bool) for _ in range(2)]
            result=old_collision(ids)
            self.sampling[3]+=len(ids);self.sampling[4]+=result.sum()
            self.sampling[5]+=self.scratch[0].sum();self.sampling[6]+=self.scratch[1].sum();self.scratch=None
            return result
        def solve(*args,**kwargs):
            q,failed=old_solve(*args,**kwargs)
            if self.scratch is not None:self.scratch[0]|=failed
            return q,failed
        def collides(*args,**kwargs):
            result=old_collides(*args,**kwargs)
            if self.scratch is not None:self.scratch[1]|=result
            return result
        e.step=step;e.check_termination=check;e._resample_ee_goal=resample;e._collision_check=collision
        e.collision_guard.solve=solve;e.collision_guard.collides=collides

    @torch.no_grad()
    def capture(self):
        e=self.e;target,rotation,random,hold=self.applied
        position=(e.ee_pos-target).norm(dim=-1)
        actual=e.ee_orn/e.ee_orn.norm(dim=-1,keepdim=True)
        orientation=torch.rad2deg(2*torch.acos((actual*rotation).sum(-1).abs().clamp(0,1)))
        r,p,_=euler_from_quat(e.root_states[:,3:7]);height=e.root_states[:,2]-e.env_origins[:,2]
        upright=(r.abs()<.35)&(p.abs()<.35)&(height>.22)
        age=e.episode_length_buf*e.dt;late=age>.5
        e.gym.refresh_dof_force_tensor(e.sim)
        torque=self.forces[:,12:18].abs().max(dim=-1).values
        contact=e.contact_forces[:,self.arm_bodies].norm(dim=-1).max(dim=-1).values
        groups=torch.stack((torch.ones_like(upright),upright&late,upright&late&random,~upright,upright&late&random&hold)).float()
        values=torch.stack((torch.ones_like(position),position,position*position,orientation,(position<.02).float(),(position<.05).float(),(position<.1).float(),
                            (orientation<5).float(),(orientation<10).float(),((position<.05)&(orientation<10)).float(),(contact>1).float(),torque),dim=-1)
        self.metrics+=groups@values
        done=e.reset_buf.bool();roll=r.abs()>.8;pitch=p.abs()>.8;z=e.root_states[:,2]<.1;timeout=e.time_out_buf
        exclusive=[done&roll,done&~roll&pitch,done&~roll&~pitch&z,done&~roll&~pitch&~z&timeout]
        counts=[done.sum(),(done&roll).sum(),(done&pitch).sum(),(done&z).sum(),(done&timeout).sum()]+[x.sum() for x in exclusive]+[(e.episode_length_buf*done).sum()]
        self.resets+=torch.stack(counts)
        ids=self.sample_ids
        self.samples.append(torch.stack((position[ids],orientation[ids],age[ids],upright[ids].float(),random[ids].float(),hold[ids].float(),done[ids].float(),contact[ids],torque[ids]),dim=-1).cpu().numpy())
        if self.step_count%2==0:
            ids=self.video_ids
            self.frames.append(dict(step=self.step_count,iteration=self.iteration,
                root=(e.root_states[ids,:7]-torch.cat((e.env_origins[ids],torch.zeros(len(ids),4,device=e.device)),dim=-1)).cpu().numpy(),
                q=e.dof_pos[ids].cpu().numpy(),target=(target[ids]-e.env_origins[ids]).cpu().numpy(),target_quat=rotation[ids].cpu().numpy(),
                ee=(e.ee_pos[ids]-e.env_origins[ids]).cpu().numpy(),done=done[ids].cpu().numpy(),random=random[ids].cpu().numpy(),
                upright=upright[ids].cpu().numpy(),position_error=position[ids].cpu().numpy()))
        self.step_count+=1

    def log(self,iteration):
        self.iteration=iteration+1
        metrics=self.metrics.cpu().numpy();resets=self.resets.cpu().numpy();sampling=self.sampling.cpu().numpy()
        row=dict(iteration=iteration,groups={g:dict(zip(METRICS,map(float,v))) for g,v in zip(GROUPS,metrics)},
                 resets=dict(zip(RESETS,map(float,resets))),sampling=dict(zip(SAMPLING,map(float,sampling))))
        self.rows.append(row)
        with (OUT/'iterations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        all_group=row['groups']['all'];up=row['groups']['upright_random_age_gt_0p5']
        print('EE_AUDIT '+json.dumps(dict(iteration=iteration,position_mean_m=all_group['position_sum_m']/max(1,all_group['count']),
            upright_random_count=up['count'],upright_random_position_mean_m=up['position_sum_m']/max(1,up['count']),
            sampling=row['sampling'],resets=row['resets'])),flush=True)
        self.metrics.zero_();self.resets.zero_();self.sampling.zero_()
        if (iteration+1)%10==0:self.save()

    def save(self):
        if self.samples:np.savez_compressed(str(OUT/'tracking_samples.npz'),samples=np.stack(self.samples))
        if self.frames:
            keys=self.frames[0].keys();np.savez_compressed(str(OUT/'trajectories.npz'),**{k:np.stack([f[k] for f in self.frames]) for k in keys})


# Sensors are enabled before gym.prepare_sim, only for this diagnostic process.
original_create=AS2PiPERForward._create_envs
def create_with_sensors(self):
    original_create(self)
    for env,actor in zip(self.envs,self.actor_handles):self.gym.enable_actor_dof_force_sensors(env,actor)
AS2PiPERForward._create_envs=create_with_sensors
original_make=task_registry.make_env
def make_env(*args,**kwargs):
    global observer
    env,cfg=original_make(*args,**kwargs);observer=Observer(env);return env,cfg
task_registry.make_env=make_env
original_runner=task_registry.make_alg_runner
def make_runner(*args,**kwargs):
    result=original_runner(*args,**kwargs);runner=result[0];original_log=runner.log
    def log(self,locs,*args,**kwargs):
        original_log(locs,*args,**kwargs);observer.log(locs['it'])
    runner.log=types.MethodType(log,runner);observer.active=True
    return result
task_registry.make_alg_runner=make_runner


if __name__=='__main__':
    start=time.monotonic()
    try:train_as2.main()
    finally:
        if observer is not None:observer.save()
        (OUT/'wall_time.json').write_text(json.dumps(dict(seconds=time.monotonic()-start)))
