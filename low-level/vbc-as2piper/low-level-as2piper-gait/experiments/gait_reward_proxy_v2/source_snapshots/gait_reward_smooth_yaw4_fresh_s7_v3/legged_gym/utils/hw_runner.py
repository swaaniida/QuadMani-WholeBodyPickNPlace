"""Extended checkpoints; original PPO updates and rollout loop are inherited."""
import os,random
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner


class HardwareRunner(OnPolicyRunner):
    def save(self,path,it,infos=None):
        state=dict(schema_version=2,model_state_dict=self.alg.actor_critic.state_dict(),
            optimizer_state_dict=self.alg.optimizer.state_dict(),
            history_optimizer_state_dict=self.alg.hist_encoder_optimizer.state_dict(),
            iter=it,next_iteration=self.alg.counter,algorithm_counter=self.alg.counter,
            learning_rate=self.alg.learning_rate,global_steps=self.env.global_steps,
            common_step_counter=self.env.common_step_counter,tot_timesteps=self.tot_timesteps,
            tot_time=self.tot_time,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),infos=infos)
        temp=path+'.tmp';torch.save(state,temp);os.chmod(temp,0o664);os.replace(temp,path)

    def load(self,path,load_optimizer=True):
        d=torch.load(path,map_location=self.device)
        self.alg.actor_critic.load_state_dict(d['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(d['optimizer_state_dict'])
            if 'history_optimizer_state_dict' in d:self.alg.hist_encoder_optimizer.load_state_dict(d['history_optimizer_state_dict'])
        self.current_learning_iteration=d.get('next_iteration',d['iter']+1)
        self.alg.counter=d.get('algorithm_counter',self.current_learning_iteration)
        self.alg.learning_rate=d['optimizer_state_dict']['param_groups'][0]['lr']
        self.env.global_steps=d.get('global_steps',24*self.current_learning_iteration)
        self.env.common_step_counter=d.get('common_step_counter',self.env.global_steps)
        self.tot_timesteps=d.get('tot_timesteps',self.current_learning_iteration*24*self.env.num_envs)
        self.tot_time=d.get('tot_time',0.)
        if 'python_rng' in d:
            random.setstate(d['python_rng']);np.random.set_state(d['numpy_rng'])
            torch.set_rng_state(d['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([v.cpu() for v in d['cuda_rng']])
        return d.get('infos')
