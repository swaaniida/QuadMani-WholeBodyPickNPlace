"""Record pre-reset physical state and actual reward calls without changing rewards."""
import json
from pathlib import Path
import numpy as np
import torch
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

class Meter:
    def __init__(self, env, verify=False):
        self.env = env
        self.verify = verify
        self.clear()
    def clear(self):
        self.physics = []
        self.rewards = []
        self.current = {}
    def record(self, key, raw, scale):
        raw = torch.as_tensor(raw, device=self.env.device).expand(self.env.num_envs)
        self.current[key] = (raw.detach().clone(), float(scale))
    def sample(self):
        e = self.env
        if 'leg/feet_contact_forces' not in self.current:
            # Pure diagnostic call: this function does not mutate timers/state.
            raw, _ = e.reward_container._reward_feet_contact_forces()
            self.record('leg/feet_contact_forces', raw, 0.)
        force = e.contact_forces[:, e.feet_indices].norm(dim=-1)
        error = e.base_lin_vel[:, :2] - e.commands[:, :2]
        yaw_error = e.base_ang_vel[:, 2] - e.commands[:, 2]
        ee = (e.curr_ee_goal_cart_world - e.ee_pos).norm(dim=-1)
        self.physics.append(torch.cat([error, yaw_error[:, None], e.base_lin_vel[:, :2], e.base_ang_vel[:, 2:3],
            ee[:, None], e.reset_buf[:, None].float(), e.time_out_buf[:, None].float(), force,
            e.rew_buf[:, None] * 100, e.arm_rew_buf[:, None] * 100], dim=-1).detach().clone())
        if self.verify:
            for branch, target in [('leg', e.rew_buf), ('arm', e.arm_rew_buf)]:
                expected = sum(v * w for k, (v, w) in self.current.items() if k.startswith(branch + '/')) / 100
                assert torch.allclose(expected, target, atol=1e-6)
        self.rewards.append(self.current)
        self.current = {}
    def summary(self, selection=slice(None)):
        if not self.physics:
            return {}
        x = torch.stack(self.physics)[:, selection].cpu().numpy()
        n = x.shape[0] * x.shape[1]
        force = x[:, :, 9:13]
        metrics = {'Tracking/vx_rmse_m_s': float(np.sqrt(np.mean(x[:, :, 0] ** 2))),
            'Tracking/vy_rmse_m_s': float(np.sqrt(np.mean(x[:, :, 1] ** 2))),
            'Tracking/yaw_rmse_rad_s': float(np.sqrt(np.mean(x[:, :, 2] ** 2))),
            'Tracking/vx_mean_m_s': float(x[:, :, 3].mean()), 'Tracking/yaw_mean_rad_s': float(x[:, :, 5].mean()),
            'EE/position_mean_m': float(x[:, :, 6].mean()), 'EE/position_p95_m': float(np.quantile(x[:, :, 6], .95)),
            'Reset/failures': int((x[:, :, 7] * (1 - x[:, :, 8])).sum()), 'Reset/timeouts': int(x[:, :, 8].sum()),
            'Reset/failures_per_1000_env_steps': float((x[:, :, 7] * (1 - x[:, :, 8])).sum() * 1000 / n),
            'Samples/env_steps': int(n), 'Force/norm_mean_N': float(force.mean()),
            'Force/norm_p95_N': float(np.quantile(force, .95)), 'Force/norm_p99_N': float(np.quantile(force, .99))}
        for j, name in enumerate(self.env.feet_names if hasattr(self.env, 'feet_names') else ['FL', 'FR', 'RL', 'RR']):
            for tag, value in [('mean', force[:, :, j].mean()), ('p95', np.quantile(force[:, :, j], .95)), ('p99', np.quantile(force[:, :, j], .99))]:
                metrics['Force/' + name + '_' + tag + '_N'] = float(value)
        weighted = {}
        for key in self.rewards[0]:
            raw = torch.stack([v[key][0] for v in self.rewards])[:, selection].cpu().numpy()
            scale = self.rewards[0][key][1]
            weighted[key] = raw * scale
            metrics['RewardRaw/' + key] = float(raw.mean())
            metrics['RewardWeighted/' + key] = float((raw * scale).mean())
        for branch in ['leg', 'arm']:
            values = {k: v for k, v in weighted.items() if k.startswith(branch + '/')}
            total = sum(v.mean() for v in values.values())
            absolute = sum(np.abs(v).mean() for v in values.values())
            metrics['RewardTotal/' + branch + '_before_div100'] = float(total)
            for key, value in values.items():
                metrics['RewardAbsShare/' + key] = float(np.abs(value).mean() / absolute) if absolute else 0.
                metrics['RewardSignedShare/' + key] = float(value.mean() / total) if abs(total) > 1e-9 else 0.
        return metrics
    def save_samples(self, path):
        arrays = {'physics': torch.stack(self.physics).cpu().numpy()}
        arrays.update({key.replace('/', '__'): torch.stack([v[key][0] for v in self.rewards]).cpu().numpy() for key in self.rewards[0]})
        np.savez_compressed(path, **arrays)

class MeteredEnvironment(ManipLoco):
    def __init__(self, *args, **kwargs):
        self.meter = None
        super().__init__(*args, **kwargs)
        self.meter = Meter(self)
        for branch, names, functions, scales in [('leg', self.reward_names, self.reward_functions, self.reward_scales),
                                                ('arm', self.arm_reward_names, self.arm_reward_functions, self.arm_reward_scales)]:
            for i, (name, fn) in enumerate(zip(names, functions)):
                def record(fn=fn, key=branch + '/' + name, scale=scales[name]):
                    raw, metric = fn()
                    self.meter.record(key, raw, scale)
                    return raw, metric
                functions[i] = record
    def compute_reward(self):
        super().compute_reward()
        if self.meter is not None:
            self.meter.sample()

class AblationRunner(OnPolicyRunner):
    def log(self, locs, *args, **kwargs):
        import wandb
        values = self.env.meter.summary()
        if values:
            wandb.log(values, step=locs['it'], commit=False)
            with (Path(self.log_dir) / 'ablation_metrics.jsonl').open('a') as f:
                f.write(json.dumps({'iteration': locs['it'], **values}) + '\n')
        self.env.meter.clear()
        return super().log(locs, *args, **kwargs)
