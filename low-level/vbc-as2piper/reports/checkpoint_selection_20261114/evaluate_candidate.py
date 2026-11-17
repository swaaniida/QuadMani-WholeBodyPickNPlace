"""Evaluation only: original 40 s protocol, with pre-reset attitude diagnostics."""
import argparse
import json
from pathlib import Path

from force_ablation import run
from force_ablation.runtime import Meter
import numpy as np
import torch

original_clear = Meter.clear
original_sample = Meter.sample
original_summary = Meter.summary
original_save = Meter.save_samples


def clear(self):
    original_clear(self)
    self.attitude = []


def sample(self):
    original_sample(self)
    e = self.env
    x, y, z, w = e.base_quat.unbind(-1)
    roll = torch.atan2(2 * (w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = torch.asin((2 * (w*y - z*x)).clamp(-1, 1))
    dot = (e.ee_goal_orn_quat * e.ee_orn).sum(-1).abs().clamp(0, 1)
    angle = 2 * torch.acos(dot)
    self.attitude.append(torch.stack([roll, pitch, angle], -1).detach().clone())


def summary(self, selection=slice(None)):
    result = original_summary(self, selection)
    if self.attitude:
        a = torch.stack(self.attitude)[:, selection].cpu().numpy() * (180 / np.pi)
        for j, name in enumerate(['roll', 'pitch', 'ee_orientation_error']):
            b = a[:, :, j]
            result['Pose/' + name + '_signed_mean_deg'] = float(b.mean())
            result['Pose/' + name + '_abs_mean_deg'] = float(np.abs(b).mean())
            result['Pose/' + name + '_abs_p95_deg'] = float(np.quantile(np.abs(b), .95))
            result['Pose/' + name + '_abs_max_deg'] = float(np.abs(b).max())
    return result


def save(self, path):
    original_save(self, path)
    np.savez_compressed(Path(path).with_name('attitude.npz'),
                        attitude_rad=torch.stack(self.attitude).cpu().numpy(),
                        columns=['roll', 'pitch', 'ee_orientation_error'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--terrain', choices=['flat', 'rough'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()
    destination = Path(a.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    run.new_directory = lambda *args, **kwargs: destination
    Meter.clear = clear
    Meter.sample = sample
    Meter.summary = summary
    Meter.save_samples = save
    torch.set_num_threads(2)
    run.register_tasks()
    args = argparse.Namespace(checkpoint=a.checkpoint, terrain=a.terrain,
                              seed=a.seed, device='cuda:0', condition='B',
                              seconds=40., name=destination.name)
    run.evaluate(args)
    result = json.loads((destination / 'result.json').read_text())
    result['extra_diagnostics'] = 'Pre-reset root quaternion roll/pitch and EE orientation; evaluation-only hooks; original reward/control/PPO files unchanged'
    (destination / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
