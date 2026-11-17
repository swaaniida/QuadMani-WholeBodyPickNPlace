"""Replay the existing vy=0 dynamic protocol with four candidate checkpoints."""
import argparse
import json
from pathlib import Path
import sys

from force_ablation import locomotion_vy0_compare as original
from force_ablation.runtime import Meter
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, nargs=4, required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
clone = Path(original.__file__).resolve().parents[1]
run = clone / 'experiments/contact_force_ablation/runs/as2piper_headpiper_orin_200N_0.18m'
attitude_records = []
old_sample = Meter.sample


def sample(self):
    old_sample(self)
    x, y, z, w = self.env.base_quat.unbind(-1)
    roll = torch.atan2(2 * (w*x+y*z), 1-2*(x*x+y*y))
    pitch = torch.asin((2*(w*y-z*x)).clamp(-1, 1))
    attitude_records.append(torch.stack([roll, pitch], -1).detach().cpu().numpy())


Meter.sample = sample
source = Path(original.__file__).read_text()
needle = 'iterations = [10000, 20000, 30000, 45000]'
assert source.count(needle) == 1
source = source.replace(needle, 'iterations = ' + repr(args.iterations))
assert source.count('errors = []') == 1
source = source.replace('errors = []', 'errors = []; attitude_records.clear()')
namespace = {'__name__': 'force_ablation.candidate_dynamic', '__package__': 'force_ablation',
             '__file__': original.__file__, 'attitude_records': attitude_records}
sys.argv = [original.__file__, '--run', str(run), '--output', args.output]
exec(compile(source, original.__file__, 'exec'), namespace)
namespace['main']()
a = np.asarray(attitude_records)
assert a.shape == (1800, 32, 2), a.shape
out = Path(args.output)
np.savez_compressed(out/'attitude.npz', attitude_rad=a, columns=['roll', 'pitch'], iterations=args.iterations)
metadata = json.loads((out/'metadata.json').read_text())
metadata['candidate_selection'] = 'Unchanged existing vy=0 dynamic protocol, only checkpoint list changed; pre-reset body attitude added. Optimizer updates=0.'
(out/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
