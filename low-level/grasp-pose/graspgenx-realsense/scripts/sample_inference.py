"""Headless official sample PC + official planner; never reads stored grasps."""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers import run_planner_on_object
from graspgenx.utils.checkpoint_io import load_model_cfg

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / 'upstream/GraspGenX'


def validate(poses, scores):
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4) and len(poses) > 0
    assert scores.shape == (len(poses),)
    assert np.isfinite(poses).all() and np.isfinite(scores).all()
    assert np.all((scores >= 0) & (scores <= 1))
    assert np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-4)
    rot = poses[:, :3, :3]
    assert np.allclose(rot @ rot.transpose(0, 2, 1), np.eye(3), atol=2e-3)
    assert np.allclose(np.linalg.det(rot), 1, atol=2e-3)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--num-grasps', type=int, default=200)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    out = ROOT / 'outputs/benchmarks'
    out.mkdir(parents=True, exist_ok=True)
    sample = sorted((UP / 'assets/sample_data/object_pc').glob('*.json'))[0]
    data = json.loads(sample.read_text())
    pc = np.asarray(data['pc'], dtype=np.float32)
    assert np.isfinite(pc).all()
    ckpt = UP / 'ext/graspgenx_checkpoints/release'
    cfg = load_model_cfg(str(ckpt / 'gen'), str(ckpt / 'dis'))
    start = time.perf_counter()
    sampler = GraspGenXSampler(cfg, 'franka_umi', assets_dir=str(
        UP / 'ext/gripper_descriptions/gripper_descriptions/assets'))
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - start) * 1000
    rows = []
    for planner in ('graspmoe', 'diffusion'):
        for i in range(args.runs + 1):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            poses, scores, tags, _ = run_planner_on_object(
                pc, sampler, planner=planner, num_grasps=args.num_grasps,
                grasp_threshold=-1, topk_num_grasps=100)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - start) * 1000
            validate(poses, scores)
            row = dict(planner=planner, iteration=i, warmup=i == 0,
                       inference_ms=ms, input_points=len(pc), output_grasps=len(poses),
                       peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                       max_score=float(scores.max()), min_score=float(scores.min()))
            rows.append(row)
            print(json.dumps(row), flush=True)
            np.savez_compressed(ROOT / f'outputs/grasps/official_sample_{planner}.npz',
                                point_cloud=pc, colors=np.asarray(data['pc_color']),
                                grasps=poses, confidences=scores,
                                branch_tags=np.asarray(tags), frame='official_sample_input')
    with (out / 'sample.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for planner in ('graspmoe', 'diffusion'):
        times = [r['inference_ms'] for r in rows if r['planner'] == planner and not r['warmup']]
        summary[planner] = dict(mean_ms=float(np.mean(times)), p50_ms=float(np.percentile(times, 50)),
                                p95_ms=float(np.percentile(times, 95)))
    report = dict(upstream_commit=subprocess.check_output(['git', '-C', str(UP), 'rev-parse', 'HEAD'], text=True).strip(),
                  sample=str(sample.relative_to(ROOT)), sample_sha256=hashlib.sha256(sample.read_bytes()).hexdigest(),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=str(torch.cuda.get_device_properties(0)), model_load_ms=load_ms,
                  num_diffusion_grasps=args.num_grasps, rows=rows, summary=summary)
    (out / 'sample_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
