"""Capture deterministic PhysX states using the existing checkpoint evaluation protocol."""
import argparse
import hashlib
import json
from pathlib import Path
from .run import environment, build_model, source_hashes, model_hash
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--terrain', choices=['flat', 'rough'], required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    torch.set_num_threads(2)
    checkpoint = Path(a.checkpoint).resolve()
    saved = json.loads((checkpoint.parent / 'manifest.json').read_text())
    assert saved['condition'] == 'B' and saved['source_sha256'] == source_hashes()
    out = Path(a.output).resolve() / a.terrain
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'states.npz').exists() and not (out/'metadata.json').exists()
    env, train = environment('B', 1, a.device, small=True, terrain=a.terrain)
    try:
        model = build_model(train, env.device)
        weights = torch.load(str(checkpoint), map_location=env.device)
        model.load_state_dict(weights['model_state_dict'], strict=True)
        model.eval()
        before = model_hash(model)
        targets = torch.tensor([[0., 0., 0.], [.4, 0., 0.], [0., 0., .6], [.4, 0., .6]], device=env.device).repeat_interleave(8, 0)
        ids = [0, 8, 16, 24]  # First replica of each command; no best-case selection.
        env._resample_commands = lambda indices: env.commands.__setitem__(indices, targets[indices])
        env.reset()
        env.global_steps = int(weights['iter']) * train.runner.num_steps_per_env
        env.cfg.env.teleop_mode = True
        records = {k: [] for k in ['time', 'root', 'q', 'target', 'ee', 'velocity', 'yaw', 'done']}
        with torch.no_grad():
            for i in range(int(10. / env.dt)):
                t = i * env.dt
                env.curr_ee_goal_sphere[:] = torch.tensor([.44 + .02*np.sin(t), .3 + .1*np.sin(t*.7), .15*np.sin(t*.5)], device=env.device)
                env.ee_goal_orn_delta_rpy.zero_()
                env._update_curr_ee_goal()
                env.commands.copy_(targets)
                env.step(model.act_inference(env.obs_buf, True).clone())
                if i % 2 == 1:
                    records['time'].append((i+1)*env.dt)
                    for key, value in [('root', env.root_states[:, :7]), ('q', env.dof_pos),
                                       ('target', env.curr_ee_goal_cart_world), ('ee', env.ee_pos),
                                       ('velocity', env.base_lin_vel[:, 0]), ('yaw', env.base_ang_vel[:, 2]),
                                       ('done', env.reset_buf)]:
                        records[key].append(value[ids].cpu().numpy().copy())
                if i % 100 == 99:
                    env.meter.clear()
        assert model_hash(model) == before
        arrays = {k: np.asarray(v) for k, v in records.items()}
        np.savez_compressed(str(out/'states.npz'), **arrays)
        # PhysX mesh coordinates include a -border translation at insertion.
        vertices = env.terrain.vertices.copy()
        vertices[:, :2] -= env.cfg.terrain.border_size
        triangles = env.terrain.triangles
        for j in range(4):
            xy = arrays['root'][:, j, :2]
            inside = ((vertices[:, :2] >= xy.min(0)-2.) & (vertices[:, :2] <= xy.max(0)+2.)).all(-1)
            selected = triangles[inside[triangles].any(axis=1)]
            used, indices = np.unique(selected, return_inverse=True)
            with (out/('terrain_%d.obj' % j)).open('w') as f:
                np.savetxt(f, vertices[used], fmt='v %.7f %.7f %.7f')
                np.savetxt(f, indices.reshape(-1, 3)+1, fmt='f %d %d %d')
        meta = {'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'iteration': int(weights['iter']), 'terrain': a.terrain, 'dof_names': env.dof_names, 'device':a.device,
                'env_ids': ids, 'commands': targets[ids].cpu().tolist(), 'seed': 1, 'fps': 25,
                'protocol': 'Existing 32-env evaluation; first replica per command; same central EE trajectory; DR/push off; 0 optimizer updates',
                'source_sha256': source_hashes(), 'model_unchanged': True,
                'urdf': str(Path(__file__).resolve().parents[1]/'resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf')}
        (out/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
        print('Saved video states:', out, flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
