"""Real server integration test with official sample; no camera or mock inference."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local_client'))
from grasp_client import GraspClient, load_sweep_params


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--port', type=int, default=5556)
    args = p.parse_args()
    pc_file = sorted((ROOT / 'upstream/GraspGenX/assets/sample_data/object_pc').glob('*.json'))[0]
    pc = np.asarray(json.loads(pc_file.read_text())['pc'], dtype=np.float32)
    client = GraspClient(port=args.port)
    out = ROOT / 'outputs/benchmarks'
    try:
        health = client.health()
        assert health == {'status': 'ok'}
        metadata = client.metadata()
        assert metadata['default_gripper'] == 'franka_umi'
        (out/'health.json').write_text(json.dumps(dict(health=health, metadata=metadata), indent=2))
        rows = []
        for api, planner in [('name', 'diffusion'), ('params', 'graspmoe'), ('params', 'diffusion')]:
            for i in range(args.runs+1):
                result = client.infer(pc, api=api, planner=planner, sweep_params=load_sweep_params())
                assert len(result['grasps']) == 100
                assert np.all(np.diff(result['confidences']) <= 0)
                row = dict(api=api, planner=planner, iteration=i, warmup=i == 0,
                           network_rtt_ms=result['network_rtt_ms'],
                           server_infer_ms=result['server_timing']['infer_ms'],
                           server_handler_sync_ms=result['server_timing'].get('handler_sync_ms'),
                           output_grasps=len(result['grasps']), **result['gpu'])
                rows.append(row)
                print(json.dumps(row), flush=True)
                np.savez_compressed(ROOT/f'outputs/grasps/network_sample_{api}_{planner}.npz',
                                    point_cloud=pc, grasps=result['grasps'], confidences=result['confidences'],
                                    branch_tags=np.asarray(result['branch_tags']), frame='official_sample_input')
        # Official error handling and REP recovery, using the real loaded model server.
        errors = []
        for payload in [dict(action='unknown'), dict(action='infer', point_cloud=np.zeros((3, 2), np.float32)),
                        dict(action='infer_object', point_cloud=pc)]:
            try:
                client.request(payload)
                raise AssertionError('Malformed request unexpectedly succeeded')
            except RuntimeError as exc:
                errors.append(str(exc))
            assert client.health()['status'] == 'ok'
        with (out/'network.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {}
        for api, planner in [('name', 'diffusion'), ('params', 'graspmoe'), ('params', 'diffusion')]:
            selected = [r for r in rows if r['api'] == api and r['planner'] == planner and not r['warmup']]
            summary[f'{api}/{planner}'] = {}
            for metric in ('network_rtt_ms', 'server_infer_ms', 'server_handler_sync_ms'):
                values = [r[metric] for r in selected if r[metric] is not None]
                if values:
                    summary[f'{api}/{planner}'][metric] = dict(mean=float(np.mean(values)),
                                                              p50=float(np.percentile(values,50)),
                                                              p95=float(np.percentile(values,95)))
        assert 'torch' not in sys.modules, 'Client imported torch'
        (out/'network_summary.json').write_text(json.dumps(dict(summary=summary, errors=errors,
                                                               client_imported_torch=False), indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        client.close()
