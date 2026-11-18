"""Official ZMQ server plus synchronized timing; no inference reimplementation."""
import argparse
import logging
import time
from pathlib import Path
import torch
from graspgenx.serving.zmq_server import GraspGenXZMQServer

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream/GraspGenX'


class MeasuredServer(GraspGenXZMQServer):
    def _dispatch(self, request):
        if request.get('action', '').startswith('infer'):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = super()._dispatch(request)
            torch.cuda.synchronize()
            # Entire handler incl. preprocessing, cached sampler lookup, scoring,
            # and device→host outputs; excludes ZMQ serialization/transport.
            result['timing']['handler_sync_ms'] = (time.perf_counter() - start) * 1000
            result['gpu'] = {
                'allocated_mib': torch.cuda.memory_allocated() / 2**20,
                'reserved_mib': torch.cuda.memory_reserved() / 2**20,
                'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
            }
            return result
        return super()._dispatch(request)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=5556)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    MeasuredServer(
        config_path=str(UPSTREAM / 'ext/graspgenx_checkpoints/release'),
        assets_dir=str(UPSTREAM / 'ext/gripper_descriptions/gripper_descriptions/assets'),
        host='127.0.0.1', port=args.port, default_gripper='franka_umi',
    ).serve_forever()
