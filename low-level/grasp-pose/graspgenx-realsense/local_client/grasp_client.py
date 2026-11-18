"""CPU-only adapter for the verified official GraspGenX msgpack wire protocol.

No import of graspgenx: its package initializer downloads assets and its full
installation depends on torch. All fields below match the pinned upstream.
"""
import argparse
import json
import time
from pathlib import Path
import msgpack
import msgpack_numpy
import numpy as np
import zmq


def validate_result(grasps, scores):
    if grasps.shape != (len(scores), 4, 4) or scores.ndim != 1:
        raise ValueError('Invalid grasp/score shape from server')
    if not np.isfinite(grasps).all() or not np.isfinite(scores).all():
        raise ValueError('Non-finite result')
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError('Confidence outside [0,1]')
    if len(grasps):
        R = grasps[:, :3, :3]
        if not (np.allclose(grasps[:, 3], [0, 0, 0, 1], atol=1e-4)
                and np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=2e-3)
                and np.allclose(np.linalg.det(R), 1, atol=2e-3)):
            raise ValueError('Server result is not SE(3)')


class GraspClient:
    """One instance per worker thread. Reset REQ socket after any transport error."""
    def __init__(self, host='127.0.0.1', port=5556, timeout_ms=60000):
        self.address = f'tcp://{host}:{port}'
        self.timeout_ms = timeout_ms
        self.socket = None
        self.last_response = None

    def close(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None

    def request(self, payload):
        if self.socket is None:
            self.socket = zmq.Context.instance().socket(zmq.REQ)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.IMMEDIATE, 1)
            self.socket.setsockopt(zmq.SNDHWM, 1)
            self.socket.connect(self.address)
        start = time.perf_counter()
        try:
            self.socket.send(msgpack.packb(payload, default=msgpack_numpy.encode, use_bin_type=True))
            raw = self.socket.recv()
        except zmq.ZMQError as exc:
            self.close()
            raise TimeoutError(f'{self.address}: transport failed; check SSH tunnel/server ({exc})') from exc
        response = msgpack.unpackb(raw, object_hook=msgpack_numpy.decode, raw=False)
        rtt_ms = (time.perf_counter()-start)*1000
        if not isinstance(response, dict):
            raise ValueError('Invalid server response')
        if 'error' in response:
            raise RuntimeError(response['error'])
        self.last_response = response
        return response, rtt_ms

    def health(self):
        return self.request({'action': 'health'})[0]

    def metadata(self):
        return self.request({'action': 'metadata'})[0]

    def infer(self, point_cloud, gripper_name='franka_umi', num_grasps=200,
              grasp_threshold=-1.0, topk_num_grasps=100, *, api='params',
              planner='graspmoe', sweep_params=None):
        pc = np.ascontiguousarray(point_cloud, dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3 or len(pc) < 100 or not np.isfinite(pc).all():
            raise ValueError('Need finite (N,3) metric cloud with >=100 points')
        if not 1 <= num_grasps <= 2000:
            raise ValueError('num_grasps must be 1..2000')
        payload = dict(point_cloud=pc, num_grasps=int(num_grasps))
        if api == 'name':
            if planner != 'diffusion':
                raise ValueError('Official name-based infer supports diffusion only; use params for graspmoe')
            payload.update(action='infer', gripper_name=gripper_name,
                           grasp_threshold=float(grasp_threshold), topk_num_grasps=int(topk_num_grasps))
        elif api == 'params':
            if planner not in ('graspmoe', 'diffusion') or sweep_params is None:
                raise ValueError('params API requires official sweep params and a supported planner')
            payload.update(action='infer_object', sweep_volume_params=sweep_params, planner=planner)
        else:
            raise ValueError('api must be name or params')
        response, rtt = self.request(payload)
        grasps = np.asarray(response['grasps'], dtype=np.float32)
        scores = np.asarray(response['confidences'], dtype=np.float32)
        validate_result(grasps, scores)
        order = np.argsort(-scores, kind='stable')
        if grasp_threshold > 0:
            order = order[scores[order] >= grasp_threshold]
        if topk_num_grasps > 0:
            order = order[:topk_num_grasps]
        tags = response.get('branch_tags', ['diff']*len(scores))
        return dict(grasps=grasps[order], confidences=scores[order],
                    branch_tags=[tags[i] for i in order], network_rtt_ms=rtt,
                    server_timing=response.get('timing', {}), gpu=response.get('gpu', {}))

    def infer_scene_depth(self, depth_m, K, instance_mask, sweep_params,
                          planner='graspmoe', num_grasps=200):
        """Optional official interface; K assumes rectified pinhole depth."""
        depth = np.ascontiguousarray(depth_m, dtype=np.float32)
        mask = np.ascontiguousarray(instance_mask, dtype=np.int32)
        if depth.ndim != 2 or mask.shape != depth.shape or np.asarray(K).shape != (3, 3):
            raise ValueError('Expected depth/mask HxW and intrinsics 3x3')
        return self.request(dict(action='infer_scene_depth', depth=depth,
                                 intrinsics=np.asarray(K, dtype=np.float32), instance_mask=mask,
                                 sweep_volume_params=sweep_params, planner=planner,
                                 num_grasps=int(num_grasps)))


def load_sweep_params():
    return json.loads((Path(__file__).parent / 'assets/franka_umi_sweep.json').read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5556)
    p.add_argument('--timeout-ms', type=int, default=5000)
    args = p.parse_args()
    client = GraspClient(args.host, args.port, args.timeout_ms)
    try:
        print(json.dumps({'health': client.health(), 'metadata': client.metadata()}, indent=2))
    finally:
        client.close()
