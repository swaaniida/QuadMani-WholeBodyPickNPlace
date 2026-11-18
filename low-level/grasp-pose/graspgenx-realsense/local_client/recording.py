import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
from official_viser_helpers import matrix_to_wxyz_position


LATENCY_FIELDS = [
    'run_id', 'frame_id', 'api', 'planner', 'points', 'grasps',
    'acquisition_ms', 'alignment_ms', 'conversion_ms', 'segmentation_ms',
    'pointcloud_ms', 'downsampling_ms', 'network_rtt_ms', 'server_infer_ms',
    'server_handler_sync_ms', 'visualization_ms', 'save_ms', 'request_total_ms',
    'frame_age_at_request_ms', 'capture_to_result_ms',
]


def append_latency(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LATENCY_FIELDS, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def latency_summary(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    result = {}
    for key in LATENCY_FIELDS:
        if key.endswith('_ms'):
            values = [float(r[key]) for r in rows if r.get(key) not in ('', None)]
            if values:
                result[key] = dict(n=len(values), mean=float(np.mean(values)),
                                   p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)))
    return result


def save_result(root, run_id, frame, points, colors, mask, result, overlay_image, config, roi, api, planner):
    start = time.perf_counter()
    root = Path(root)
    for folder in ('rgb', 'depth', 'pointcloud', 'grasps', 'benchmarks'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / 'rgb' / f'{run_id}.png'), frame.bgr)
    cv2.imwrite(str(root / 'rgb' / f'{run_id}_overlay.png'), overlay_image)
    cv2.imwrite(str(root / 'depth' / f'{run_id}_mask.png'), mask.astype(np.uint8)*255)
    np.savez_compressed(root / 'depth' / f'{run_id}.npz', depth_raw=frame.depth_raw,
                        depth_m=frame.depth_m, K=frame.K, depth_scale=frame.depth_scale)
    np.savez_compressed(root / 'pointcloud' / f'{run_id}.npz', point_cloud=points, colors=colors)
    grasps, scores = result['grasps'], result['confidences']
    ranking_fields = {key: result[key] for key in ('selection_scores', 'orientation_scores', 'original_candidate_indices') if key in result}
    np.savez_compressed(root / 'grasps' / f'{run_id}.npz', point_cloud=points, colors=colors,
                        grasps=grasps, confidences=scores,
                        branch_tags=np.asarray(result['branch_tags']), frame='camera_optical', **ranking_fields)
    entries = []
    for i, (T, score, tag) in enumerate(zip(grasps, scores, result['branch_tags'])):
        wxyz, position = matrix_to_wxyz_position(T)
        entries.append(dict(rank=i+1, T_camera_gripper_base=T.tolist(), position_m=position.tolist(),
                            quaternion_xyzw=wxyz[[1, 2, 3, 0]].tolist(), confidence=float(score), branch=tag))
        if 'selection_scores' in result:
            entries[-1].update(final_score=float(result['selection_scores'][i]),
                               R=float(result['orientation_scores'][i, 0]),
                               P=float(result['orientation_scores'][i, 1]),
                               Y=float(result['orientation_scores'][i, 2]),
                               original_candidate_index=int(result['original_candidate_indices'][i]),
                               selected=(i == 0),
                               confidence_best=(int(result['original_candidate_indices'][i]) ==
                                                result['selection_metadata']['confidence_best_original_index']))
    metadata = dict(run_id=run_id, frame_id=frame.frame_id,
                    source=getattr(frame, 'source', 'realsense_usb'),
                    camera_timestamp_ms=frame.camera_timestamp_ms, host_timestamp_s=frame.host_timestamp_s,
                    serial=frame.serial, frame='camera_optical', units='meter',
                    pose_convention='T_camera_gripper_base; not calibrated robot TCP',
                    intrinsics=frame.intrinsics, color_intrinsics=frame.color_intrinsics,
                    depth_scale=frame.depth_scale, roi=list(roi), filter_config=vars(config),
                    api=api, planner=planner, gripper='franka_umi',
                    server_timing=result['server_timing'], gpu=result['gpu'], grasps=entries)
    if 'selection_metadata' in result:
        metadata['selection'] = result['selection_metadata']
    provenance = Path(__file__).parent / 'assets/provenance.json'
    if provenance.exists():
        metadata['model_and_gripper_provenance'] = json.loads(provenance.read_text(encoding='utf-8'))
    if 'tracking_metadata' in result:
        metadata['tracking'] = result['tracking_metadata']
        tracking_provenance = Path(__file__).parent / 'assets/tracking_provenance.json'
        if tracking_provenance.exists():
            metadata['tracking']['provenance'] = json.loads(tracking_provenance.read_text(encoding='utf-8'))
    (root / 'grasps' / f'{run_id}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return (time.perf_counter()-start)*1000


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    print(json.dumps(latency_summary(p.parse_args().csv), indent=2))
