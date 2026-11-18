"""Replay existing real SAM/AOT + GraspGenX outputs; no model inference or camera.
Creates ranked results, comparison overlay and a CSV from the recorded RGB-D test.
"""
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import cv2
import numpy as np
import requests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'local_client'))
from grasp_ranking import rerank_result, rank_grasps, display_order
from pointcloud import FilterConfig
from recording import save_result
from visualization import GraspViewer, overlay


def main(source_folder, stem, run_id):
    source = ROOT/source_folder
    metadata = json.loads((source/'grasps'/f'{stem}.json').read_text())
    with np.load(source/'grasps'/f'{stem}.npz') as data:
        points, colors = data['point_cloud'], data['colors']
        raw = dict(grasps=data['grasps'], confidences=data['confidences'],
                   branch_tags=data['branch_tags'], server_timing=metadata['server_timing'], gpu=metadata['gpu'])
    result = rerank_result(raw, points)
    depth = np.load(source/'depth'/f'{stem}.npz')
    frame = SimpleNamespace(**{key: metadata[key] for key in (
        'frame_id','source','camera_timestamp_ms','host_timestamp_s','serial','intrinsics', 'color_intrinsics','depth_scale')},
        bgr=cv2.imread(str(source/'rgb'/f'{stem}.png')), depth_raw=depth['depth_raw'], depth_m=depth['depth_m'], K=depth['K'])
    image = overlay(frame.bgr,frame.K,result['grasps'],result['confidences'],frame.frame_id,
                    frame.color_intrinsics,point_cloud=points,original_indices=result['original_candidate_indices'])
    mask = cv2.imread(str(source/'depth'/f'{stem}_mask.png'),0)>0
    out = ROOT/'outputs/ranking'
    save_result(out,run_id,frame,points,colors,mask,result,image,
                FilterConfig(**metadata['filter_config']),metadata['roi'],metadata['api'],metadata['planner'])
    ranking = rank_grasps(result['grasps'],result['confidences'],points,result['original_candidate_indices'])
    cb = ranking['confidence_best']
    rows=[]
    for i in range(len(result['grasps'])):
        rows.append(dict(rank=i+1,original_index=int(result['original_candidate_indices'][i]),
                         C=float(result['confidences'][i]),R=float(result['orientation_scores'][i,0]),
                         P=float(result['orientation_scores'][i,1]),Y=float(result['orientation_scores'][i,2]),
                         S=float(result['selection_scores'][i]),original_confidence_best=(i==cb)))
    with (out/f'{run_id}_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    durations=[]
    for _ in range(100):
        start=time.perf_counter();rerank_result(raw,points);durations.append((time.perf_counter()-start)*1000)
    summary=dict(source=str(source/'grasps'/f'{stem}.npz'), validation='Previously recorded actual model inference; no new inference',
                 candidates=len(rows), selected=rows[0], confidence_best=rows[cb],
                 rerank_ms=dict(mean=float(np.mean(durations)),p95=float(np.percentile(durations,95))))
    viewer=GraspViewer(port=18089)
    try:
        viewer.show(points,colors,result['grasps'],result['confidences'],frame.frame_id,
                    original_indices=result['original_candidate_indices'])
        assert 'S#1' in viewer.details.content and 'C#1' in viewer.details.content
        assert len(viewer.mesh_handles)==len(display_order(ranking))
        assert requests.get('http://127.0.0.1:18089',timeout=10).status_code==200
        summary['viewer_meshes']=len(viewer.mesh_handles)
        summary['viewer_http']=200
    finally:
        viewer.close()
    reloaded=np.load(out/'grasps'/f'{run_id}.npz')
    np.testing.assert_array_equal(reloaded['confidences'],raw['confidences'][result['original_candidate_indices']])
    np.testing.assert_allclose(reloaded['selection_scores'],result['selection_scores'])
    summary['save_roundtrip']='passed'
    (out/f'{run_id}_verification.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='outputs/tracking')
    p.add_argument('--stem', default='integration_static_rgbd')
    p.add_argument('--run-id', default='tracking_ranked')
    args=p.parse_args()
    main(args.source,args.stem,args.run_id)
