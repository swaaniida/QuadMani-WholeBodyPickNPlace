"""Optional real API check using official recorded depth/K/instance annotation."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_client'))
from grasp_client import GraspClient,load_sweep_params,validate_result

if __name__=='__main__':
    sample=ROOT/'upstream/GraspGenX/assets/sample_data/real_world/00'
    depth=np.load(sample/'depth.npy').astype(np.float32)
    md=json.loads((sample/'meta_data.json').read_text())
    mask=(cv2.imread(str(sample/'seg.png'),cv2.IMREAD_UNCHANGED)==md['label_map']['obj_1']).astype(np.int32)
    c=GraspClient()
    try:
        response,rtt=c.infer_scene_depth(depth,np.array(md['intrinsics']),mask,load_sweep_params())
        assert list(response['instance_ids'])==[1]
        g,s=response['grasps'][0],response['confidences'][0]
        validate_result(g,s)
        assert len(g)>0
        np.savez_compressed(ROOT/'outputs/grasps/official_scene_depth.npz',
                            grasps=g,confidences=s,frame='camera_optical')
        report=dict(source='official recorded RGB-D, not Windows USB camera',
                    shape=list(depth.shape),instance_id=1,grasps=len(g),network_rtt_ms=rtt,
                    server_timing=response['timing'],gpu=response.get('gpu'),
                    frame='camera_optical',max_score=float(s.max()))
        (ROOT/'outputs/benchmarks/scene_depth.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally:
        c.close()
