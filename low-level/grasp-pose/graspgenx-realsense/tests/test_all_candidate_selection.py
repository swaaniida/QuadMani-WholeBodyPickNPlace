"""Postprocessing regression; no inference or camera requests."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_client'))
from temporal_selection import TemporalConfig,select_candidates


def test_valid_candidate_below_confidence_top100_is_retained():
    g=np.repeat(np.eye(4)[None],101,axis=0)
    g[:100,2,3]=1.1  # All higher-confidence candidates penetrate z=1 ground.
    g[100,2,3]=.8
    c=np.linspace(.99,.5,101)
    result=select_candidates(g,c,[[0,0,.8]],[[0,0,0]],[0,0,-1,1],TemporalConfig())
    assert result['selected']==100
    assert result['valid'].sum()==1


def test_service_all_sentinel_keeps_every_returned_candidate():
    # Isolate the real service method to avoid loading SAM/CUDA in this test.
    tree=ast.parse((ROOT/'tracking_service/serve.py').read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run_grasp')
    g=np.repeat(np.eye(4)[None],250,axis=0);c=np.linspace(.01,.99,250,dtype=np.float32)
    class Client:
        def __init__(self,**kwargs):pass
        def close(self):pass
        def infer_scene_depth(self,*args,**kwargs):
            return dict(instance_ids=[1],grasps=[g],confidences=[c],
                        branch_tags=[['diff']*200+['obb']*50],timing={'infer_ms':0}),0
    scope=dict(np=np,time=time,GraspClient=Client,load_sweep_params=lambda:{},
               validate_result=lambda *args:None,append_csv=lambda *args:None)
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<service method>','exec'),scope)
    snap=dict(depth=None,K=None,mask=np.ones((1,1)),frame_id='fixture',generation=1,filters={})
    service=SimpleNamespace(grasp_port=5556,out=Path('.'))
    result=scope['run_grasp'](service,snap,'graspmoe',200,-1,-1)
    assert len(result['grasps'])==250 and result['candidate_count_generated']==250
    assert result['branch_tags'].count('obb')==50
    legacy=scope['run_grasp'](service,snap,'graspmoe',200,100,-1)
    assert len(legacy['grasps'])==100
