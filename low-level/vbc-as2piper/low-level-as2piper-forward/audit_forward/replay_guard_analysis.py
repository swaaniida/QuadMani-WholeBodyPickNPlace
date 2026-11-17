"""Check whether rejection at t=0 can be fixed by destination-only retries."""
import importlib,json,sys,types
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent.parent
pkg=types.ModuleType('guard_utils');pkg.__path__=[str(ROOT/'legged_gym/utils')];sys.modules['guard_utils']=pkg
Guard=importlib.import_module('guard_utils.arm_collision').ArmCollisionGuard


def main():
    out=ROOT/'audit_forward/ee_fixed_base_20261029'
    a=np.load(str(out/'tracking.npz'))['samples'];d=np.load(str(out/'trajectories.npz'))
    candidates=[]
    for i,step in enumerate(d['step']):
        if step<200:continue
        for env in range(4):
            if a[int(step),env,2]>.5:candidates.append((i,env))
    rng=np.random.default_rng(20261029);chosen=rng.choice(len(candidates),min(1000,len(candidates)),replace=False)
    q=[];target=[];orn=[]
    for j in chosen:
        i,env=candidates[j];root=d['root'][i,env];body=Rotation.from_quat(root[3:7]).as_matrix()
        origin=root[:3]+body@np.array([.005,0,.08202847821788371])
        local=body.T@(d['target'][i,env]-origin)
        pitch=np.arctan2(local[2],np.linalg.norm(local[:2]));yaw=np.arctan2(local[1],local[0]);delta=rng.uniform(-.15,.15,3)
        R=Rotation.from_euler('xyz',[delta[0],2.455216137744035-pitch+delta[1],yaw+delta[2]]).as_matrix()
        q.append(d['q'][i,env,12:18]);target.append(local);orn.append(R)
    q=torch.tensor(np.array(q),dtype=torch.float);p=torch.tensor(np.array(target),dtype=torch.float);R=torch.tensor(np.array(orn),dtype=torch.float)
    asset=ROOT/'resources/robots/as2piper/urdf/as2piper_plate_forward_training.urdf'
    guard=Guard(asset,ROOT/'resources/commands/arm_collision.json','cpu');offset=torch.tensor([.005,0,.0820284782])
    q12,fail12=guard.solve(p,R,q);body12=guard.collides(q12,offset);reject12=fail12|body12
    q64,fail64=guard.solve(p,R,q,iterations=64);body64=guard.collides(q64,offset);reject64=fail64|body64
    results=dict(snapshot_count=len(q),source='Fixed-base actual hold states; previous goal position held fixed; new orientation delta independently sampled',
        start_ik_reject_12=int(fail12.sum()),start_body_reject_12=int(body12.sum()),start_reject_12=int(reject12.sum()),
        start_ik_reject_64=int(fail64.sum()),start_body_reject_64=int(body64.sum()),start_reject_64=int(reject64.sum()),
        rejected_12_but_passed_64=int((reject12&~reject64).sum()),
        consequence='At t=0, destination position cancels out of torch.lerp. Destination-only retries cannot change this failure. Orientation delta is held across all ten original B1 retries.',
        limitation='Conservative boxes and iterative IK; rejection is not a proof of physical impossibility. Snapshots are not an exact resampling-event distribution.')
    (out/'guard_start_analysis.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))


if __name__=='__main__':main()
