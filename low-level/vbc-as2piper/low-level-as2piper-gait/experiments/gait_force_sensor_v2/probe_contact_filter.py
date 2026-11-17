"""Offline sphere/actual-triangle distance validates reported contact filtering."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import isaacgym
from isaacgym import gymapi,gymutil,gymtorch
import torch,numpy as np
from diagnostic_loader import load
AS2PiPERGaitStudy,AS2PiPERGaitStudyCfg,AS2PiPERGaitStudyCfgPPO,configure=load()
from legged_gym.utils.helpers import class_to_dict,set_seed
from legged_gym.utils.hw_runner import HardwareRunner

def distances(points,triangles):
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2];ab=b-a;ac=c-a;n=np.cross(ab,ac);nn=(n*n).sum(-1)
    good=nn>1e-16;a,b,c,ab,ac,n,nn=[z[good] for z in [a,b,c,ab,ac,n,nn]]
    aa=(ab*ab).sum(-1);bb=(ab*ac).sum(-1);cc=(ac*ac).sum(-1);denom=aa*cc-bb*bb
    result=[]
    for offset in range(0,len(points),32):
        p=points[offset:offset+32,None,:];ap=p-a
        d0=(ap*ab).sum(-1);d1=(ap*ac).sum(-1)
        u=(cc*d0-bb*d1)/np.maximum(denom,1e-20);v=(aa*d1-bb*d0)/np.maximum(denom,1e-20)
        inside=(u>=0)&(v>=0)&(u+v<=1)
        d=np.where(inside,(ap*n).sum(-1)**2/nn,np.inf)
        for first,last in [(a,b),(b,c),(c,a)]:
            edge=last-first;t=np.clip(((p-first)*edge).sum(-1)/np.maximum((edge*edge).sum(-1),1e-20),0,1)
            delta=p-(first+t[...,None]*edge);d=np.minimum(d,(delta*delta).sum(-1))
        result.append(np.sqrt(d.min(-1)))
    return np.concatenate(result)

def metrics(predicted,reference,valid):
    p=predicted[valid].reshape(-1);r=reference[valid].reshape(-1)
    return dict(precision=float((p&r).sum()/max(1,p.sum())),recall=float((p&r).sum()/max(1,r.sum())),agreement=float((p==r).mean()),positive_fraction=float(p.mean()))

p=argparse.ArgumentParser();p.add_argument('--terrain',choices=['plane','trimesh'],required=True);p.add_argument('--vx',type=float,default=.25);p.add_argument('--output',required=True);a=p.parse_args()
set_seed(20261104);cfg=configure(AS2PiPERGaitStudyCfg(),'stable_filter',controlled=True)
cfg.terrain.mesh_type=a.terrain;cfg.terrain.measure_heights=a.terrain=='trimesh';cfg.env.num_envs=4
cfg.init_state.rand_yaw_range=0.;cfg.init_state.origin_perturb_range=0.;cfg.init_state.init_vel_perturb_range=0.
params=gymapi.SimParams();params.use_gpu_pipeline=True;params.physx.use_gpu=True;gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
e=AS2PiPERGaitStudy(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True)
def reset(ids):
 e.dof_pos[ids]=e.default_dof_pos;e.dof_vel[ids]=0;e.gym.set_dof_state_tensor(e.sim,gymtorch.unwrap_tensor(e.dof_state))
e._reset_dofs=reset
def commands(ids):
 e.commands[ids]=0.;e.commands[ids,0]=a.vx
e._resample_commands=commands
r=HardwareRunner(e,class_to_dict(AS2PiPERGaitStudyCfgPPO()),None,'cuda:0')
r.load(str(ROOT/'logs/as2piper-low/gait_stable_fresh_s7_v1/model_200.pt'));set_seed(20261104);commands(torch.arange(4,device=e.device));e.compute_observations()
records=[]
with torch.no_grad():
 for step in range(250):
  e.goal_timer.zero_();action=r.alg.actor_critic.act_inference(e.obs_buf,hist_encoding=True);e.step(action)
  records.append(dict(feet=e.rigid_body_state[:,e.feet_indices,:3].cpu().numpy(),net=e.contact_forces[:,e.feet_indices].cpu().numpy(),peak=e.foot_contact_force_peak.cpu().numpy(),observed=e.foot_contact_observed.cpu().numpy(),sensor_old=(e.force_sensor_tensor.norm(dim=-1)>1.5).cpu().numpy(),done=e.reset_buf.cpu().numpy()))
arrays={k:np.stack([row[k] for row in records]) for k in records[0]};gap=np.empty((250,4,4),dtype=np.float32)
if a.terrain=='plane':gap=arrays['feet'][:,:,:,2]-.028
else:
 vertices=e.terrain.vertices.copy();vertices[:,:2]-=e.terrain.cfg.border_size;all_triangles=e.terrain.triangles
 for env in range(4):
  points=arrays['feet'][:,env].reshape(-1,3);lo=points[:,:2].min(0)-.05;hi=points[:,:2].max(0)+.05;near=[]
  for offset in range(0,len(all_triangles),200000):
   tri=vertices[all_triangles[offset:offset+200000]];mask=((tri[:,:,:2].min(1)<=hi)&(tri[:,:,:2].max(1)>=lo)).all(-1)
   if mask.any():near.append(tri[mask])
  local=np.concatenate(near);gap[:,env]=distances(points,local).reshape(250,4)-.028
reference=gap<.002;valid=np.ones((250,4),dtype=bool);valid[:100]=False
valid&=~arrays['done'];valid[1:]&=~arrays['done'][:-1]
report=dict(terrain=a.terrain,vx=a.vx,reference='Offline sphere/actual simulator triangle mesh distance gap<2mm; geometric near-contact, not a normal force measurement.',raw=metrics(arrays['net'][:,:,:,2]>5,reference,valid),substep_peak=metrics(arrays['peak'][:,:,:,2]>5,reference,valid),filtered=metrics(arrays['observed'],reference,valid),legacy_sensor=metrics(arrays['sensor_old'],reference,valid),valid_frames=int(valid.sum()))
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'contacts.npz',**arrays,gap=gap,valid=valid);(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));e.gym.destroy_sim(e.sim)
