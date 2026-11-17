"""Read-only production audit and offline candidate reward probes. No training."""
from pathlib import Path
import json,hashlib,ast
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[2];DEPLOY=O.parents[1]
production=[DEPLOY/'legged_gym/envs/rewards/maniploco_rewards.py',DEPLOY/'legged_gym/envs/manip_loco/manip_loco.py',DEPLOY/'legged_gym/envs/manip_loco/as2_head_orin_config.py',DEPLOY/'resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf']
hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in production}
(O/'source_before.json').write_text(json.dumps(hashes,indent=2))
z=np.load(R/'reports/deployment_workspace_review_20261114/contact_capture/foot_physics.npz');a=z['values'];dt=float(z['dt']);window=round(.5/dt)
# Candidate formula probes ONLY. These are not added to the reward container.
def rolling_mean(x):
 c=np.concatenate([np.zeros_like(x[:1]),np.cumsum(x,axis=0)],axis=0)
 end=np.arange(1,len(x)+1);start=np.maximum(0,end-window)
 return (c[end]-c[start])/(end-start).reshape((-1,)+(1,)*(x.ndim-1))
def rolling_min(x):
 padded=np.concatenate([np.repeat(x[:1],window-1,axis=0),x],axis=0)
 return np.lib.stride_tricks.sliding_window_view(padded,window,axis=0).min(axis=-1)
def costs(gap,force):
 normal=force[...,2]>2.
 valid=normal|np.concatenate([normal[:1],normal[:-1]],axis=0)
 duty=rolling_mean(valid.astype(float))
 contact_cost=np.clip((.6-duty)/.6,0,1).mean(axis=-1)
 sustained=rolling_min(np.maximum(gap,0))
 proximity_cost=np.minimum(np.maximum((sustained-.005)/.020,0)**2,1).mean(axis=-1)
 return proximity_cost,contact_cost,duty
feet=a[...,:12].reshape(*a.shape[:2],4,3);force=a[...,12:24].reshape(*a.shape[:2],4,3)
gap=feet[...,2]-.028;proximity,contact,duty=costs(gap,force);time=(np.arange(len(a))+1)*dt;mask=(time>=12)&(time<=60)
results={}
for j,it in enumerate(z['iterations']):
 s=slice(j*8,(j+1)*8);h=gap[mask,s];f=force[mask,s];p=proximity[mask,s];c=contact[mask,s]
 results[str(it)]=dict(RR_gap_mean_mm=float(h[...,3].mean()*1000),RR_gap_p95_mm=float(np.quantile(h[...,3],.95)*1000),RR_force_norm_contact_fraction=float((np.linalg.norm(f[...,3,:],axis=-1)>2).mean()),RR_positive_Fz_contact_fraction=float((f[...,3,2]>2).mean()),ground_proximity_cost_mean=float(p.mean()),valid_contact_cost_mean=float(c.mean()),scale_candidates={str(scale):dict(total_extra_before_div100=float((-scale*(p+c)).mean()),total_extra_after_div100=float((-scale*(p+c)).mean()/100)) for scale in [.25,.5,1.]})
# Force-pulse probes show continuous unloaded-leg punishment and brief step tolerance.
synthetic={};T=150
for name in ['four_feet','persistent_RR_lift','brief_120ms_RR_step','long_400ms_RR_step','repeated_short_taps']:
 h=np.zeros((T,1,4));f=np.zeros((T,1,4,3));f[...,2]=50
 if name=='persistent_RR_lift':h[:,0,3]=.030;f[:,0,3]=0
 if name=='brief_120ms_RR_step':h[50:56,0,3]=.08;f[50:56,0,3]=0
 if name=='long_400ms_RR_step':h[50:70,0,3]=.08;f[50:70,0,3]=0
 if name=='repeated_short_taps':
  for t in range(T):
   if t%10!=0:h[t,0,3]=.03;f[t,0,3]=0
 p,c,d=costs(h,f)
 synthetic[name]=dict(ground_cost_mean=float(p[25:].mean()),contact_cost_mean=float(c[25:].mean()),extra_reward_scale_0_5_mean=float((-.5*(p+c))[25:].mean()),extra_reward_scale_0_5_min=float((-.5*(p+c))[25:].min()))
assert synthetic['four_feet']['extra_reward_scale_0_5_mean']==0
assert synthetic['brief_120ms_RR_step']['extra_reward_scale_0_5_mean']==0
assert synthetic['persistent_RR_lift']['extra_reward_scale_0_5_mean']<synthetic['long_400ms_RR_step']['extra_reward_scale_0_5_mean']<0
assert synthetic['repeated_short_taps']['extra_reward_scale_0_5_mean']<0
# Reproduce the production rough-ground function on an exact 30-degree plane.
# Its min-of-three grid rule underestimates local height without interpolation.
import torch
from types import SimpleNamespace
source=DEPLOY/'legged_gym/envs/manip_loco/manip_loco.py';tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ManipLoco');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_reward_foot_ground_heights');namespace={'torch':torch}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
terrain=SimpleNamespace(mesh_type='trimesh',border_size=0.,horizontal_scale=.05,vertical_scale=.005)
# 5mm-quantized grid, foot at 90% into a 5cm cell.
x=.145;slope=np.tan(np.deg2rad(30));height=np.round(np.arange(10)[:,None]*.05*slope/.005)*np.ones((1,10))
state=torch.zeros(1,4,13);state[:,:,0]=x;state[:,:,1]=.145;actual_surface=x*slope
state[:,:,2]=actual_surface+.028/np.cos(np.deg2rad(30))
e=SimpleNamespace(cfg=SimpleNamespace(terrain=terrain),rigid_body_state=state,feet_indices=torch.arange(4),height_samples=torch.tensor(height))
ground=namespace['_reward_foot_ground_heights'](e);reported=state[:,:,2]-ground-.028
rough=dict(slope_deg=30.,cell_m=.05,grid_height_quantization_m=.005,foot_xy_m=[x,.145],actual_surface_m=float(actual_surface),reported_ground_m=float(ground[0,0]),actual_sphere_surface_clearance_m=0.,naive_reported_sphere_bottom_clearance_mm=float(reported[0,0]*1000),conclusion='Existing min-of-three grid helper is unsuitable as a 5mm proximity classifier on rough/slopes; actual contact/mesh ground reference is required. This is a geometric probe, not a PhysX rollout.')
legref=json.loads((R/'reports/checkpoint_selection_20261114/evaluations/45000_flat/result.json').read_text())['metrics']['stand']
result=dict(status='inspection_only_pending_user_review',window_s=.5,min_contact_duty=.6,normal_force_N=2.,clearance_deadband_m=.005,clearance_scale_m=.020,measured_flat_replay=results,synthetic_reward_probes=synthetic,rough_ground_probe=rough,reference_45k_in_distribution_stand=dict(stand_still_weighted=legref['RewardWeighted/leg/stand_still'],leg_total_before_div100=legref['RewardTotal/leg_before_div100']),proposed_scale_each=-.5,production_modified=False,optimizer_updates=0,notes=['Replay data use exact old no-camera training URDF; not a new-policy/current-camera evaluation.','No causal learning improvement can be inferred from offline reward relabeling.','No proposed gate/scale is applied. Rough-ground reference, reset/warmup and push recovery exclusion must be finalized before implementation.'])
(O/'analysis.json').write_text(json.dumps(result,indent=2))
for path,sha in hashes.items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==sha
print(json.dumps(result,indent=2))
