"""Offline summaries and candidate reward values; no settings or weights changed."""
from pathlib import Path
import numpy as np,json,csv,math
out=Path(__file__).resolve().parent;project=out.parents[2];old=project/'low-level-as2piper-head-orin';previous=project/'reports/b1_contact_audit_20261111'
ratio=(.212+.21344)/(.35+.35);candidates=[.15,.18,.30]
records=[];eventrows=[];candidate_rows=[]
def stats(v):
 return {'count':int(v.size),**({q:float(np.quantile(v,p)) for q,p in [('p10',.1),('p50',.5),('p90',.9),('p95',.95),('p99',.99),('max',1)]} if v.size else {})}
for terrain in ['flat','rough']:
 x=np.load(out/'b1_swing'/(terrain+'_normal')/'samples.npz');h=x['foot_heights'][100:];age=x['episode_age'][100:];sensor=np.linalg.norm(x['sensors'][100:],axis=-1);net=np.linalg.norm(x['forces'][100:],axis=-1);clock=x['clock_targets'][100:]
 for i,condition in enumerate(['stand','forward','turn','forward_turn']):
  hh=h[:,i*8:(i+1)*8];ss=sensor[:,i*8:(i+1)*8];nn=net[:,i*8:(i+1)*8];cc=clock[:,i*8:(i+1)*8];aa=age[:,i*8:(i+1)*8]
  masks={'sensor_swing':ss<=1.5,'net_swing':nn<=2,'agreed_swing':(ss<=1.5)&(nn<=2),'clock_swing':cc<.5}
  for label,mask in masks.items():
   valid=mask&(aa[:,:,None]>100)
   records.append({'terrain':terrain,'condition':condition,'selection':label,'foot_center_above_ground_m':stats(hh[...,2][valid]),'bottom_clearance_m':stats(hh[...,3][valid])})
  # Complete no-contact events, not sample-weighted values; reject reset crossings.
  for maskname in ['sensor_swing','agreed_swing']:
   for env in range(8):
    for foot in range(4):
     mask=masks[maskname][:,env,foot]&(aa[:,env]>100)
     edge=np.diff(np.pad(mask.astype(int),(1,1)));starts=np.where(edge==1)[0];ends=np.where(edge==-1)[0]
     for start,end in zip(starts,ends):
      if start==0 or end==len(mask) or end-start<2 or np.any(np.diff(aa[start:end,env])!=1):continue
      eventrows.append({'terrain':terrain,'condition':condition,'selection':maskname,'foot':['FL','FR','RL','RR'][foot],'peak_bottom_clearance_m':float(hh[start:end,env,foot,3].max()),'duration_s':float((end-start)*.02)})
  if condition!='stand':
   # Kinematic similarity projection, not a dynamic AS2 rollout.
   projected=.028+ratio*(hh[...,2]-.04)
   pair=np.linalg.norm(projected[:,:,:2],axis=-1);realpair=np.linalg.norm(hh[:,:,:2,2],axis=-1);valid=aa>100
   for target in candidates:candidate_rows.append({'source':'B1 geometrically projected to AS2','terrain':terrain,'condition':condition,'target_m':target,'mean_raw_hinge':float(np.minimum(pair[valid]-target,0).mean()),'target_reached_fraction':float((pair[valid]>=target).mean()),'projected_pair_norm_p95_m':float(np.quantile(pair[valid],.95)),'B1_pair_norm_p95_m':float(np.quantile(realpair[valid],.95))})
# Re-score already measured AS2 PD rollout states, preserving the actual reward expression.
for terrain in ['plane','trimesh']:
 x=np.load(old/'validation/reward_review/after'/terrain/'samples.npz')['physical'][100:]
 for i,condition in enumerate(['stand','forward','turn','forward_turn']):
  v=x[:,i*4:(i+1)*4];pair=np.linalg.norm(v[:,:,:2]-v[:,:,4:6],axis=-1)
  for target in candidates:candidate_rows.append({'source':'actual AS2 zero-action PD states; NOT trained gait','terrain':terrain,'condition':condition,'target_m':target,'mean_raw_hinge':0. if condition=='stand' else float(np.minimum(pair-target,0).mean()),'target_reached_fraction':float((pair>=target).mean()),'projected_pair_norm_p95_m':float(np.quantile(pair,.95)),'B1_pair_norm_p95_m':None})
(out/'swing_height_distributions.json').write_text(json.dumps({'B1_foot_collision_radius_m':.04,'AS2_radius_m':.028,'leg_length_ratio':ratio,'records':records},indent=2)+'\n')
with (out/'swing_events.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(eventrows[0]));writer.writeheader();writer.writerows(eventrows)
with (out/'feet_height_candidates.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(candidate_rows[0]));writer.writeheader();writer.writerows(candidate_rows)
# Force recommendation from original B1 evaluation, gated as actual penalty (>2s).
x=np.load(previous/'flat_normal/samples.npz');force=np.linalg.norm(x['forces'][100:],axis=-1);age=x['episode_age'][100:]
b1=json.loads((previous/'flat_normal/result.json').read_text());hardware=json.loads((old/'validation/hardware.json').read_text());static_b1=np.array(b1['conditions']['stand']['net_contact_force_N']['mean_per_foot']);static_as2=np.array(hardware['static_foot_load_N'])
force_rows=[]
for i,condition in enumerate(['stand','forward','turn','forward_turn']):
 v=force[:,i*8:(i+1)*8][age[:,i*8:(i+1)*8]>100]
 p95=np.quantile(v,.95,axis=0);p99=np.quantile(v,.99,axis=0)
 force_rows.append({'condition':condition,'B1_steady_p95_N':p95.tolist(),'B1_steady_p99_N':p99.tolist(),'AS2_static_load_normalized_p95_N':(p95/static_b1*static_as2).tolist(),'AS2_static_load_normalized_p99_N':(p99/static_b1*static_as2).tolist(),'B1_pooled_p99_N':float(np.quantile(v,.99))})
force_report={'source':'original B1 45000 normal policy flat; episode age>2s','B1_mass_kg':57.98696983,'AS2_mass_kg':hardware['mass_kg'],'mass_ratio':hardware['mass_kg']/57.98696983,'B1_stand_measured_mean_per_foot_N':static_b1.tolist(),'AS2_nominal_static_foot_load_N':static_as2.tolist(),'rows':force_rows,'recommended_single_threshold_candidate_N':200,'sensitivity_candidates_N':[180,200,220],'meaning':'reward threshold candidate, not certified hardware force limit; no config change'}
(out/'force_threshold_recommendation.json').write_text(json.dumps(force_report,indent=2)+'\n')
print('Height, swing events, AS2 candidate rescoring and force recommendations recorded')
for condition in ['forward','turn','forward_turn']:
 v=next(x for x in records if x['terrain']=='flat' and x['condition']==condition and x['selection']=='agreed_swing')
 events=np.array([x['peak_bottom_clearance_m'] for x in eventrows if x['terrain']=='flat' and x['condition']==condition and x['selection']=='agreed_swing'])
 print(condition,'sample clearance',v['bottom_clearance_m'],'eventpeak',stats(events))
