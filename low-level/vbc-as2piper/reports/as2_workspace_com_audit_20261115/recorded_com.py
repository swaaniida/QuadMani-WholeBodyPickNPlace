"""Exact training asset COM of existing AS2 measured-force evaluation."""
from pathlib import Path
import json,hashlib
import numpy as np,mujoco
from scipy.spatial import ConvexHull
import geometry_tools_old as g
O=Path(__file__).resolve().parent;src=g.ROOT/'reports/deployment_workspace_review_20261114/contact_capture';meta=json.loads((src/'metadata.json').read_text());z=np.load(src/'states.npz');physics=np.load(src/'foot_physics.npz')['values'][1::2]
assert hashlib.sha256(g.URDF.read_bytes()).hexdigest()==meta['source_sha256']['resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf']
m,d=g.m,g.d;bid=m.body('base_link').id;qids=[int(m.joint(n).qposadr[0]) for n in meta['dof_names']]
ids=meta['video_envs'];coms=[];margins=[];fmargins=[];gaps=[];active_margins=[];alignment=[]
for t in range(len(z['time'])):
 row=[];mg=[];fg=[];gap=[];am=[]
 for j,eid in enumerate(ids):
  root=z['root'][t,j];d.qpos[:3]=root[:3];d.qpos[3:7]=root[[6,3,4,5]];d.qpos[qids]=z['q'][t,j];g.fk()
  com=d.subtree_com[bid].copy();feet=d.xpos[g.feet].copy();force=physics[t,eid,12:24].reshape(4,3);measured=physics[t,eid,:12].reshape(4,3)
  alignment.append(np.max(np.abs(feet-measured)))
  hull=ConvexHull(feet[:,:2]);mg.append(float(-np.max(hull.equations[:,:2]@com[:2]+hull.equations[:,2])))
  front=feet[:2,:2].mean(0);rear=feet[2:,:2].mean(0);axis=front-rear;axis/=np.linalg.norm(axis);fg.append(float((front-com[:2])@axis));gap.append(feet[:,2]-.028);row.append(com)
  contacts=force[:,2]>2
  if contacts.sum()>=3:
   h=ConvexHull(feet[contacts,:2]);am.append(float(-np.max(h.equations[:,:2]@com[:2]+h.equations[:,2])))
  else:am.append(float('nan'))
 coms.append(row);margins.append(mg);fmargins.append(fg);gaps.append(gap);active_margins.append(am)
coms=np.array(coms);margins=np.array(margins);fmargins=np.array(fmargins);gaps=np.array(gaps);active_margins=np.array(active_margins)
assert np.max(alignment)<.003,np.max(alignment)
result={}
mask=(z['time']>=12)&(z['time']<=60)
for j,it in enumerate(meta['iterations']):
 force=physics[mask,ids[j],12:24].reshape(-1,4,3);fz=force[:,:,2];rrup=(fz[:,3]<=2)&(gaps[mask,j,3]>.005)
 result[str(it)]=dict(samples=int(mask.sum()),RR_contact_fraction=float((fz[:,3]>2).mean()),RR_up_fraction=float(rrup.mean()),rear_total_mean_N=float(fz[:,2:].sum(-1).mean()),front_total_mean_N=float(fz[:,:2].sum(-1).mean()),front_margin_min_mm=float(fmargins[mask,j].min()*1000),front_margin_mean_mm=float(fmargins[mask,j].mean()*1000),front_margin_P5_mm=float(np.quantile(fmargins[mask,j],.05)*1000),all4_projected_support_margin_min_mm=float(margins[mask,j].min()*1000),active_contact_polygon_margin_mean_mm=float(np.nanmean(active_margins[mask,j])*1000),RR_up_while_front_margin_gt40mm_fraction=float((rrup&(fmargins[mask,j]>.04)).mean()),EE_position_error_mean_m=float(z['position_error'][mask,j].mean()))
for j,it in enumerate(meta['iterations']):
 fz=physics[mask,ids[j],12:24].reshape(-1,4,3)[:,:,2];positive=np.maximum(fz,0);positions=physics[mask,ids[j],:12].reshape(-1,4,3)[:,:,:2]
 cop=(positions*positive[:,:,None]).sum(1)/positive.sum(1)[:,None];delta=np.linalg.norm(cop-coms[mask,j,:2],axis=1)
 result[str(it)].update(measured_CoP_COM_distance_mean_mm=float(delta.mean()*1000),measured_CoP_COM_distance_P95_mm=float(np.quantile(delta,.95)*1000))
np.savez_compressed(O/'recorded_com.npz',time=z['time'],iterations=meta['iterations'],com=coms,all4_margin=margins,front_margin=fmargins,active_contact_margin=active_margins,clearance=gaps)
out=dict(source=str(src),asset_sha256=hashlib.sha256(g.URDF.read_bytes()).hexdigest(),asset_note='Exact original45k training asset, without later camera; distinct from current-URDF static scan.',max_FK_measured_foot_difference_m=float(np.max(alignment)),phase='12..60s, zero velocity EE trajectory, one display replica/checkpoint',results=result)
(O/'recorded_com.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
