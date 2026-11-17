"""Offline static witness graph. No training-time IK or policy updates."""
from pathlib import Path
import json, hashlib, time, collections
import numpy as np
import mujoco
import geometry_tools as g
O=Path(__file__).resolve().parent
# Reuse the inspected model, support wrench and orientation convention.
namespace={'__name__':'workspace_checks','__file__':str(O/'inspect_workspace.py')}
exec((O/'inspect_workspace.py').read_text().split('rows=[];start=time.time()')[0],namespace)
gravity_support=namespace['gravity_support'];center=namespace['center'];baseid=namespace['baseid']
inspection=json.loads((O/'candidate_inspection.json').read_text())
assert inspection['urdf_sha256']==hashlib.sha256(g.URDF.read_bytes()).hexdigest()
home=np.array([.427898468676838,.421686284147847,0.])
assert g.leg_ik(.3244069024246581,0)
g.d.qpos[g.aq]=g.neutral;g.fk()
nodes=[dict(sphere=home.tolist(),body_z=.3244069024246581,body_pitch_deg=0.,leg_q=g.d.qpos[g.lq].tolist(),arm_q=g.neutral.tolist(),home=True)]
for row in inspection['samples']:
 if not row['path_pass']: continue
 w=row['witness']
 nodes.append(dict(sphere=[row['r_m'],row['phi_rad'],row['yaw_rad']],body_z=w['body_z'],body_pitch_deg=w['body_pitch_deg'],leg_q=w['leg_q'],arm_q=w['arm_q'],target=row['target'],home=False))

def pose(sphere):
 r,phi,yaw=sphere
 target=center+r*np.array([np.cos(phi)*np.cos(yaw),np.cos(phi)*np.sin(yaw),np.sin(phi)])
 x=center[0]+np.linalg.norm((target-center)[:2]);z=target[2]
 blend=np.clip((.25-z)/.15,0,1)
 pitch=(1-blend)*(2.455216137744035-phi)+blend*np.deg2rad(175-25*np.clip((x-.55)/.425,0,1))
 return target,pitch,yaw

def check(a,b):
 previous=np.array(a['arm_q']);samples=[];margin_min=1.;max_tau=np.zeros(6)
 for t in np.linspace(0,1,25):
  h=(1-t)*a['body_z']+t*b['body_z'];theta=(1-t)*a['body_pitch_deg']+t*b['body_pitch_deg']
  if not g.leg_ik(h,np.deg2rad(theta)):return None,'leg_ik'
  sphere=(1-t)*np.array(a['sphere'])+t*np.array(b['sphere']);target,pitch,yaw=pose(sphere)
  ok,pe,ae=g.arm_ik(target,pitch,[previous],yaw)
  if not ok:return None,'arm_ik'
  q=g.d.qpos[g.aq].copy()
  if np.max(abs(q-previous))>.5:return None,'joint_branch_jump'
  if t==1 and np.max(abs(q-np.array(b['arm_q'])))>.15:return None,'endpoint_branch'
  previous=q
  if g.collisions():return None,'collision'
  com=g.d.subtree_com[baseid];margin=float(-np.max(g.hull.equations[:,:2]@com[:2]+g.hull.equations[:,2]));margin_min=min(margin_min,margin)
  if margin<.02:return None,'com_margin'
  torque=gravity_support();max_tau=np.maximum(max_tau,np.abs(torque['arm_gravity_tau_Nm']))
  if max(torque['max_leg_urdf_effort_ratio'],torque['max_arm_urdf_effort_ratio'])>1 or torque['force_residual_N']>.5 or torque['moment_residual_Nm']>.2:return None,'static_support_torque'
  samples.append(dict(t=float(t),body_z=float(h),body_pitch_deg=float(theta),leg_q=g.d.qpos[g.lq].tolist(),arm_q=q.tolist(),**torque))
 return dict(samples=samples,min_com_margin_m=margin_min,max_arm_gravity_tau_Nm=max_tau.tolist()),None

edges=[];fail=collections.Counter();start=time.time()
for i,a in enumerate(nodes):
 for j in range(i+1,len(nodes)):
  path,reason=check(a,nodes[j])
  if path:edges.append(dict(source=i,target=j,**path))
  else:fail[reason]+=1
 print('row',i,'/',len(nodes),'edges',len(edges),'seconds',round(time.time()-start,1),flush=True)
 (O/'graph_progress.json').write_text(json.dumps(dict(row=i,edges=len(edges),failed=dict(fail))))
# Only retain the connected component reachable from the reset/home state.
adj=[[] for _ in nodes]
for e in edges:adj[e['source']].append(e['target']);adj[e['target']].append(e['source'])
seen={0};todo=[0]
while todo:
 for v in adj[todo.pop()]:
  if v not in seen:seen.add(v);todo.append(v)
selected=sorted(seen);mapping={old:new for new,old in enumerate(selected)}
kept=[dict(e,source=mapping[e['source']],target=mapping[e['target']]) for e in edges if e['source'] in seen and e['target'] in seen]
assets={str(g.URDF.relative_to(O.parents[1])):hashlib.sha256(g.URDF.read_bytes()).hexdigest()}
for mesh in g.c.urdf.findall('.//mesh') if hasattr(g.c,'urdf') else []:pass
# Bind every referenced mesh as well as URDF; a mesh-only edit must invalidate the bank.
import xml.etree.ElementTree as ET
for mesh in ET.parse(g.URDF).findall('.//mesh'):
 p=(g.URDF.parent/mesh.get('filename')).resolve()
 assets[str(p.relative_to(O.parents[1].resolve()))]=hashlib.sha256(p.read_bytes()).hexdigest()
bank=dict(schema_version=1,asset_sha256=assets,center=center.tolist(),home_sphere=home.tolist(),
 orientation=dict(arm_induced_pitch=2.455216137744035,roll=0.,floor_x=[.55,.975],floor_pitch_deg=[175.,150.],blend_z=[.10,.25]),
 nodes=[nodes[i] for i in selected],edges=[dict(source=e['source'],target=e['target']) for e in kept],
 scope='Sampled nominal flat-ground static feasibility witnesses. No continuous collision, dynamic stability, payload, terrain or hardware guarantee.',
 constraints=inspection['assumptions'],path_samples=25,hardware_torque_verified=False)
assert len(kept)>0 and len(bank['nodes'])>1
(O/'feasible_workspace_bank.json').write_text(json.dumps(bank,indent=2))
(O/'transition_witnesses.json').write_text(json.dumps(dict(edges=kept,failed=dict(fail),excluded_nodes=sorted(set(range(len(nodes)))-seen)),indent=2))
print('DONE',len(bank['nodes']),len(kept),'seconds',round(time.time()-start,1),flush=True)
