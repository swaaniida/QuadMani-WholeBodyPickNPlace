from pathlib import Path
import json,hashlib,math
import numpy as np,mujoco
import geometry_tools as g
O=Path(__file__).resolve().parent;root=O.parents[2];records=[]
for mode in ['grid','corners','random']:
 data=json.loads((O/(mode+'.json')).read_text())['records']
 refined=json.loads((O/('refined_'+mode+'.json')).read_text()) if (O/('refined_'+mode+'.json')).exists() else []
 for rec in data:
  if not rec['passed']:
   match=next(x for x in refined if np.linalg.norm(np.array(x['target'])-rec['target'])<1e-10)
   rec={**rec,**match}
   g.d.qpos[:]=rec['qpos'];mujoco.mj_forward(g.m,g.d)
   rec['com_margin_m']=float(-np.max(g.hull.equations[:,:2]@g.d.subtree_com[g.m.body('base_link').id,:2]+g.hull.equations[:,2]))
   rec['passed']=rec['position_error_m']<.003 and rec['orientation_error_deg']<2 and rec['foot_error_m']<.002 and not rec['hits'] and rec['com_margin_m']>=.02
   rec['refined']=True
  assert rec['passed'],rec
  q=np.array(rec['qpos']);assert (q[g.aq]>=g.alo).all() and (q[g.aq]<=g.ahi).all()
  assert (q[g.lq]>=g.llo).all() and (q[g.lq]<=g.lhi).all()
  rec['mode']=mode;records.append(rec)
protected=json.loads((O/'preserved_before.json').read_text())
for name,digest in protected['sources'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
for name,st in protected['checkpoints'].items():
 s=(root/name).stat();assert s.st_size==st['size'] and s.st_mtime_ns==st['mtime'],name
s=dict(total=len(records),passed=sum(x['passed'] for x in records),refined=sum(x.get('refined',False) for x in records),
 position_error_max_mm=max(x['position_error_m'] for x in records)*1000,
 orientation_error_max_deg=max(x['orientation_error_deg'] for x in records),
 foot_error_max_mm=max(x['foot_error_m'] for x in records)*1000,
 com_margin_min_mm=min(x['com_margin_m'] for x in records)*1000,
 protected_sources=len(protected['sources']),protected_checkpoints=len(protected['checkpoints']),
 geometry=dict(center=[.28,0,.5],radius=[.35,.65],elevation_rad=[-math.asin(.48/.65),math.asin(.5/.65)],yaw_rad=[-.3,.3],
 max_forward_x=.93,floor_z=.02,floor_x_at_yaw0=.28+math.sqrt(.65**2-.48**2),upper_z=1.,upper_x_at_yaw0=.28+math.sqrt(.65**2-.5**2)),
 limitations=['Finite static flat-ground witnesses, not an analytic reachability proof.','No trajectory/policy/rough-terrain/torque/hardware certification.','Whole-body IK may require different body postures for different goals.'])
(O/'summary.json').write_text(json.dumps(s,indent=2));(O/'verified_poses.json').write_text(json.dumps(records,indent=2));print(json.dumps(s,indent=2))
