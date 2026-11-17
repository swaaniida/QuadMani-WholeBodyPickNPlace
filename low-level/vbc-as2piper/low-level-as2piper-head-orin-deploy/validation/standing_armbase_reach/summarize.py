from pathlib import Path
import json,math,hashlib,numpy as np
O=Path(__file__).resolve().parent;root=O.parents[2]
def key(r):return tuple(round(float(x),10) for x in [r['r'],r['phi'],r['yaw'],*r['delta']])
steps={key(x):x for x in json.loads((O/'step_refined.json').read_text())}
records=[];groups={}
for mode in ['grid','random','corners']:
 base=json.loads((O/(mode+'.json')).read_text())['records'];refined={key(x):x for x in json.loads((O/('refine_'+mode+'_relaxed.json')).read_text())['records']}
 fixed=0;step=0
 for rec in base:
  k=key(rec)
  if k in refined:rec=refined[k]
  rec={**rec,'fixed_feet_pass':rec['passed'],'mode':mode}
  fixed+=int(rec['passed'])
  if not rec['passed'] and k in steps:rec={**steps[k],'fixed_feet_pass':False,'mode':mode}
  step+=int(rec['passed']);records.append(rec)
 groups[mode]=dict(total=len(base),fixed_feet_pass=fixed,with_foot_adjustment_pass=step)
protected=json.loads((root/'low-level-as2piper-head-orin-deploy/validation/spherical_reach/preserved_before.json').read_text())
for name,digest in protected['sources'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
for name,st in protected['checkpoints'].items():
 s=(root/name).stat();assert (s.st_size,s.st_mtime_ns)==(st['size'],st['mtime']),name
summary=dict(center=[.22435,0,.4563069024246581],candidate_radius=[.35,.8],candidate_yaw=[-1.2,1.2],candidate_elevation=[-math.asin((.4563069024246581-.02)/.8),math.asin((1-.4563069024246581)/.8)],total=len(records),fixed_feet_pass=sum(x['fixed_feet_pass'] for x in records),with_foot_adjustment_pass=sum(x['passed'] for x in records),groups=groups,unresolved=[{k:v for k,v in x.items() if k!='qpos'} for x in records if not x['passed']],position_only=json.loads((O/'position_only.json').read_text()),preserved_sources=len(protected['sources']),preserved_checkpoints=len(protected['checkpoints']),current_config_note='Only standing arm-base center, elevation limits and initial spherical coordinates applied. Radius remains .35..65 and yaw +/-.30. Expansion .80/+/-1.2 is an offline candidate, not applied.',current_sanity_pass=sum(x['passed'] for x in json.loads((O/'current.json').read_text())['records']),training_started=False)
(O/'summary.json').write_text(json.dumps(summary,indent=2));(O/'verified_poses.json').write_text(json.dumps(records,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['unresolved','position_only']},indent=2))
# Fix metadata of the narrower applied-range sanity sample.
p=O/'current.json';a=json.loads(p.read_text());a.update(radius=[.35,.65],elevation=[-math.asin((.4563069024246581-.02)/.65),math.asin((1-.4563069024246581)/.65)],yaw=[-.3,.3]);p.write_text(json.dumps(a,indent=2))
