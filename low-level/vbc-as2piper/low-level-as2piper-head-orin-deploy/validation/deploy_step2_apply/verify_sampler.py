"""Graph, orientation, asset-binding and vectorized sampling checks. No learning."""
from pathlib import Path
import importlib.util,json,hashlib,tempfile,time
import numpy as np
import torch
O=Path(__file__).resolve().parent;R=O.parents[1]
spec=importlib.util.spec_from_file_location('workspace',R/'legged_gym/utils/feasible_workspace.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
filename=R/'resources/robots/as2piper/workspace/head_orin_feasible.json';b=json.loads(filename.read_text())
torch.manual_seed(123);torch.set_num_threads(1)
def make(path=filename):return mod.FeasibleWorkspaceSampler(path,R,6144,'cpu',b['center'],b['home_sphere'],b['orientation']['arm_induced_pitch'])
s=make();ids=torch.arange(6144);adj=np.zeros((len(b['nodes']),)*2,dtype=bool)
for edge in b['edges']:a,c=edge['source'],edge['target'];adj[a,c]=adj[c,a]=True
seen=set();start=time.time()
for _ in range(200):
 previous=s.current.clone();q=s.sample(ids)
 assert adj[previous.numpy(),s.current.numpy()].all()
 assert torch.equal(q,s.spheres[s.current])
 seen.update(s.current.tolist())
assert len(seen)==len(b['nodes'])
untouched=s.current[1::2].clone();s.reset(ids[::2]);assert (s.current[::2]==0).all();assert torch.equal(s.current[1::2],untouched)
# Verify all 25 point orientation evaluations against the offline formula.
maxerr=0.
for edge in b['edges']:
 a=np.array(b['nodes'][edge['source']]['sphere']);c=np.array(b['nodes'][edge['target']]['sphere'])
 q=np.linspace(a,c,25);r,phi=q[:,:2].T;z=b['center'][2]+r*np.sin(phi);x=b['center'][0]+r*np.cos(phi)
 mix=np.clip((.25-z)/.15,0,1);expected=(1-mix)*(2.455216137744035-phi)+mix*np.deg2rad(175-25*np.clip((x-.55)/.425,0,1))
 actual=s.pitch(torch.tensor(q,dtype=torch.float64)).numpy();maxerr=max(maxerr,float(np.max(abs(actual-expected))))
assert maxerr<1e-12
# An asset mismatch and disconnected graph must fail loudly; no fallback sampling.
for mode in ['asset','graph']:
 bad=json.loads(filename.read_text())
 if mode=='asset':bad['asset_sha256'][next(iter(bad['asset_sha256']))]='0'*64
 else:bad['edges']=[]
 with tempfile.TemporaryDirectory(dir=O) as tmp:
  path=Path(tmp)/'bad.json';path.write_text(json.dumps(bad))
  try:make(path)
  except ValueError:pass
  else:raise AssertionError(mode)
# The graph contains floor and upper goals, not only the initial narrow range.
zs=[n['target'][2] for n in b['nodes'] if not n['home']]
assert min(zs)==.02 and max(zs)==.85
result=dict(passed=True,nodes=len(b['nodes']),undirected_edges=len(b['edges']),directed_edges=2*len(b['edges']),
 draws=200*6144,all_nodes_reached=True,partial_reset_pass=True,stale_asset_rejected=True,disconnected_bank_rejected=True,
 orientation_max_error_rad=maxerr,goal_ground_z_min_m=min(zs),goal_ground_z_max_m=max(zs),seconds=time.time()-start,optimizer_updates=0)
(O/'sampler_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
