"""Actual sampler + geometric resampling on CPU; no policy/optimizer."""
from pathlib import Path
import ast,math,json,types,difflib,hashlib
import torch
O=Path(__file__).resolve().parent;R=O.parents[1];root=R.parent
torch.set_num_threads(1);torch.manual_seed(20261115)
cfgpath=R/'legged_gym/envs/manip_loco/as2_head_orin_config.py';source=R/'legged_gym/envs/manip_loco/manip_loco.py'
def config(text):
 c=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='AS2HeadOrinCfg')
 g=next(n for n in c.body if isinstance(n,ast.ClassDef) and n.name=='goal_ee')
 def vals(nodes):return {n.targets[0].id:eval(compile(ast.Expression(n.value),'<config>','eval'),{'__builtins__':{},'math':math}) for n in nodes if isinstance(n,ast.Assign)}
 return vals(g.body),*(vals(next(n for n in g.body if isinstance(n,ast.ClassDef) and n.name==name).body) for name in ['ranges','sphere_center'])
goal,ranges,center=config(cfgpath.read_text());oldg,oldr,oldc=config((O/'before_config.py.txt').read_text())
assert not any('floor_pitch' in k for k in goal)
assert ranges['pos_p']==[-math.pi/2.5,math.pi/3] and ranges['pos_y']==[-1.2,1.2]
assert ranges['pos_l']==[.35,.8]
zmax=center['z_invariant_offset']+.8*math.sin(math.pi/3)
assert ranges['height_above_ground']==[.02,zmax]
cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ManipLoco')
assert not any(isinstance(n,ast.FunctionDef) and n.name=='_get_ee_goal_pitch' for n in cls.body)
def get(cls,name):return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
def pitch_assign(cls):return next(n for n in ast.walk(get(cls,'_update_curr_ee_goal')) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='default_pitch' for t in n.targets))
original=root/'low-level-as2piper-head-orin-nocontact/legged_gym/envs/manip_loco/manip_loco.py'
originalcls=next(n for n in ast.parse(original.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ManipLoco')
assert ast.dump(pitch_assign(cls))==ast.dump(pitch_assign(originalcls))
oldcls=next(n for n in ast.parse((O/'before_manip_loco.py.txt').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ManipLoco')
for name in ['_resample_ee_goal_sphere_once','_resample_ee_goal_orn_once','_resample_ee_goal','_collision_check','_get_ee_goal_spherical_center']:
 assert ast.dump(get(cls,name))==ast.dump(get(oldcls,name)),name
for rg,ce,gg in [(oldr,oldc,oldg)]:
 assert rg['init_pos_start']==ranges['init_pos_start'] and ce==center and gg['arm_induced_pitch']==goal['arm_induced_pitch']
def sphere2cart(s):
 r,p,y=s.unbind(-1);return torch.stack([r*torch.cos(p)*torch.cos(y),r*torch.cos(p)*torch.sin(y),r*torch.sin(p)],-1)
def qa(q,v):
 t=2*torch.cross(q[:,1:],v,dim=-1);return v+q[:,:1]*t+torch.cross(q[:,1:],t,dim=-1)
def qi(q,v):
 q=q.clone();q[:,1:]*=-1;return qa(q,v)
ns=dict(torch=torch,torch_rand_float=lambda low,high,shape,device:low+(high-low)*torch.rand(shape,device=device),sphere2cart=sphere2cart,quat_apply=qa,quat_rotate_inverse=qi)
methods=['_resample_ee_goal_sphere_once','_resample_ee_goal_orn_once','_resample_ee_goal','_collision_check']
for name in methods:exec(compile(ast.Module(body=[get(cls,name)],type_ignores=[]),str(source),'exec'),ns)
e=types.SimpleNamespace(goal_ee_ranges=ranges,ee_goal_sphere=torch.zeros(6144,3),device='cpu')
raw=[]
for _ in range(20):ns[methods[0]](e,torch.arange(6144));raw.append(e.ee_goal_sphere.clone())
a=torch.cat(raw);xyz=sphere2cart(a)+torch.tensor([center['x_offset'],0,center['z_invariant_offset']]);assert xyz[:,2].max()<=zmax+1e-6
N=4096;e=types.SimpleNamespace(goal_ee_ranges=ranges,device='cpu',num_collision_check_samples=10,cfg=types.SimpleNamespace(env=types.SimpleNamespace(teleop_mode=False),goal_ee=types.SimpleNamespace(sphere_center=types.SimpleNamespace(**center))))
for name in methods:setattr(e,name,types.MethodType(ns[name],e))
for name in ['ee_start_sphere','ee_goal_sphere']:setattr(e,name,torch.tensor(ranges['init_pos_start']).repeat(N,1))
e.ee_goal_cart=sphere2cart(e.ee_goal_sphere);e.ee_goal_orn_delta_rpy=torch.zeros(N,3);e.goal_timer=torch.zeros(N)
e.collision_check_t=torch.linspace(0,1,10).view(1,1,-1)
e.collision_lower_limits=torch.tensor(goal['collision_lower_limits']);e.collision_upper_limits=torch.tensor(goal['collision_upper_limits'])
e.base_pos=torch.tensor([0,0,.3244069024246581]).repeat(N,1);e.arm_base_offset=torch.tensor([.22435,0,.1319]).repeat(N,1)
e.base_quat=torch.tensor([0.,0.,0.,1.]).repeat(N,1);e.base_yaw_quat=e.base_quat.clone();e.measured_heights=torch.zeros(N,1);e.env_origins=torch.zeros(N,3)
e._get_ee_goal_spherical_center=lambda:torch.tensor([center['x_offset'],0,center['z_invariant_offset']]).repeat(N,1)
active=[]
for _ in range(10):
 e._resample_ee_goal(torch.arange(N));assert not e._collision_check(torch.arange(N)).any()
 active.append(sphere2cart(e.ee_goal_sphere)+e._get_ee_goal_spherical_center())
active=torch.cat(active);assert active[:,2].min()>=.02-1e-6 and active[:,2].max()>1.10
protected=json.loads((R/'validation/spherical_reach/preserved_before.json').read_text())
for name,digest in protected['sources'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
for name,st in protected['checkpoints'].items():
 stat=(root/name).stat();assert (stat.st_size,stat.st_mtime_ns)==(st['size'],st['mtime']),name
result=dict(passed=True,raw_samples=len(a),active_geometric_goals=len(active),max_height_m=zmax,raw_min_height_m=float(xyz[:,2].min()),raw_max_height_m=float(xyz[:,2].max()),active_min_height_m=float(active[:,2].min()),active_max_height_m=float(active[:,2].max()),raw_below_clearance_fraction=float((xyz[:,2]<.02).float().mean()),pitch_assignment_matches_original=True,initial_EE_unchanged=True,protected_sources=len(protected['sources']),protected_checkpoints=len(protected['checkpoints']),IK_validity_checks=0,optimizer_updates=0,note='Range and geometric rejection only; not full workspace IK/trajectory/policy validation.')
(O/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
patch=''
for path,before in [(cfgpath,'before_config.py.txt'),(source,'before_manip_loco.py.txt')]:patch+=''.join(difflib.unified_diff((O/before).read_text().splitlines(True),path.read_text().splitlines(True),fromfile='before/'+str(path.relative_to(R)),tofile=str(path.relative_to(R))))
(O/'changes.patch').write_text(patch)
