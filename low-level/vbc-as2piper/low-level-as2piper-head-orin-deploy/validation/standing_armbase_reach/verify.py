"""Execute actual sampler on CPU and check preserved initialization."""
from pathlib import Path
import ast, json, types, math, hashlib, difflib
import torch
O=Path(__file__).resolve().parent
R=O.parents[1]
torch.set_num_threads(1);torch.manual_seed(20261115)
src=R/'legged_gym/envs/manip_loco/manip_loco.py'
cfg=R/'legged_gym/envs/manip_loco/as2_head_orin_config.py'
def assignments(nodes):
    return {n.targets[0].id:ast.literal_eval(n.value) for n in nodes if isinstance(n,ast.Assign)}
def config(text):
    c=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='AS2HeadOrinCfg')
    goal=next(n for n in c.body if isinstance(n,ast.ClassDef) and n.name=='goal_ee')
    rg=next(n for n in goal.body if isinstance(n,ast.ClassDef) and n.name=='ranges')
    ce=next(n for n in goal.body if isinstance(n,ast.ClassDef) and n.name=='sphere_center')
    return assignments(goal.body),assignments(rg.body),assignments(ce.body)
goal,ranges,center=config(cfg.read_text())
oldgoal,oldranges,oldcenter=config((O/'before_config.py.txt').read_text())
tree=ast.parse(src.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ManipLoco')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_resample_ee_goal_sphere_once')
ns=dict(torch=torch,torch_rand_float=lambda low,high,shape,device:(high-low)*torch.rand(shape,device=device)+low)
exec(compile(ast.Module(body=[method],type_ignores=[]),str(src),'exec'),ns)
env=types.SimpleNamespace(goal_ee_ranges=ranges,ee_goal_sphere=torch.zeros(6144,3),device='cpu')
samples=[]
for _ in range(20):
    ns['_resample_ee_goal_sphere_once'](env,torch.arange(6144))
    samples.append(env.ee_goal_sphere.clone())
s=torch.cat(samples)
assert torch.isfinite(s).all()
for i,key in enumerate(('pos_l','pos_p','pos_y')):
    assert (s[:,i]>=ranges[key][0]).all() and (s[:,i]<=ranges[key][1]).all()
z=center['z_invariant_offset']+s[:,0]*torch.sin(s[:,1])
assert (z>=.02-1e-6).all() and (z<=1+1e-6).all()
def initial(g,rg,ce):
    r,p,y=rg['init_pos_start']
    return [ce['x_offset']+r*math.cos(p)*math.cos(y),ce['y_offset']+r*math.cos(p)*math.sin(y),ce['z_invariant_offset']+r*math.sin(p)],g['arm_induced_pitch']-p
newhome,newpitch=initial(goal,ranges,center)
oldhome,oldpitch=initial(oldgoal,oldranges,oldcenter)
assert max(abs(x-y) for x,y in zip(newhome,oldhome))<1e-12
assert abs(newpitch-oldpitch)<1e-12
assert 'horizontal_reach' not in ranges
assert 'horizontal_reach' not in ast.dump(method)
for name in ['_resample_ee_goal_orn_once','_get_ee_goal_pitch','_get_ee_goal_spherical_center','_resample_ee_goal','_collision_check']:
    oldcls=next(n for n in ast.parse((O/'before_manip_loco.py.txt').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ManipLoco')
    a=next(n for n in oldcls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    b=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(a)==ast.dump(b),name
result=dict(samples=len(s),passed=True,initial_world_xyz=newhome,initial_pitch_rad=newpitch,
            empirical_height_min_m=float(z.min()),empirical_height_max_m=float(z.max()),
            radius_elevation_correlation=float(torch.corrcoef(s[:,:2].T)[0,1]),
            runtime_IK_validity_checks=0,optimizer_updates=0,
            scope='Offline finite IK tests do not certify every continuous pose, trajectory, torque or rough terrain.')
(O/'verification.json').write_text(json.dumps(result,indent=2))
patch=''
for path,before in [(cfg,'before_config.py.txt'),(src,'before_manip_loco.py.txt')]:
    patch+=''.join(difflib.unified_diff((O/before).read_text().splitlines(True),path.read_text().splitlines(True),fromfile='before/'+str(path.relative_to(R)),tofile=str(path.relative_to(R))))
(O/'spherical.patch').write_text(patch)
print(json.dumps(result,indent=2))
