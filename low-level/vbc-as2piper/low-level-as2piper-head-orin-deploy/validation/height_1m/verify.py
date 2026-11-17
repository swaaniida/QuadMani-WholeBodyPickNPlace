"""Execute the actual continuous sampler AST; config-only boundary change."""
from pathlib import Path
import ast,json,math,types
import torch,numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];torch.set_num_threads(1);torch.manual_seed(123)
source=R/'legged_gym/envs/manip_loco/manip_loco.py';tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ManipLoco');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_resample_ee_goal_sphere_once')
ns=dict(torch=torch,torch_rand_float=lambda low,high,shape,device:(high-low)*torch.rand(shape,device=device)+low)
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
tree=ast.parse((R/'legged_gym/envs/manip_loco/as2_head_orin_config.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AS2HeadOrinCfg');goal=next(n for n in cls.body if isinstance(n,ast.ClassDef) and n.name=='goal_ee');ranges=next(n for n in goal.body if isinstance(n,ast.ClassDef) and n.name=='ranges');center=next(n for n in goal.body if isinstance(n,ast.ClassDef) and n.name=='sphere_center')
def values(nodes):return {n.targets[0].id:ast.literal_eval(n.value) for n in nodes if isinstance(n,ast.Assign)}
ranges=values(ranges.body);center=values(center.body);cfg=types.SimpleNamespace(goal_ee=types.SimpleNamespace(sphere_center=types.SimpleNamespace(**center)))
e=types.SimpleNamespace(cfg=cfg,goal_ee_ranges=ranges,ee_goal_sphere=torch.zeros(6144,3),device='cpu');ids=torch.arange(6144)
mn=torch.ones(2)*2;mx=torch.ones(2)*-2
for _ in range(20):
 ns['_resample_ee_goal_sphere_once'](e,ids);r,p,y=e.ee_goal_sphere.T
 x=center['x_offset']+r*torch.cos(p);z=center['z_invariant_offset']+r*torch.sin(p)
 assert torch.isfinite(e.ee_goal_sphere).all()
 assert (x>=.45-1e-6).all() and (x<=.975+1e-6).all()
 assert (z>=.02-1e-6).all() and (z<=1.+1e-6).all()
 assert (y.abs()<=.30+1e-6).all()
 mn=torch.minimum(mn,torch.tensor([x.min(),z.min()]));mx=torch.maximum(mx,torch.tensor([x.max(),z.max()]))
assert mx[1]>.995
# Confirm the farthest upper corner belongs to the revised outer bounds.
r=math.hypot(.75065,1-center['z_invariant_offset']);phi=math.atan2(1-center['z_invariant_offset'],.75065)
assert r<=ranges['pos_l'][1]+1e-10 and phi<=ranges['pos_p'][1]
result=dict(passed=True,samples=122880,nominal_x_min_m=float(mn[0]),nominal_x_max_m=float(mx[0]),ground_z_min_m=float(mn[1]),ground_z_max_m=float(mx[1]),configured_z_max_m=1.,radius_outer_max_m=ranges['pos_l'][1],elevation_max_rad=ranges['pos_p'][1],bank=False,IK_validity_checks=0,optimizer_updates=0,note='Geometric generation boundaries only, not reachability or dynamic performance certification.')
(O/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
