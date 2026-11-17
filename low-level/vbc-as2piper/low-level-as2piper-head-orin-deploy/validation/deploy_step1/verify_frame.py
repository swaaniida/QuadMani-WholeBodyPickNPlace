"""Execute actual center/terrain methods on CPU; no training/simulator creation."""
import ast, json, hashlib, types, difflib
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / 'low-level-as2piper-head-orin-nocontact'
OUT = Path(__file__).parent

def quat_apply(q, v):
    t = 2 * torch.cross(q[:, :3], v, dim=-1)
    return v + q[:, 3:] * t + torch.cross(q[:, :3], t, dim=-1)

def quat_rpy(r, p, y):
    r,p,y = (torch.as_tensor(x, dtype=torch.float64)/2 for x in (r,p,y))
    return torch.stack([r.sin()*p.cos()*y.cos()-r.cos()*p.sin()*y.sin(),
                        r.cos()*p.sin()*y.cos()+r.sin()*p.cos()*y.sin(),
                        r.cos()*p.cos()*y.sin()-r.sin()*p.sin()*y.cos(),
                        r.cos()*p.cos()*y.cos()+r.sin()*p.sin()*y.sin()], -1)

path = Path('legged_gym/envs/manip_loco/manip_loco.py')
ns = {'torch': torch, 'quat_apply': quat_apply}
cls = next(n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n, ast.ClassDef))
for name in ('_get_ee_goal_spherical_center', '_termination_ground_height'):
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
N=1000
e = types.SimpleNamespace(num_envs=N, device='cpu')
e.cfg = types.SimpleNamespace(goal_ee=types.SimpleNamespace(sphere_center=types.SimpleNamespace(height_roll_pitch_invariant=True)),
                              terrain=types.SimpleNamespace(mesh_type='plane', border_size=0., horizontal_scale=.1, vertical_scale=.01))
e.root_states = torch.zeros(N, 13, dtype=torch.float64)
e.root_states[:,0]=1.; e.root_states[:,1]=2.
e.root_states[:,2]=.30 + .10*torch.sin(torch.linspace(0.,25.,N))
e.base_pos=e.root_states[:,:3]
e.base_quat=quat_rpy(torch.linspace(-.5,.5,N),torch.linspace(-.65,.65,N),torch.zeros(N))
e.base_yaw_quat=quat_rpy(torch.zeros(N),torch.zeros(N),torch.zeros(N))
e.arm_base_offset=torch.tensor([[.22435,0,.1319]],dtype=torch.float64).repeat(N,1)
e.ee_goal_center_offset=torch.tensor([[.22435,0,.4563069024246581]],dtype=torch.float64).repeat(N,1)
e._termination_ground_height=types.MethodType(ns['_termination_ground_height'],e)
e._get_ee_goal_spherical_center=types.MethodType(ns['_get_ee_goal_spherical_center'],e)
centers=e._get_ee_goal_spherical_center()
assert (centers-centers[:1]).abs().max()<1e-12
# Identical yaw-frame floor/up targets stay fixed through body z/roll/pitch.
errors={}
for height in (.02,.80):
    relative=torch.tensor([[.5,0,height-.4563069024246581]],dtype=torch.float64).repeat(N,1)
    goals=centers+quat_apply(e.base_yaw_quat,relative)
    errors[str(height)]=float((goals[:,2]-height).abs().max())
    assert errors[str(height)]<1e-12
actual_arm=e.base_pos+quat_apply(e.base_quat,e.arm_base_offset)
assert (actual_arm[:,2].max()-actual_arm[:,2].min())>.1
# Yaw/translation equivariance, while virtual height remains constant.
e.base_yaw_quat=quat_rpy(torch.zeros(N),torch.zeros(N),torch.ones(N)*1.5707963267948966)
yawcenter=e._get_ee_goal_spherical_center()
assert torch.allclose(yawcenter[:,0],torch.ones(N,dtype=torch.float64))
assert torch.allclose(yawcenter[:,1],torch.ones(N,dtype=torch.float64)*2.22435)
# Rough local ground is read from terrain grid, not body height/roll/pitch probes.
e.cfg.terrain.mesh_type='trimesh'
e.height_samples=torch.full((40,40),20.,dtype=torch.float64)
rough=e._get_ee_goal_spherical_center()
assert (rough[:,2]-.6563069024246581).abs().max()<1e-12
# Exercise the unchanged real goal-update pipeline with utility from Isaac source.
utils = ROOT.parent/'third_party/isaacgym/python/isaacgym/torch_utils.py'
fn = next(n for n in ast.parse(utils.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='sphere2cart')
fn.decorator_list=[]
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(utils),'exec'),ns)
ns['quat_from_euler_xyz']=quat_rpy
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_update_curr_ee_goal')
exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
e.cfg.env=types.SimpleNamespace(teleop_mode=True)
e.cfg.goal_ee.arm_induced_pitch=2.455216137744035
e.cfg.goal_ee.orientation_roll_offset=0.
e.cfg.terrain.mesh_type='plane'
e.curr_ee_goal_sphere=torch.tensor([[.60,-.8,0.]],dtype=torch.float64).repeat(N,1)
e.curr_ee_goal_cart=torch.zeros(N,3,dtype=torch.float64)
e.ee_goal_orn_delta_rpy=torch.zeros(N,3,dtype=torch.float64)
e.goal_timer=torch.zeros(N);e.traj_total_timesteps=torch.ones(N)*100
e.stop_update_goal=False;e._resample_ee_goal=lambda ids:None
ns['_update_curr_ee_goal'](e)
assert (e.curr_ee_goal_cart_world-e.curr_ee_goal_cart_world[:1]).abs().max()<1e-12
assert (e.ee_goal_orn_quat-e.ee_goal_orn_quat[:1]).abs().max()<1e-12

# Legacy flag-off arm-base behavior remains available.
e.cfg.goal_ee.sphere_center.height_roll_pitch_invariant=False
e.ee_goal_center_offset.zero_()
assert torch.equal(e._get_ee_goal_spherical_center(),actual_arm)
# Byte-level preservation of all pre-existing source/assets and checkpoints.
manifest=json.loads((OUT/'source_manifest.json').read_text())
external_file = ROOT/'validation/deploy_step7/external_asset_updates.json'
external = {x['path']:x for x in json.loads(external_file.read_text())['files']} if external_file.exists() else {}
for rel,digest in manifest.items():
    expected_source = external[rel]['observed_original_sha256'] if rel in external else digest
    assert hashlib.sha256((SOURCE/rel).read_bytes()).hexdigest()==expected_source,rel
changes=[]; diff=[]
for rel,digest in manifest.items():
    target=ROOT/rel
    if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
        changes.append(rel)
        diff.extend(difflib.unified_diff((SOURCE/rel).read_text().splitlines(True),target.read_text().splitlines(True),fromfile='original/'+rel,tofile='deploy/'+rel))
expected=['CHANGES_KO.md','legged_gym/envs/manip_loco/as2_head_orin_config.py','legged_gym/envs/manip_loco/manip_loco.py']
step7 = ROOT/'validation/deploy_step7/applied_changes.json'
if step7.exists():
    expected = sorted(set(expected) | (set(json.loads(step7.read_text())) & set(manifest)) | set(external))
assert sorted(changes)==sorted(expected),changes
for c in json.loads((OUT/'checkpoint_inventory.json').read_text()):
    stat=Path(c['path']).stat();assert stat.st_size==c['size'] and stat.st_mtime_ns==c['mtime_ns']
result=dict(passed=True,body_poses=N,ground_and_upper_target_z_max_error_m=errors,
            actual_arm_height_span_m=float(actual_arm[:,2].max()-actual_arm[:,2].min()),
            rough_ground_offset_m=.2,source_files_unchanged=len(manifest)-len(external),external_asset_updates_since_step1=list(external),checkpoints_stat_unchanged=226,
            modified_cloned_files=changes,training_started=False,
            limitation='CPU actual-method coordinate tests only; no dynamics/IK feasibility claim. XY/yaw still follow base, terrain from local grid.')
(OUT/'verification.json').write_text(json.dumps(result,indent=2))
(OUT/'step1.patch').write_text(''.join(diff))
print(json.dumps(result,indent=2))
