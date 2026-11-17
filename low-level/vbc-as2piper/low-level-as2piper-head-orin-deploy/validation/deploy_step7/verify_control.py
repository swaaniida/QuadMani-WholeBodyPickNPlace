"""CPU tests of actual command filter, delay selection and PhysX drive-property mapping."""
import ast,json,importlib.util
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('piper_control',ROOT/'legged_gym/utils/piper_control.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
initial=torch.zeros(3,6);lo=torch.ones(6)*-1;hi=torch.ones(6)
f=mod.ArmCommandFilter(initial,lo,hi,3.,.02)
commands=[]
for _ in range(25):commands.append(f.step(torch.ones(3,6)*10))
trajectory=torch.stack([initial]+commands)
assert torch.diff(trajectory,dim=0).abs().max()<=.060001
assert trajectory.max()<=1
f.reset(torch.tensor([1]),torch.ones(1,6)*-.5)
assert torch.all(f.last[1]==-.5) and torch.all(f.history[1]==-.5)
delayed=mod.ArmCommandFilter(initial,lo,hi,3.,.02,2)
assert torch.equal(delayed.step(torch.ones_like(initial)),initial)
assert torch.equal(delayed.step(torch.ones_like(initial)),initial)
assert torch.allclose(delayed.step(torch.ones_like(initial)),torch.ones_like(initial)*.06)
# Exact pre-physics action selection extracted from current environment method.
source=(ROOT/'legged_gym/envs/manip_loco/manip_loco.py').read_text();tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
statements=[]
for node in step.body:
 statements.append(node)
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='actions' for t in node.targets):break
function=ast.FunctionDef(name='select',args=step.args,body=statements,decorator_list=[],returns=None,type_comment=None)
module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]));scope={'torch':torch};exec(compile(module,'actual_step_prefix','exec'),scope)
from types import SimpleNamespace as NS
for actual_delay in [0,1,3]:
 for time in [0,240000]:
  e=NS(action_delay=actual_delay,action_history_buf=torch.zeros(3,actual_delay+2,18),global_steps=time,cfg=NS(env=NS(action_delay_warmup_steps=240000)),clip_actions=100,device='cpu',render=lambda:None,_reindex_all=lambda x:x)
  for t in range(8):scope['select'](e,torch.full((3,18),float(t+1)))
  expected=8-(0 if time<240000 else actual_delay)
  assert torch.all(e.actions[:,:12]==expected)
  assert torch.all(e.actions[:,12:]==0)
props=np.zeros(20,dtype=[(n,'f8') for n in ['stiffness','damping','effort','velocity']]);props['effort']=100;props['stiffness']=123;props['damping']=12;props['velocity']=5
for strength in [.7,1,1.3]:
 out=mod.arm_drive_properties(props,np.ones(6)*strength,400.,40.,3.)
 assert np.allclose(out['stiffness'][12:18],400*strength)
 assert np.allclose(out['damping'][12:18],40*strength)
 assert np.array_equal(out[:12],props[:12]) and np.array_equal(out[18:],props[18:])
 assert np.array_equal(out['effort'],props['effort'])
readspec=importlib.util.spec_from_file_location('read_piper_limits',OUT/'read_piper_limits.py');reader=importlib.util.module_from_spec(readspec);readspec.loader.exec_module(reader)
assert reader.speed_unit('S-V1.5-7')==.001
assert reader.speed_unit('S-V1.5-8')==.01
assert reader.speed_unit('S-V1.8-8')==.01
assert reader.speed_unit(-1199) is None
# Official basic-PiPER mechanical presets agree; retain existing angular bounds.
import xml.etree.ElementTree as ET
constants=ast.parse((OUT/'official/pyAgxArm/pyAgxArm/api/constants.py').read_text())
assignment=next(n for n in constants.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROBOT_JOINT_LIMIT_PRESET_RAD' for t in n.targets))
preset=ast.literal_eval(assignment.value)['piper']
urdf=ET.parse(ROOT/'resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf').getroot()
for i in range(1,7):
 limit=urdf.find("joint[@name='piper_joint%d']/limit"%i)
 assert np.allclose([float(limit.get('lower')),float(limit.get('upper'))],preset['joint%d'%i],atol=1e-6)
 assert float(limit.get('velocity'))==3.
result=dict(passed=True,command_slew_limit_rad_s=3,per_step_delta_max_rad=float(torch.diff(trajectory,dim=0).abs().max()),partial_reset_clean=True,arm_delay_test_steps=2,leg_delay_tests=[0,1,3],arm_policy_actions_masked=True,drive_stiffness_DR=[280,520],drive_damping_DR=[28,52],effort_saturation_unchanged=True,official_basic_joint_limits_match=True,firmware_speed_units_tested=True,training_started=False)
(OUT/'unit_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
