#!/usr/bin/env python3
# Python 3.8 - verify dimensions, actuated joints and optional Isaac Gym loading.
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'low-level/resources/robots/as2piper'
U = ASSET / 'urdf/as2piper_umi_actuated_v7.urdf'
VIEW = ASSET / 'urdf/as2piper_umi_open_pose_v7.urdf'
REPORT = ROOT / 'tools/umi_v7_fit_report.json'
for p in (U,VIEW,REPORT):
    assert p.is_file(), 'Not found: '+str(p)

r = ET.parse(str(U)).getroot()
v = ET.parse(str(VIEW)).getroot()
active = lambda r: [j for j in r.findall('joint') if j.get('type') in ('revolute','continuous','prismatic')]
assert len(active(r)) == 20, 'Actuated model should be 20 DOF'
assert len(active(v)) == 18, 'Open viewer should be 18 DOF'
assert [j.get('name') for j in active(r)[-2:]] == ['piper_gripper_joint1','piper_gripper_joint2']
for j in active(r)[-2:]:
    assert j.get('type') == 'prismatic'
rep=json.loads(REPORT.read_text(encoding='utf-8'))
assert abs(rep['new_open_outer_width_mm']-134.5)<.001
assert abs(rep['right_left_height_center_difference_after_mm']) < .001
assert abs(rep['world_UP_shift_mm']['right']+rep['world_UP_shift_mm']['left']) < .001
assert rep['closed_inner_AABB_clearance_mm'] >= 2-.001
print('Open full OUTER width: %.3f mm' % rep['new_open_outer_width_mm'])
print('Right UP / left UP displacement: %+.3f / %+.3f mm' % (
    rep['world_UP_shift_mm']['right'],rep['world_UP_shift_mm']['left']))
print('Height centers difference after: %.6f mm' % rep['right_left_height_center_difference_after_mm'])
print('Closed inner clearance (STL Y-AABB): %.3f mm' % rep['closed_inner_AABB_clearance_mm'])
print('Finger joint limits (m):',rep['joint_limits_m'])
print('V7 GEOMETRIC CHECK: PASS')

if '--skip-isaacgym' in sys.argv:
    print('Isaac Gym load check skipped by request')
    sys.exit(0)

# Isaac Gym module must precede torch in same process.
from isaacgym import gymapi

gym=gymapi.acquire_gym()
params=gymapi.SimParams()
params.dt=1.0/60.0
params.up_axis=gymapi.UP_AXIS_Z
params.gravity=gymapi.Vec3(0,0,-9.81)
params.use_gpu_pipeline=False
params.physx.use_gpu=False
sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,params)
assert sim is not None,'Isaac Gym PhysX sim creation failed'
try:
    opt=gymapi.AssetOptions()
    opt.fix_base_link=False
    opt.collapse_fixed_joints=True
    asset=gym.load_asset(sim,str(ASSET),'urdf/'+U.name,opt)
    assert asset is not None,'Isaac Gym load_asset failed'
    names=list(gym.get_asset_dof_names(asset))
    assert len(names)==20 and names[-2:]==['piper_gripper_joint1','piper_gripper_joint2'],names
    props=gym.get_asset_dof_properties(asset)
    for i in (18,19):
        print('%s range [%.4f, %.4f] m' % (names[i],props['lower'][i],props['upper'][i]))
    print('ISAAC GYM V7 20DOF LOAD: PASS')
finally:
    gym.destroy_sim(sim)
