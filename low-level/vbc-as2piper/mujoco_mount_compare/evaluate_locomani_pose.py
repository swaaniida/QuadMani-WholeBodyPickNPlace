#!/usr/bin/env python3
"""Compare the former high arm pose with the center-mount locomotion-ready pose."""
import json
from pathlib import Path

import mujoco
import numpy as np

from run_locomani_demo import camera_metrics, setup, support_metrics
from run_verified_demo import build_controller, capture, pd_control

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/locomani'


def evaluate(variant, label, arm_pose, mount_cfg, loco_cfg):
    model = mujoco.MjModel.from_xml_path(str(HERE / 'models/locomani' / (variant + '.xml')))
    data = mujoco.MjData(model)
    local_cfg = dict(loco_cfg)
    local_cfg['travel_arm_rad'] = list(arm_pose)
    setup(model, data, mount_cfg, local_cfg)
    controller = build_controller(model)
    max_arm_command = np.zeros(6)
    while data.time < 5.0:
        targets = {}
        for leg in ('FL', 'FR', 'RL', 'RR'):
            targets[leg + '_hip_joint'] = 0.0
            targets[leg + '_thigh_joint'] = float(mount_cfg['initial_thigh_rad'])
            targets[leg + '_calf_joint'] = float(mount_cfg['initial_calf_rad'])
        for i, value in enumerate(arm_pose, 1):
            targets['piper_joint{}'.format(i)] = float(value)
        targets['piper_gripper_joint1'] = 0.049
        targets['piper_gripper_joint2'] = -0.049
        effort = pd_control(model, data, targets, controller)
        max_arm_command = np.maximum(max_arm_command, np.abs(data.ctrl[12:18]))
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    row = capture(model, data, effort)
    row.update(support_metrics(model, data, row))
    row.update(camera_metrics(model, data))
    base = model.body('base_link').id
    tcp = model.body('piper_umi_nominal_tcp').id
    arm_bodies = [
        body for body in range(1, model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or '').startswith('piper_')
    ]
    arm_mass = sum(float(model.body_mass[body]) for body in arm_bodies)
    arm_com = sum(float(model.body_mass[body]) * data.xipos[body] for body in arm_bodies) / arm_mass
    jacobian = np.zeros((3, model.nv))
    rotation_jacobian = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacobian, rotation_jacobian, tcp)
    dofs = [int(model.joint('piper_joint{}'.format(i)).dofadr) for i in range(1, 7)]
    singular = np.linalg.svd(jacobian[:, dofs], compute_uv=False)
    result = {
        'variant': variant,
        'pose': label,
        'commanded_arm_rad': list(arm_pose),
        'actual_arm_rad': [float(data.qpos[int(model.joint('piper_joint{}'.format(i)).qposadr)]) for i in range(1, 7)],
        'tcp_relative_to_base_m': (data.xpos[tcp] - data.xpos[base]).tolist(),
        'whole_robot_com_relative_to_base_m': [
            row['com_x_m'] - row['base_x_m'], row['com_y_m'] - row['base_y_m'], row['com_z_m'] - row['base_z_m']
        ],
        'arm_com_relative_to_base_m': (arm_com - data.xpos[base]).tolist(),
        'arm_mass_kg': arm_mass,
        'position_jacobian_singular_values': singular.tolist(),
        'position_manipulability_product': float(np.prod(singular)),
        'max_arm_command_Nm': max_arm_command.tolist(),
        'pitch_deg': row['pitch_deg'],
        'roll_deg': row['roll_deg'],
        'four_feet_loaded': all(row[leg + '_ground_normal_N'] > 1.0 for leg in ('FL', 'FR', 'RL', 'RR')),
        'com_support_margin_m': row['com_support_margin_m'],
        'camera': {key: value for key, value in row.items() if 'camera_' in key or 'nominal_vertical_fov' in key},
        'mujoco_warning_count': row['mujoco_warning_count'],
    }
    return result


def main():
    mount_cfg = json.loads((HERE / 'models/mesh_seated/mount_config.json').read_text(encoding='utf-8'))
    loco_cfg = json.loads((HERE / 'locomani_config.json').read_text(encoding='utf-8'))
    old_pose = [0.0, 1.2, -1.4, 0.0, 0.2, 0.0]
    selected = loco_cfg['travel_arm_rad']
    results = []
    for variant in ('profiles_plate', 'plate_only'):
        results.append(evaluate(variant, 'former_high_pose', old_pose, mount_cfg, loco_cfg))
        results.append(evaluate(variant, 'selected_locomotion_ready_pose', selected, mount_cfg, loco_cfg))
    (OUT / 'pose_evaluation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    for result in results:
        print(result['variant'], result['pose'], 'TCP', result['tcp_relative_to_base_m'], 'COM', result['whole_robot_com_relative_to_base_m'])


if __name__ == '__main__':
    main()
