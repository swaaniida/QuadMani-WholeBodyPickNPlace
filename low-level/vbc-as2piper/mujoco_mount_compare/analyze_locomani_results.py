#!/usr/bin/env python3
"""Summarize and assert the limits of the procedural loco-manipulation suite."""
import csv
import json
import math
from pathlib import Path

import mujoco

from run_locomani_demo import DURATIONS, SCENARIOS

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/locomani'


def main():
    summary = {}
    for variant in ('profiles_plate', 'plate_only'):
        model = mujoco.MjModel.from_xml_path(str(HERE / 'models/locomani' / (variant + '.xml')))
        for scenario in SCENARIOS:
            path = OUT / '{}_{}.csv'.format(variant, scenario)
            rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(path.open(encoding='utf-8'))]
            assert rows[-1]['time_s'] >= DURATIONS[scenario] - 0.03
            item = {
                'variant': variant,
                'scenario': scenario,
                'duration_s': rows[-1]['time_s'],
                'base_xyz_range_m': {
                    axis: [min(row['base_' + axis + '_m'] for row in rows), max(row['base_' + axis + '_m'] for row in rows)]
                    for axis in ('x', 'y', 'z')
                },
                'max_abs_roll_deg': max(abs(row['roll_deg']) for row in rows),
                'pitch_range_deg': [min(row['pitch_deg'] for row in rows), max(row['pitch_deg'] for row in rows)],
                'four_feet_loaded_fraction': sum(row['loaded_foot_count'] == 4 for row in rows) / len(rows),
                'min_com_support_margin_m_when_3plus_feet': min(
                    row['com_support_margin_m'] for row in rows if row['loaded_foot_count'] >= 3
                ),
                'max_robot_self_contacts': max(row['robot_self_contact_count'] for row in rows),
                'max_robot_contact_penetration_m': max(0.0, -min(row['robot_self_min_contact_dist_m'] for row in rows)),
                'mujoco_warning_count': max(row['mujoco_warning_count'] for row in rows),
                'object_in_vertical_fov_fraction': sum(row['object_in_nominal_vertical_fov'] for row in rows) / len(rows),
                'tcp_in_vertical_fov_fraction': sum(row['tcp_in_nominal_vertical_fov'] for row in rows) / len(rows),
                'object_horizontal_angle_range_deg': [
                    min(row['object_camera_horizontal_deg'] for row in rows),
                    max(row['object_camera_horizontal_deg'] for row in rows),
                ],
                'tcp_horizontal_angle_range_deg': [
                    min(row['tcp_camera_horizontal_deg'] for row in rows),
                    max(row['tcp_camera_horizontal_deg'] for row in rows),
                ],
                'object_xyz_final_m': [rows[-1]['object_' + axis + '_m'] for axis in ('x', 'y', 'z')],
                'object_max_z_m': max(row['object_z_m'] for row in rows),
                'soft_grasp_max_error_m': max(
                    (row['soft_grasp_position_error_m'] for row in rows if row['soft_grasp_active']), default=0.0
                ),
                'virtual_base_force_max_N': max(
                    math.hypot(row['virtual_base_force_x_N'], row['virtual_base_force_y_N']) for row in rows
                ),
                'arm_actuator_saturation_fraction': {},
            }
            for index in range(12, 18):
                actuator = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
                cap = max(abs(float(value)) for value in model.actuator_ctrlrange[index])
                item['arm_actuator_saturation_fraction'][actuator] = sum(
                    abs(row[actuator + '_command']) >= cap - 1e-6 for row in rows
                ) / len(rows)
            assert item['mujoco_warning_count'] == 0
            assert item['max_abs_roll_deg'] < 12.0
            assert item['pitch_range_deg'][1] < 35.0
            assert item['base_xyz_range_m']['z'][0] > 0.15
            assert item['four_feet_loaded_fraction'] > 0.94
            # MuJoCo's soft mesh contact can produce sub-millimetre transient
            # overlap while preventing deeper penetration.
            assert item['max_robot_contact_penetration_m'] <= 0.001
            assert item['object_in_vertical_fov_fraction'] > 0.98
            assert item['tcp_in_vertical_fov_fraction'] > 0.98
            horizontal_half_fov = math.degrees(math.atan((16.0 / 9.0) * math.tan(math.radians(float(model.cam_fovy[model.camera('wrist_camera').id])) / 2.0)))
            assert max(abs(value) for value in item['object_horizontal_angle_range_deg']) <= horizontal_half_fov
            assert max(abs(value) for value in item['tcp_horizontal_angle_range_deg']) <= horizontal_half_fov
            if scenario.startswith('pick_'):
                assert item['soft_grasp_max_error_m'] < 0.065
            if scenario == 'pick_recover':
                assert item['object_max_z_m'] > 0.30
            summary[variant + '_' + scenario] = item

    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = [
        '| Variant | Scenario | Roll max | Pitch range | 4 feet | COM margin min | Camera object/TCP | Grasp error |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for key, item in summary.items():
        variant = item['variant']
        scenario = item['scenario']
        lines.append(
            '| {} | {} | {:.2f}° | {:.1f}..{:.1f}° | {:.1f}% | {:.1f}mm | {:.1f}/{:.1f}% | {:.1f}mm |'.format(
                variant,
                scenario,
                item['max_abs_roll_deg'],
                item['pitch_range_deg'][0],
                item['pitch_range_deg'][1],
                100.0 * item['four_feet_loaded_fraction'],
                1000.0 * item['min_com_support_margin_m_when_3plus_feet'],
                100.0 * item['object_in_vertical_fov_fraction'],
                100.0 * item['tcp_in_vertical_fov_fraction'],
                1000.0 * item['soft_grasp_max_error_m'],
            )
        )
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
