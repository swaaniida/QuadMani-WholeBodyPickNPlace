#!/usr/bin/env python3
"""Evaluate a centerline rail moving the PiPER arm base above the AS2 head.

This is a design study. Rail/carriage/computer masses are explicit assumptions;
the leg motion is an actuator-only scripted stepping perturbation, not a gait policy.
"""
import csv
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / 'models/mount_workspace'
OUT = HERE / 'results/head_overhead_stability'
SOURCE = MODEL_DIR / 'profiles_plate_forward.xml'
DT = 0.002
FPS = 12
DURATION = 8.0
COMPUTER_MASS_KG = 1.0  # placeholder; replace with measured computer + battery mass
CARRIAGE_MASS_KG = 0.50  # placeholder; rail carriage / linear actuator allowance
RAIL_MASS_KG = 0.50  # placeholder; pair of longitudinal guide rails
AS2_HEAD_X_M = 0.33470532  # max-X from the original base_link.STL in the base frame
RAIL_START_X_M = 0.005
RAIL_TRAVEL_M = AS2_HEAD_X_M - RAIL_START_X_M
ARM_BASE_Z_M = 0.180  # 30 mm above the original AS2 base-mesh maximum Z
TRAVEL_ARM = np.array([0.0, 0.76, -0.89, 0.0, 0.45, 0.0])


def add_design_parts():
    root = ET.parse(SOURCE).getroot()
    root.set('model', 'as2piper_centerline_forward_rail_study')
    base = root.find("./worldbody/body[@name='base_link']")
    mount = root.find(".//body[@name='mount_plate']")
    piper = mount.find("body[@name='piper_base_link']")
    piper_pos = [float(x) for x in piper.get('pos').split()]
    mount.remove(piper)

    # Centered compute payload fixed above the AS2 body. Mass and location are
    # placeholders and are applied identically to both position conditions.
    computer = ET.SubElement(base, 'body', name='compute_payload', pos='0 0 0.155')
    ET.SubElement(computer, 'inertial', pos='0 0 0', mass=str(COMPUTER_MASS_KG),
                  diaginertia='0.0009083 0.001525 0.0020333')
    ET.SubElement(computer, 'geom', name='compute_payload_visual', type='box',
                  pos='0 0 0', size='0.06 0.05 0.015', rgba='0.10 0.42 0.82 1',
                  contype='0', conaffinity='0')

    # Pair of front-back guide rails. The 430 mm span covers the center position
    # and a carriage centered above the AS2 nose, with 50 mm front overhang.
    for side, y in (('left', 0.070), ('right', -0.070)):
        rail = ET.SubElement(base, 'body', name='linear_rail_' + side,
                             pos='0.17 {} 0.175'.format(y))
        mass = RAIL_MASS_KG / 2.0
        ET.SubElement(rail, 'inertial', pos='0 0 0', mass=str(mass),
                      diaginertia='0.0000044 0.00385 0.00385')
        ET.SubElement(rail, 'geom', name='linear_rail_' + side + '_visual',
                      type='box', size='0.215 0.006 0.004', rgba='0.90 0.58 0.12 1',
                      contype='0', conaffinity='0')

    carriage = ET.SubElement(base, 'body', name='piper_carriage',
                             pos='{} 0 {}'.format(RAIL_START_X_M, ARM_BASE_Z_M))
    ET.SubElement(carriage, 'joint', name='piper_rail_slide', type='slide',
                  axis='1 0 0', range='0 {}'.format(RAIL_TRAVEL_M), limited='true',
                  damping='20', armature='0.02')
    ET.SubElement(carriage, 'inertial', pos='0 0 0', mass=str(CARRIAGE_MASS_KG),
                  diaginertia='0.0008 0.0014 0.0014')
    ET.SubElement(carriage, 'geom', name='piper_carriage_visual', type='box',
                  pos='0 0 -0.003', size='0.055 0.08 0.003',
                  rgba='0.83 0.72 0.26 1', contype='0', conaffinity='0')
    piper.set('pos', '0 0 0')
    carriage.append(piper)
    actuator = root.find('actuator')
    ET.SubElement(actuator, 'motor', name='u_piper_rail_slide', joint='piper_rail_slide',
                  gear='1', ctrllimited='true', ctrlrange='-80 80')

    # Mocap marker for whole-system CoM. Position is updated from inertial
    # centers every rendered/logged frame.
    world = root.find('worldbody')
    marker = ET.SubElement(world, 'body', name='whole_system_com_marker',
                           mocap='true', pos='0 0 0')
    ET.SubElement(marker, 'geom', name='whole_system_com_geom', type='sphere',
                  size='0.018', rgba='1 0.04 0.8 1', contype='0', conaffinity='0')
    destination = MODEL_DIR / 'head_overhead_arm_rail_study.xml'
    ET.ElementTree(root).write(destination, encoding='utf-8', xml_declaration=True)
    return destination


def smooth(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def pose_rpy(q):
    w, x, y, z = map(float, q)
    r = math.atan2(2 * (w*x + y*z), 1 - 2*(x*x+y*y))
    p = math.asin(max(-1.0, min(1.0, 2*(w*y-z*x))))
    return math.degrees(r), math.degrees(p)


def whole_com(model, data):
    masses = model.body_mass[1:]
    positions = data.xipos[1:]
    total = float(masses.sum())
    return (masses[:, None] * positions).sum(axis=0) / total


def contact_margin(model, data, com):
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'ground')
    feet = []
    for leg in ('FL', 'FR', 'RL', 'RR'):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, leg + '_foot')
        loaded = False
        for k in range(data.ncon):
            c = data.contact[k]
            if ground not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == ground else c.geom1
            if model.geom_bodyid[other] == bid:
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, k, force)
                loaded = loaded or force[0] > 1.0
        if loaded:
            feet.append(data.xpos[bid, :2].copy())
    if len(feet) < 3:
        return -1.0, len(feet)
    points = sorted(set((float(p[0]), float(p[1])) for p in feet))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo = []
    for p in points:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    hi = []
    for p in reversed(points):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0: hi.pop()
        hi.append(p)
    hull = lo[:-1] + hi[:-1]
    distances = []
    for a, b in zip(hull, hull[1:] + hull[:1]):
        edge = np.asarray(b) - np.asarray(a)
        distances.append(float(np.cross(edge, com[:2] - np.asarray(a)) / np.linalg.norm(edge)))
    return min(distances), len(feet)


def init_state(model, data):
    data.qpos[:] = model.qpos0
    data.qpos[:7] = [0, 0, 0.35, 1, 0, 0, 0]
    for leg in ('FL', 'FR', 'RL', 'RR'):
        for suffix, value in (('hip', 0.0), ('thigh', 0.8), ('calf', -1.6)):
            j = model.joint(leg + '_' + suffix + '_joint')
            data.qpos[int(j.qposadr)] = value
    for i, value in enumerate(TRAVEL_ARM, 1):
        data.qpos[int(model.joint('piper_joint{}'.format(i)).qposadr)] = value
    data.qpos[int(model.joint('piper_gripper_joint1').qposadr)] = 0.049
    data.qpos[int(model.joint('piper_gripper_joint2').qposadr)] = -0.049
    mujoco.mj_forward(model, data)
    foot_lows = []
    for leg in ('FL', 'FR', 'RL', 'RR'):
        bid = model.body(leg + '_foot').id
        gids = [g for g in range(model.ngeom)
                if model.geom_bodyid[g] == bid and model.geom_contype[g]
                and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        if gids:
            foot_lows.append(data.geom_xpos[gids[0], 2] - model.geom_size[gids[0], 0])
    if foot_lows:
        data.qpos[2] += 0.002 - min(foot_lows)
    mujoco.mj_forward(model, data)


def targets(model, data, t, moving):
    desired = {}
    rail_target = RAIL_TRAVEL_M * smooth((t - 1.0) / 2.0) if moving else 0.0
    gait_blend = smooth((t - 4.0) / 0.6)
    phase = 2 * math.pi * 0.85 * max(0.0, t - 4.0)
    for leg in ('FL', 'FR', 'RL', 'RR'):
        diagonal = 0 if leg in ('FL', 'RR') else math.pi
        cyc = phase + diagonal
        thigh = 0.8 - 0.07 * math.sin(cyc) * gait_blend
        calf = -1.6 - 0.12 * max(0.0, math.sin(cyc)) * gait_blend
        desired[leg + '_hip_joint'] = 0.0
        desired[leg + '_thigh_joint'] = thigh
        desired[leg + '_calf_joint'] = calf
    # Modest arm sweep while standing; arm stays tucked during stepping.
    arm_blend = smooth((t - 1.0) / 2.0) if moving else smooth((t - 1.0) / 2.0)
    for i, value in enumerate(TRAVEL_ARM, 1):
        delta = (0.0, 0.10, -0.10, 0.0, -0.08, 0.0)[i-1]
        desired['piper_joint{}'.format(i)] = float(value + delta * arm_blend * (1.0 - gait_blend))
    desired['piper_gripper_joint1'] = 0.049
    desired['piper_gripper_joint2'] = -0.049
    desired['piper_rail_slide'] = rail_target
    return desired


def controller(model):
    result = {}
    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                  int(model.actuator_trnid[aid, 0]))
        jid = model.joint(jname).id
        kind = ('rail' if jname == 'piper_rail_slide' else
                'gripper' if 'gripper' in jname else
                'arm' if jname.startswith('piper_') else 'leg')
        gains = {'rail': (900.0, 65.0), 'gripper': (1800.0, 35.0),
                 'arm': (35.0, 1.5), 'leg': (90.0, 3.5)}[kind]
        result[jname] = (aid, int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]), gains)
    return result


def run_case(model_path, name, moving):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    init_state(model, data)
    controls = controller(model)
    renderer = mujoco.Renderer(model, height=360, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.05, 0, 0.36]
    camera.distance, camera.azimuth, camera.elevation = 1.45, 145, -12
    video_path = OUT / (name + '.mp4')
    csv_path = OUT / (name + '.csv')
    writer = imageio.get_writer(str(video_path), fps=FPS, codec='libx264', quality=8,
                                macro_block_size=1)
    rows = []
    com_mid = model.body('whole_system_com_marker').mocapid[0]
    frame_stride = round(1.0 / (FPS * DT))
    log_stride = round(0.02 / DT)
    with csv_path.open('w', newline='') as handle:
        csv_writer = None
        for step in range(round(DURATION / DT)):
            t = data.time
            desired = targets(model, data, t, moving)
            mujoco.mj_forward(model, data)
            for jname, (aid, qp, qv, (kp, kd)) in controls.items():
                err = desired[jname] - float(data.qpos[qp])
                u = kp * err - kd * float(data.qvel[qv]) + float(data.qfrc_bias[qv])
                lower, upper = model.actuator_ctrlrange[aid]
                data.ctrl[aid] = max(float(lower), min(float(upper), u))
            mujoco.mj_step(model, data)
            if step % log_stride == 0 or step == round(DURATION/DT)-1:
                mujoco.mj_forward(model, data)
                com = whole_com(model, data)
                margin, nfeet = contact_margin(model, data, com)
                bid = model.body('base_link').id
                roll, pitch = pose_rpy(data.xquat[bid])
                rail_q = float(data.qpos[int(model.joint('piper_rail_slide').qposadr)])
                row = {'time_s': float(data.time), 'rail_position_m': rail_q,
                       'total_mass_kg': float(model.body_mass[1:].sum()),
                       'com_x_m': float(com[0]), 'com_y_m': float(com[1]),
                       'com_z_m': float(com[2]), 'base_x_m': float(data.xpos[bid, 0]),
                       'base_y_m': float(data.xpos[bid, 1]),
                       'base_z_m': float(data.xpos[bid, 2]), 'roll_deg': roll,
                       'pitch_deg': pitch, 'loaded_foot_count': nfeet,
                       'com_support_margin_m': margin, 'contacts': int(data.ncon)}
                rows.append(row)
                if csv_writer is None:
                    csv_writer = csv.DictWriter(handle, fieldnames=list(row))
                    csv_writer.writeheader()
                csv_writer.writerow(row)
            if step % frame_stride == 0:
                com = whole_com(model, data)
                data.mocap_pos[com_mid] = com
                camera.lookat[:] = data.xpos[model.body('base_link').id] + [0.10, 0, 0.12]
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                phase_label = ('CENTERED ARM / STOW' if data.time < 1 else
                               'SLIDING ARM ABOVE AS2 HEAD' if moving and data.time < 3.1 else
                               'ARM OVER HEAD / SCRIPTED STEP' if moving and data.time >= 4.6 else
                               'CENTERED ARM / SCRIPTED STEP' if data.time >= 4.6 else
                               'ARM MOTION / SETTLING')
                draw.rectangle((0, 0, 640, 56), fill=(0, 0, 0, 185))
                draw.text((12, 7), phase_label, fill='white')
                draw.text((12, 29),
                          't={:.1f}s  rail={:+.0f}mm  COM=({:+.3f},{:+.3f},{:.3f})m  pitch={:+.1f}deg'.format(
                              data.time, 1000*float(data.qpos[int(model.joint('piper_rail_slide').qposadr)]),
                              *com, pose_rpy(data.xquat[model.body('base_link').id])[1]), fill='white')
                draw.text((12, 337), 'Magenta marker: whole-system COM | actuator-only scripted stepping; not a learned gait', fill='white')
                writer.append_data(np.asarray(frame))
    writer.close()
    renderer.close()
    arr = np.array([[r['com_x_m'], r['com_y_m'], r['com_z_m']] for r in rows])
    pitch = np.array([r['pitch_deg'] for r in rows])
    roll = np.array([r['roll_deg'] for r in rows])
    margin = np.array([r['com_support_margin_m'] for r in rows])
    feet = np.array([r['loaded_foot_count'] for r in rows])
    valid_margin = margin[feet >= 3]
    settled = [r for r in rows if r['time_s'] >= 3.5]
    local_com_x = np.array([r['com_x_m'] - r['base_x_m'] for r in settled])
    local_com_z = np.array([r['com_z_m'] - r['base_z_m'] for r in settled])
    at_release = min(rows, key=lambda r: abs(r['time_s'] - 3.5))
    final = rows[-1]
    settled_base_z = [r['base_z_m'] for r in rows if 1.0 <= r['time_s'] <= 3.8]
    summary = {
        'condition': name, 'arm_moves_forward': bool(moving),
        'initial_arm_base_x_relative_to_as2_m': RAIL_START_X_M,
        'final_arm_base_x_relative_to_as2_m': AS2_HEAD_X_M if moving else RAIL_START_X_M,
        'arm_base_z_relative_to_as2_m': ARM_BASE_Z_M,
        'rail_travel_m': RAIL_TRAVEL_M if moving else 0.0,
        'head_x_from_original_base_mesh_m': AS2_HEAD_X_M,
        'rail_lower_surface_clearance_above_base_mesh_max_z_m': 0.175 - 0.004 - 0.13478276,
        'piper_base_bottom_clearance_above_base_mesh_max_z_m': ARM_BASE_Z_M - 0.13478276,
        'computer_mass_assumption_kg': COMPUTER_MASS_KG,
        'computer_com_position_relative_to_base_m': [0, 0, 0.155],
        'rail_pair_mass_assumption_kg': RAIL_MASS_KG,
        'carriage_mass_assumption_kg': CARRIAGE_MASS_KG,
        'total_model_mass_kg': float(model.body_mass[1:].sum()),
        'com_xyz_min_m': arr.min(axis=0).tolist(), 'com_xyz_max_m': arr.max(axis=0).tolist(),
        'com_xyz_mean_m': arr.mean(axis=0).tolist(),
        'com_x_relative_to_base_at_rail_release_m': float(at_release['com_x_m']-at_release['base_x_m']),
        'com_x_relative_to_base_mean_during_scripted_step_m': float(local_com_x.mean()),
        'com_z_relative_to_base_mean_after_settle_m': float(local_com_z.mean()),
        'base_vertical_settle_drop_m': float(rows[0]['base_z_m'] - np.mean(settled_base_z)),
        'base_x_drift_m': float(final['base_x_m']-rows[0]['base_x_m']),
        'pitch_abs_max_deg': float(np.max(np.abs(pitch))),
        'roll_abs_max_deg': float(np.max(np.abs(roll))),
        'support_margin_min_when_at_least_three_feet_loaded_m': float(np.min(valid_margin)),
        'support_margin_mean_when_at_least_three_feet_loaded_m': float(np.mean(valid_margin)),
        'support_margin_invalid_sample_count': int(np.sum(feet < 3)),
        'frames_with_less_than_three_loaded_feet_fraction': float(np.mean(feet < 3)),
        'duration_s': DURATION, 'timestep_s': DT, 'render_fps': FPS,
        'controller': 'PD + joint gravity feedforward; no base wrench or learned gait policy',
        'video': str(video_path), 'timeseries': str(csv_path),
    }
    return summary


def check_arm_clearance(model_path):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    init_state(model, data)
    controls = controller(model)
    pairs = {}
    samples = 0
    minimum = float('inf')
    for _ in range(round(3.2 / DT)):
        desired = targets(model, data, data.time, True)
        mujoco.mj_forward(model, data)
        for name, (aid, qp, qv, (kp, kd)) in controls.items():
            value = kp * (desired[name] - data.qpos[qp]) - kd * data.qvel[qv] + data.qfrc_bias[qv]
            lower, upper = model.actuator_ctrlrange[aid]
            data.ctrl[aid] = max(float(lower), min(float(upper), float(value)))
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                   int(model.geom_bodyid[c.geom1])) or ''
            b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                   int(model.geom_bodyid[c.geom2])) or ''
            arm1, arm2 = b1.startswith('piper_'), b2.startswith('piper_')
            if arm1 != arm2 and 'ground' not in (b1, b2):
                samples += 1
                minimum = min(minimum, float(c.dist))
                key = '{} <-> {}'.format(b1 if arm1 else b2, b2 if arm1 else b1)
                pairs[key] = pairs.get(key, 0) + 1
    return {
        'test': 'actuator-controlled slide from center to AS2 head coordinate; arm stowed',
        'duration_s': float(data.time),
        'rail_travel_m': float(data.qpos[int(model.joint('piper_rail_slide').qposadr)]),
        'arm_vs_nonarm_self_contact_samples': samples,
        'arm_vs_nonarm_pairs': pairs,
        'minimum_contact_distance_m': None if samples == 0 else minimum,
        'mesh_aabb_head_x_m': AS2_HEAD_X_M,
        'mesh_aabb_max_z_m': 0.13478276,
        'rail_bottom_z_m': 0.171,
        'rail_to_mesh_aabb_vertical_clearance_m': 0.171 - 0.13478276,
        'piper_base_bottom_z_m': ARM_BASE_Z_M,
        'piper_base_to_mesh_aabb_vertical_clearance_m': ARM_BASE_Z_M - 0.13478276,
        'model_nq_nv_nu': [int(model.nq), int(model.nv), int(model.nu)],
        'structural_strength_flex_and_fastener_validation': 'not modeled',
    }


def make_visual_comparisons():
    center = imageio.get_reader(str(OUT / 'arm_center_actuator_step.mp4'))
    overhead = imageio.get_reader(str(OUT / 'arm_over_head_actuator_step.mp4'))
    writer = imageio.get_writer(str(OUT / 'center_vs_head_overhead.mp4'), fps=FPS,
                                codec='libx264', quality=8, macro_block_size=1)
    try:
        count = min(center.count_frames(), overhead.count_frames())
        for i in range(count):
            pair = Image.fromarray(np.concatenate((center.get_data(i), overhead.get_data(i)), axis=1))
            draw = ImageDraw.Draw(pair)
            draw.rectangle((0, 0, 1280, 25), fill=(0, 0, 0, 200))
            draw.text((12, 6), 'ARM AT CENTER', fill='white')
            draw.text((652, 6), 'ARM MOVES ABOVE AS2 HEAD', fill='white')
            writer.append_data(np.asarray(pair))
    finally:
        writer.close()
        center.close()
        overhead.close()

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for name, label, color in (('arm_center_actuator_step', 'center', 'tab:blue'),
                               ('arm_over_head_actuator_step', 'above head', 'tab:orange')):
        with (OUT / (name + '.csv')).open() as handle:
            rows = list(csv.DictReader(handle))
        t = np.asarray([float(r['time_s']) for r in rows])
        com_x = np.asarray([float(r['com_x_m']) - float(r['base_x_m']) for r in rows])
        pitch = np.asarray([float(r['pitch_deg']) for r in rows])
        rail_mm = 1000 * np.asarray([float(r['rail_position_m']) for r in rows])
        margin = np.asarray([float(r['com_support_margin_m']) if int(r['loaded_foot_count']) >= 3 else np.nan for r in rows])
        axes[0].plot(t, com_x, color=color, label=label)
        axes[1].plot(t, pitch, color=color, label=label)
        axes[2].plot(t, margin, color=color, label=label)
        if label == 'above head':
            rail_axis = axes[0].twinx()
            rail_axis.plot(t, rail_mm, color='tab:green', linestyle='--', label='rail displacement')
            rail_axis.set_ylabel('Rail displacement (mm)')
            rail_axis.set_ylim(-5, 350)
            rail_axis.legend(loc='lower right')
    axes[0].set_ylabel('CoM x relative to base (m)')
    axes[1].set_ylabel('Body pitch (deg)')
    axes[2].set_ylabel('CoM support margin (m)')
    axes[2].set_xlabel('Time (s)')
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc='best')
    fig.suptitle('AS2 + PiPER: moving arm base above the head (330 mm travel)')
    fig.savefig(OUT / 'com_stability_timeseries.png', dpi=160)
    plt.close(fig)


def main():
    os.umask(0o002)
    OUT.mkdir(parents=True, exist_ok=True)
    model_path = add_design_parts()
    results = [run_case(model_path, 'arm_center_actuator_step', False),
               run_case(model_path, 'arm_over_head_actuator_step', True)]
    clearance = check_arm_clearance(model_path)
    make_visual_comparisons()
    (OUT / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (OUT / 'collision_clearance.json').write_text(json.dumps(clearance, indent=2), encoding='utf-8')
    with (OUT / 'summary.md').open('w', encoding='utf-8') as f:
        f.write('# AS2 PiPER 머리 위까지 이동하는 전후 레일의 안정성 검토\n\n')
        f.write('AS2 base_link.STL의 머리 전방 기준 x={:.1f} mm까지 arm base를 x={:.1f} mm에서 {:.1f} mm 이동시켰다. 레일 하단은 원본 base 메시 최고점보다 {:.1f} mm, PiPER 베이스 하단은 {:.1f} mm 위에 둔 설계 가정이다.\n\n'.format(
            1000*AS2_HEAD_X_M,1000*RAIL_START_X_M,1000*AS2_HEAD_X_M,1000*(0.175-0.004-0.13478276),1000*(ARM_BASE_Z_M-0.13478276)))
        f.write('컴퓨팅 장치 1.0 kg, 레일 0.50 kg, 캐리지 0.50 kg은 실측 전 가정이며 두 조건에 동일하게 적용했다.\n\n')
        f.write('발 동작은 관절 PD+중력 보상으로 만든 작은 scripted stepping 동작이다. 발이 실제로 들리는 학습 보행 정책이 아니며 안정성 보증 시험도 아니다. 자홍색 구는 전체 시스템 질량중심이다.\n\n')
        f.write('| 조건 | CoM x / base 기준, rail end (m) | 최대 pitch / roll (deg) | 지지 여유 최소 (m) | 3발 미만 비율 |\n|---|---:|---:|---:|---:|\n')
        for x in results:
            f.write('| {} | {:.3f} | {:.2f}/{:.2f} | {:.3f} | {:.3f} |\n'.format(
                x['condition'], x['com_x_relative_to_base_at_rail_release_m'],
                x['pitch_abs_max_deg'], x['roll_abs_max_deg'],
                x['support_margin_min_when_at_least_three_feet_loaded_m'],
                x['frames_with_less_than_three_loaded_feet_fraction']))
        f.write('\nAt the rail end the whole-system CoM x is {:.1f} mm forward of center condition. Peak pitch rises from {:.2f}° to {:.2f}° and the minimum support margin drops from {:.3f} m to {:.3f} m, remaining positive in this test.\n\n'.format(
            1000*(results[1]['com_x_relative_to_base_at_rail_release_m']-results[0]['com_x_relative_to_base_at_rail_release_m']),
            results[0]['pitch_abs_max_deg'], results[1]['pitch_abs_max_deg'],
            results[0]['support_margin_min_when_at_least_three_feet_loaded_m'],
            results[1]['support_margin_min_when_at_least_three_feet_loaded_m']))
        f.write('COM x (base-relative) at rail end: center {:.3f} m, above-head {:.3f} m. '.format(
            results[0]['com_x_relative_to_base_at_rail_release_m'],
            results[1]['com_x_relative_to_base_at_rail_release_m']))
        f.write('Support margin is reported only when three or more feet are loaded; frames with fewer loaded feet are counted separately.\n\n')
        f.write('Both models settle downward by about {:.0f} mm in the first second. Four feet are loaded for nearly all subsequent samples because the scripted leg motion is small; base drift was {:.0f}/{:.0f} mm backward in this artificial motion. These results show no immediate overturning in the test, but do not establish gait stability.\n\n'.format(
            1000*results[0]['base_vertical_settle_drop_m'],
            -1000*results[0]['base_x_drift_m'], -1000*results[1]['base_x_drift_m']))
        f.write('During the 3.2 s slide-to-head clearance check, the model recorded {} arm-to-body contact samples. The rail bottom is {:.1f} mm above the global base-mesh maximum Z by AABB; structural stiffness and fasteners are not modeled.\n\n'.format(
            clearance['arm_vs_nonarm_self_contact_samples'],
            1000*clearance['rail_to_mesh_aabb_vertical_clearance_m']))
        f.write('\n')
        for x in results:
            f.write('- [{}](./{})\n'.format(x['condition'], Path(x['video']).name))
        f.write('- [장착부 mesh clearance / 충돌 결과](./collision_clearance.json)\n')
        f.write('- [CoM 및 동역학 시계열 비교 영상](./center_vs_head_overhead.mp4)\n')
        f.write('- [CoM, pitch, support margin 그래프](./com_stability_timeseries.png)\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
