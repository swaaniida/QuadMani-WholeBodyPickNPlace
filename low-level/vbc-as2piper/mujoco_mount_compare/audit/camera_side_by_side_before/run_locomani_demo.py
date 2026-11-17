#!/usr/bin/env python3
"""Procedural whole-body pose and locomotion visualizations for two mount variants.

Walking uses a bounded virtual base-velocity servo and scripted stepping because
this repository does not yet contain a learned AS2+PiPER locomotion policy.
Object attachment is a bounded compliant Cartesian grasp proxy after finger close.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from run_verified_demo import build_controller, capture, pd_control, safe_joint_pose

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/locomani'
SCENARIOS = (
    'standing',
    'tilt_in_place',
    'stand_walk_omni',
    'tilt_walk_omni',
    'pick_recover',
    'pick_drag',
    'pick_lift_drag',
)
DURATIONS = {
    'standing': 6.0,
    'tilt_in_place': 9.0,
    'stand_walk_omni': 18.0,
    'tilt_walk_omni': 20.0,
    'pick_recover': 16.0,
    'pick_drag': 16.0,
    'pick_lift_drag': 16.0,
}


def smooth(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def lerp(a, b, u):
    return np.asarray(a) + (np.asarray(b) - np.asarray(a)) * float(u)


def setup(model, data, mount_cfg, loco_cfg):
    data.qpos[:] = model.qpos0
    for leg in ('FL', 'FR', 'RL', 'RR'):
        safe_joint_pose(model, data, leg + '_hip_joint', 0.0)
        safe_joint_pose(model, data, leg + '_thigh_joint', float(mount_cfg['initial_thigh_rad']))
        safe_joint_pose(model, data, leg + '_calf_joint', float(mount_cfg['initial_calf_rad']))
    for i, value in enumerate(loco_cfg['travel_arm_rad'], 1):
        safe_joint_pose(model, data, 'piper_joint{}'.format(i), float(value))
    safe_joint_pose(model, data, 'piper_gripper_joint1', 0.049)
    safe_joint_pose(model, data, 'piper_gripper_joint2', -0.049)
    data.qpos[:3] = [0.0, 0.0, 0.35]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    lows = []
    for leg in ('FL', 'FR', 'RL', 'RR'):
        bid = model.body(leg + '_foot').id
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == bid and model.geom_contype[gid] and int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                lows.append(float(data.geom_xpos[gid, 2]) - float(model.geom_size[gid, 0]))
    data.qpos[2] += 0.002 - min(lows)
    obj_joint = model.joint('target_object_free')
    adr = int(obj_joint.qposadr)
    data.qpos[adr:adr + 3] = loco_cfg['object_initial_xyz_m']
    data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def phases(scenario, t):
    tilt = 0.0
    lower = 0.0
    close = 0.0
    lift = 0.0
    grasp = False
    command = np.zeros(3)  # body vx, body vy, yaw rate
    if scenario == 'tilt_in_place':
        tilt = smooth((t - 2.0) / 4.0)
    elif scenario == 'stand_walk_omni':
        command = omni_command(t, start=2.0)
    elif scenario == 'tilt_walk_omni':
        tilt = smooth((t - 1.0) / 3.0)
        command = omni_command(t, start=5.0)
    elif scenario in ('pick_recover', 'pick_drag', 'pick_lift_drag'):
        approach = smooth((t - 2.0) / 4.0)
        tilt = approach
        lower = approach
        close = smooth((t - 6.0) / 0.7)
        grasp = t >= 6.45
        if scenario == 'pick_recover':
            recover = smooth((t - 7.0) / 5.5)
            tilt *= 1.0 - recover
            lower *= 1.0 - recover
            lift = recover
        elif scenario == 'pick_drag':
            command = omni_command(t, start=7.2, segment=2.0, include_yaw=False)
        else:
            lift = smooth((t - 6.8) / 1.3)
            command = omni_command(t, start=8.2, segment=1.8, include_yaw=False)
    return tilt, lower, close, lift, grasp, command


def omni_command(t, start, segment=2.5, include_yaw=True):
    commands = ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0))
    if include_yaw:
        commands += ((0, 0, 1),)
    index = int((t - start) // segment)
    if t < start or index < 0 or index >= len(commands):
        return np.zeros(3)
    return np.asarray(commands[index], dtype=float)


def target_dict(mount_cfg, loco_cfg, scenario, t):
    tilt, lower, close, lift, grasp, command = phases(scenario, t)
    targets = {}
    walking = np.linalg.norm(command) > 0.0
    cycle = 2.0 * math.pi * 1.25 * t
    for leg in ('FL', 'FR', 'RL', 'RR'):
        front = leg.startswith('F')
        diagonal = 0.0 if leg in ('FL', 'RR') else math.pi
        swing = math.sin(cycle + diagonal) if walking else 0.0
        lift_phase = max(0.0, swing)
        side = 1.0 if leg.endswith('L') else -1.0
        yaw_side = side * command[2]
        targets[leg + '_hip_joint'] = 0.055 * command[1] * swing + 0.045 * yaw_side * swing
        targets[leg + '_thigh_joint'] = float(mount_cfg['initial_thigh_rad']) + (float(loco_cfg['front_tilt_thigh_delta_rad']) * tilt if front else 0.0) - 0.055 * command[0] * swing
        targets[leg + '_calf_joint'] = float(mount_cfg['initial_calf_rad']) + (float(loco_cfg['front_tilt_calf_delta_rad']) * tilt if front else 0.0) - 0.11 * lift_phase

    travel = np.asarray(loco_cfg['travel_arm_rad'])
    ground = np.asarray(loco_cfg['ground_grasp_arm_rad'])
    arm = lerp(travel, ground, lower)
    if scenario == 'pick_lift_drag':
        lifted = lerp(ground, travel, float(loco_cfg['lift_blend_from_grasp_to_travel']))
        arm = lerp(arm, lifted, lift)
    for i, value in enumerate(arm, 1):
        targets['piper_joint{}'.format(i)] = float(value)
    opening = 0.049 + (0.0185 - 0.049) * close
    targets['piper_gripper_joint1'] = opening
    targets['piper_gripper_joint2'] = -opening
    return targets, grasp, command, tilt, lower, close, lift


def apply_virtual_locomotion(model, data, loco_cfg, command):
    base = model.body('base_link').id
    data.xfrc_applied[base] = 0.0
    if np.linalg.norm(command) == 0.0:
        return np.zeros(3), 0.0
    quat = data.xquat[base]
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, quat)
    rotation = mat.reshape(3, 3)
    target_xy = rotation[:2, :2] @ (command[:2] * float(loco_cfg['locomotion_velocity_m_s']))
    velocity = data.qvel[:2]
    force = float(model.body_subtreemass[base]) * float(loco_cfg['virtual_base_velocity_gain_s_inv']) * (target_xy - velocity)
    limit = float(loco_cfg['virtual_base_force_limit_N'])
    norm = np.linalg.norm(force)
    if norm > limit:
        force *= limit / norm
    target_yaw = command[2] * float(loco_cfg['locomotion_yaw_rate_rad_s'])
    yaw_torque = float(loco_cfg['virtual_base_yaw_rate_gain_Nm_s_rad']) * (target_yaw - float(data.qvel[5]))
    yaw_limit = float(loco_cfg['virtual_base_yaw_torque_limit_Nm'])
    yaw_torque = max(-yaw_limit, min(yaw_limit, yaw_torque))
    data.xfrc_applied[base, :2] = force
    data.xfrc_applied[base, 5] = yaw_torque
    return force, yaw_torque


def apply_soft_grasp(model, data, loco_cfg, active):
    obj = model.body('target_object').id
    tcp = model.body('piper_umi_nominal_tcp').id
    if not active:
        return 0.0, float(np.linalg.norm(data.xpos[tcp] - data.xpos[obj]))
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, tcp, velocity, 0)
    object_joint = model.joint('target_object_free')
    object_velocity = data.qvel[int(object_joint.dofadr):int(object_joint.dofadr) + 3]
    offset = data.xpos[tcp] - data.xpos[obj]
    force = float(loco_cfg['soft_grasp_stiffness_N_m']) * offset + float(loco_cfg['soft_grasp_damping_N_s_m']) * (velocity[3:6] - object_velocity)
    limit = float(loco_cfg['soft_grasp_force_limit_N'])
    magnitude = float(np.linalg.norm(force))
    if magnitude > limit:
        force *= limit / magnitude
        magnitude = limit
    data.xfrc_applied[obj, :3] += force
    data.xfrc_applied[tcp, :3] -= force
    return magnitude, float(np.linalg.norm(offset))


def camera_metrics(model, data):
    cam = model.camera('wrist_camera').id
    obj = model.body('target_object').id
    tcp = model.body('piper_umi_nominal_tcp').id
    rotation = data.cam_xmat[cam].reshape(3, 3)
    forward = -rotation[:, 2]
    right = rotation[:, 0]
    up = rotation[:, 1]
    values = {}
    for name, point in (('object', data.xpos[obj]), ('tcp', data.xpos[tcp])):
        delta = point - data.cam_xpos[cam]
        depth = float(np.dot(delta, forward))
        horizontal = math.degrees(math.atan2(float(np.dot(delta, right)), depth))
        vertical = math.degrees(math.atan2(float(np.dot(delta, up)), depth))
        values[name + '_camera_depth_m'] = depth
        values[name + '_camera_horizontal_deg'] = horizontal
        values[name + '_camera_vertical_deg'] = vertical
    half = float(model.cam_fovy[cam]) / 2.0
    values['object_in_nominal_vertical_fov'] = float(values['object_camera_depth_m'] > 0 and abs(values['object_camera_vertical_deg']) <= half)
    values['tcp_in_nominal_vertical_fov'] = float(values['tcp_camera_depth_m'] > 0 and abs(values['tcp_camera_vertical_deg']) <= half)
    return values


def support_metrics(model, data, row):
    feet = []
    for leg in ('FL', 'FR', 'RL', 'RR'):
        if row[leg + '_ground_normal_N'] > 1.0:
            feet.append(tuple(float(v) for v in data.xpos[model.body(leg + '_foot').id, :2]))
    points = sorted(set(feet))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    if len(points) >= 3:
        lower = []
        for point in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)
        upper = []
        for point in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)
        hull = lower[:-1] + upper[:-1]
        com = np.asarray([row['com_x_m'], row['com_y_m']])
        signed = []
        for a, b in zip(hull, hull[1:] + hull[:1]):
            edge = np.asarray(b) - np.asarray(a)
            signed.append(float(np.cross(edge, com - np.asarray(a)) / np.linalg.norm(edge)))
        margin = min(signed)
    else:
        margin = -1.0
    robot_geoms = set()
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
        if name not in ('ground', 'target_object_geom'):
            robot_geoms.add(gid)
    self_contacts = 0
    minimum = 0.0
    distances = []
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.geom1 in robot_geoms and contact.geom2 in robot_geoms:
            self_contacts += 1
            distances.append(float(contact.dist))
    return {
        'loaded_foot_count': len(feet),
        'com_support_margin_m': margin,
        'robot_self_contact_count': self_contacts,
        'robot_self_min_contact_dist_m': min(distances) if distances else minimum,
    }


def run(args):
    mount_cfg = json.loads((HERE / 'models/mesh_seated/mount_config.json').read_text(encoding='utf-8'))
    loco_cfg = json.loads((HERE / 'locomani_config.json').read_text(encoding='utf-8'))
    model = mujoco.MjModel.from_xml_path(str(HERE / 'models/locomani' / (args.variant + '.xml')))
    data = mujoco.MjData(model)
    setup(model, data, mount_cfg, loco_cfg)
    controller = build_controller(model)
    duration = DURATIONS[args.scenario]
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / '{}_{}.csv'.format(args.variant, args.scenario)
    video_path = OUT / '{}_{}.mp4'.format(args.variant, args.scenario)
    renderer = wrist_renderer = writer = None
    camera = None
    if args.video:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
        renderer = mujoco.Renderer(model, height=360, width=640)
        wrist_renderer = mujoco.Renderer(model, height=180, width=320)
        camera = mujoco.MjvCamera()
        camera.distance = 1.65
        camera.azimuth = 145
        camera.elevation = -18
        writer = imageio.get_writer(str(video_path), fps=12, codec='libx264', quality=8, macro_block_size=1)
    dt = float(model.opt.timestep)
    log_stride = max(1, int(0.02 / dt))
    render_stride = max(1, int((1.0 / 12.0) / dt))
    writer_csv = None
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        csv_file = None
        step = 0
        while data.time < duration:
            targets, grasp, command, tilt, lower, close, lift = target_dict(mount_cfg, loco_cfg, args.scenario, data.time)
            effort = pd_control(model, data, targets, controller)
            data.xfrc_applied[:] = 0.0
            locomotion_force, yaw_torque = apply_virtual_locomotion(model, data, loco_cfg, command)
            grasp_force, grasp_error = apply_soft_grasp(model, data, loco_cfg, grasp)
            mujoco.mj_step(model, data)
            if step % log_stride == 0:
                mujoco.mj_forward(model, data)
                row = capture(model, data, effort)
                row.update(camera_metrics(model, data))
                row.update(support_metrics(model, data, row))
                row.update({
                    'tilt_blend': tilt,
                    'arm_lower_blend': lower,
                    'gripper_close_blend': close,
                    'arm_lift_blend': lift,
                    'soft_grasp_active': float(grasp),
                    'soft_grasp_force_N': grasp_force,
                    'soft_grasp_position_error_m': grasp_error,
                    'virtual_base_force_x_N': locomotion_force[0],
                    'virtual_base_force_y_N': locomotion_force[1],
                    'virtual_base_yaw_torque_Nm': yaw_torque,
                    'command_body_vx_unit': command[0],
                    'command_body_vy_unit': command[1],
                    'command_yaw_unit': command[2],
                    'object_x_m': data.xpos[model.body('target_object').id, 0],
                    'object_y_m': data.xpos[model.body('target_object').id, 1],
                    'object_z_m': data.xpos[model.body('target_object').id, 2],
                })
                if csv_file is None:
                    csv_file = csv.DictWriter(handle, fieldnames=list(row))
                    csv_file.writeheader()
                csv_file.writerow(row)
                if not all(math.isfinite(float(value)) for value in row.values()):
                    raise RuntimeError('Non-finite simulation state')
            if renderer is not None and step % render_stride == 0:
                base = model.body('base_link').id
                camera.lookat[:] = data.xpos[base] + np.array([0.12, 0.0, 0.08])
                renderer.update_scene(data, camera=camera)
                main = renderer.render()
                wrist_renderer.update_scene(data, camera='wrist_camera')
                wrist = wrist_renderer.render()
                frame = main.copy()
                frame[5:185, 315:635] = wrist
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                draw.rectangle((314, 4, 635, 185), outline=(80, 220, 255), width=2)
                direction = 'HOLD'
                if command[0] > 0.5:
                    direction = 'FORWARD'
                elif command[0] < -0.5:
                    direction = 'BACKWARD'
                elif command[1] > 0.5:
                    direction = 'LEFT'
                elif command[1] < -0.5:
                    direction = 'RIGHT'
                elif command[2] > 0.5:
                    direction = 'YAW LEFT'
                draw.text((8, 8), '{} | {} | {}'.format(args.variant, args.scenario, direction), fill=(255, 255, 255))
                draw.text((322, 10), 'WRIST CAMERA', fill=(80, 220, 255))
                frame = np.asarray(image)
                writer.append_data(frame)
            step += 1
    if writer is not None:
        writer.close()
        renderer.close()
        wrist_renderer.close()
        video_path.chmod(0o664)
    csv_path.chmod(0o664)
    print('Saved', csv_path, video_path if args.video else '')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('profiles_plate', 'plate_only'), required=True)
    parser.add_argument('--scenario', choices=SCENARIOS, required=True)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
