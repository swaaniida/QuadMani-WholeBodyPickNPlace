#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AS2+PiPER UMI V7: V6 -> open outer width 134.5mm + symmetric height registration.

Only transforms *printed finger* visuals/collisions/estimated inertial origins.
Does not change arm or OEM gripper joints/mount, source STL or V6.
Height: AS2 base_link +Z, correcting the remaining gripper cross-section X
misregistration (V6 had aligned the *longitudinal* gripper_base Z only).
The two carriages remain at their OEM locations; validate screws/rail fit!
Python 3.8-compatible. Needs finish_umi_gripper_v6.py alongside this script.
"""
import json
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
import numpy as np

import finish_umi_gripper_v6 as g

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'low-level/resources/robots/as2piper'
SRC = ASSET / 'urdf/as2piper_umi_actuated_v6.urdf'
OUT = ASSET / 'urdf/as2piper_umi_actuated_v7.urdf'
OPEN = ASSET / 'urdf/as2piper_umi_open_pose_v7.urdf'
REPORT = ROOT / 'tools/umi_v7_fit_report.json'

BODY_WIDTH = 0.1425
END_INSET = 0.004
OPEN_Q = {1: 0.05, 2: -0.05}
MIN_INNER_GAP = 0.002
LIMIT_RESOLUTION = .0001
MAX_HEIGHT_SHIFT = .02  # stop instead of hiding an unexpectedly broken frame


def fail(message):
    raise RuntimeError(message)


def parse_frame_rotation(root, target):
    """Orientation of any link expressed in AS2 base_link frame at joint q=0.
    Only orientation matters; joint translations/axes are handled separately.
    """
    parents = {}
    for j in root.findall('joint'):
        pn = j.find('parent')
        cn = j.find('child')
        if pn is None or cn is None:
            continue
        parents[cn.get('link')] = j
    chain = []
    cursor = target
    seen = set()
    while cursor != 'base_link':
        if cursor in seen:
            fail('Cycle in URDF tree at ' + cursor)
        seen.add(cursor)
        if cursor not in parents:
            fail('No path from base_link to ' + target + '; stopped at ' + cursor)
        j = parents[cursor]
        chain.append(j)
        cursor = j.find('parent').get('link')
    R = np.eye(3)
    for j in reversed(chain):
        o = j.find('origin')
        R = R @ g.rotation(g.V(o.get('rpy', '0 0 0')) if o is not None else [0, 0, 0])
    return R


def shape_points(root, i, cache):
    return g.finger_points(root, i, OPEN_Q[i], cache)


def part_centroid(points):
    # AABB center avoids vertex-density bias caused by STL triangulation.
    return .5 * (np.min(points, axis=0) + np.max(points, axis=0))


def outside_inside(p, low_y, high_y):
    out_width = g.bounds(p[high_y], 1)[1] - g.bounds(p[low_y], 1)[0]
    in_gap = g.bounds(p[high_y], 1)[0] - g.bounds(p[low_y], 1)[1]
    return out_width, in_gap


def main():
    if not SRC.is_file():
        fail('Source V6 missing: ' + str(SRC))
    root = ET.parse(str(SRC)).getroot()
    joints = [j for j in root.findall('joint') if j.get('type') in ('revolute', 'continuous', 'prismatic')]
    if len(joints) != 20:
        fail('Input V6 must have 20 actuated DOF (found %d)' % len(joints))
    j1, j2 = g.finger_joint(root, 1), g.finger_joint(root, 2)
    for j in (j1, j2):
        if j.get('type') != 'prismatic':
            fail('Gripper joint not prismatic: ' + j.get('name'))
    lim1, lim2 = j1.find('limit'), j2.find('limit')
    if abs(float(lim1.get('upper')) - OPEN_Q[1]) > 1e-8 or abs(float(lim2.get('lower')) - OPEN_Q[2]) > 1e-8:
        fail('OEM gripper ±50mm OPEN physical stroke changed; refusing to extrapolate')
    a1 = g.joint_transform(root, 1, 0.)[2]
    a2 = g.joint_transform(root, 2, 0.)[2]
    if not (np.allclose(a1, [0,-1,0], atol=1e-6) and np.allclose(a2, [0,-1,0], atol=1e-6)):
        fail('Unexpected gripper slide axis; inspect actual URDF')

    # Gripper_base-to-AS2 orientation at the viewer's neutral arm pose.
    # AS2 robot forward +X, robot LEFT +Y, robot UP +Z.
    Rbg = parse_frame_rotation(root, 'piper_gripper_base')
    up_in_gripper = Rbg.T @ [0., 0., 1.]
    cache = {}
    p = {i: shape_points(root, i, cache) for i in (1,2)}
    c = {i: part_centroid(p[i]) for i in (1,2)}
    y_robot = {i: float((Rbg @ c[i])[1]) for i in (1,2)}
    right, left = sorted((1,2), key=lambda i: y_robot[i])
    if abs(y_robot[left] - y_robot[right]) < .02:
        fail('Robot-front RIGHT/LEFT fingers cannot be distinguished in base_link Y; no guessed sign')
    low_y, high_y = sorted((1,2), key=lambda i: float(c[i][1]))
    print('Robot-front RIGHT -> piper_gripper_link%d, LEFT -> piper_gripper_link%d' % (right,left))
    print('Gripper-base Y- / Y+ fingers: %d / %d' % (low_y,high_y))
    print('Gripper local X to robot UP projection: %+0.5f' % up_in_gripper[0])
    if (low_y, high_y) != (1,2):
        fail('Expected finger1 on gripper Y- and finger2 on Y+ at q_open; check joint layout before setting close limits')
    if abs(up_in_gripper[0]) < 0.65:
        fail('Local X is not predominantly robot-up at the neutral arm pose (projection %.3f). Refusing ambiguous height shift.' % up_in_gripper[0])

    # 1) Increase finger exterior span from 130.5 -> 134.5mm without moving the gripper chassis.
    # Outward carriage location is still at q=±0.05m. Translate only custom assemblies.
    old_width, old_gap = outside_inside(p, low_y, high_y)
    if abs(old_width - .1305) > .003:
        fail('V6 source outer-open span should be about 130.5mm, but is %.3fmm. Check the source file/config rather than assuming this is V6.' % (old_width*1000))
    old_y = {i: g.bounds(p[i], 1) for i in (1,2)}
    half = BODY_WIDTH/2 - END_INSET
    delta_y = {low_y: -half - old_y[low_y][0], high_y: half - old_y[high_y][1]}
    for i in (1,2):
        g.move_printed_assembly(root, i, [0., delta_y[i], 0.])
    p = {i: shape_points(root, i, cache) for i in (1,2)}
    width, gap_open = outside_inside(p, low_y, high_y)
    if abs(width - .1345) > 1e-7:
        fail('134.5mm OUTER width check failed: %.5fmm' % (width*1000))
    if gap_open < MIN_INNER_GAP:
        fail('Even fully OPEN the printed fingers overlap, gap %.3fmm' % (gap_open*1000))

    # 2) Register *cross-section height* (gripper local X, not local Z).
    # At current arm pose the above rotation tells whether X+ or X- is up.
    # Compare actual printed STL AABB centers projected onto AS2 world UP.
    # Height is the midpoint of the actual transformed mesh's projected robot-Z
    # extent (rather than just the gripper-local Z or an STL vertex mean).
    up_extent_before = {i: (float((p[i] @ up_in_gripper).min()), float((p[i] @ up_in_gripper).max()))
                        for i in (1,2)}
    height_before = {i: 0.5*sum(up_extent_before[i]) for i in (1,2)}
    mid = (height_before[right] + height_before[left]) / 2
    x_up = float(up_in_gripper[0])
    delta_x = {i: (mid - height_before[i]) / x_up for i in (1,2)}
    if max(abs(v) for v in delta_x.values()) > MAX_HEIGHT_SHIFT:
        fail('Height correction >20mm; inspect frames rather than detaching PLA mount: %s' % delta_x)
    for i in (1,2):
        g.move_printed_assembly(root, i, [delta_x[i], 0., 0.])
    p = {i: shape_points(root, i, cache) for i in (1,2)}
    up_extent_after = {i: (float((p[i] @ up_in_gripper).min()), float((p[i] @ up_in_gripper).max()))
                       for i in (1,2)}
    height_after = {i: 0.5*sum(up_extent_after[i]) for i in (1,2)}
    delta_h = height_after[right] - height_after[left]
    if abs(delta_h) > 1e-7:
        fail('Robot-up height still not aligned: %.4f mm' % (delta_h*1000))
    if abs(delta_x[right] + delta_x[left]) > 1e-8:
        fail('Heights were not moved in equal and opposite amounts')
    # Per user, right should go UP, left DOWN. We report if the input frame
    # gives the opposite diagnosis instead of silently swapping finger names.
    move_up_right = x_up * delta_x[right]
    move_up_left = x_up * delta_x[left]
    if move_up_right < -1e-7 or move_up_left > 1e-7:
        fail('Detected RIGHT above LEFT in the URDF neutral AS2 frame, contradicting the requested RIGHT-UP / LEFT-DOWN correction. Refusing to move them in the wrong directions; inspect finger mapping/frame.')

    # 3) Recalculate max-safe closing stroke from real printed-mesh Y bounds.
    # The two finger slides both have axis -Y, with q1>0,q2<0.
    width_after, gap_open_after = outside_inside(p,low_y,high_y)
    if abs(width_after - .1345) > 1e-7 or abs(gap_open_after-gap_open)>1e-7:
        fail('Cross-section height correction unexpectedly changed finger width')
    q_exact = .05 - (gap_open_after - MIN_INNER_GAP)/2
    qmin = max(0., math.ceil((q_exact-1e-11)/LIMIT_RESOLUTION)*LIMIT_RESOLUTION)
    if not 0 <= qmin < .05:
        fail('No admissible gripper closure within OEM 50mm stroke')
    lim1.set('lower', '%.6f' % qmin)
    lim2.set('upper', '%.6f' % -qmin)
    p_closed = {i: g.finger_points(root,i,(qmin if i==1 else -qmin),cache) for i in (1,2)}
    actual_close = g.bounds(p_closed[high_y],1)[0]-g.bounds(p_closed[low_y],1)[1]
    if actual_close < MIN_INNER_GAP - 1e-7:
        fail('Closed visual STL AABB intersects: %.3fmm' % (actual_close*1000))

    # TCP remains on gripper centerline. Align along *distal tip* coordinate Z.
    joint = root.find("joint[@name='piper_umi_tcp_joint']")
    if joint is not None:
        o = g.origin(joint)
        pos = g.V(o.get('xyz','0 0 0'))
        tips = [g.bounds(g.part_points(root,i,'tpu',OPEN_Q[i],cache),2)[1] for i in (1,2)]
        pos[1] = 0.
        pos[2] = sum(tips)/2
        o.set('xyz',g.fmt(pos))

    def bounds_x_mm(i):
        return [1000*x for x in g.bounds(p[i],0)]
    def to_mm_dict(d):
        return {'right':round(float(d[right])*1000,6), 'left':round(float(d[left])*1000,6)}
    report = {
        'source':str(SRC), 'full_open_chassis_width_mm':142.5,
        'inset_each_side_mm':4.0, 'old_open_outer_width_mm':old_width*1000,
        'new_open_outer_width_mm':width*1000,
        'old_open_inner_gap_mm':old_gap*1000, 'new_open_inner_gap_mm':gap_open_after*1000,
        'closed_inner_AABB_clearance_mm':actual_close*1000,
        'robot_right_link': 'piper_gripper_link%d' % right,
        'robot_left_link': 'piper_gripper_link%d' % left,
        'local_X_dot_robot_UP':x_up,
        'right_left_world_UP_centers_before_mm':to_mm_dict(height_before),
        'right_left_world_UP_centers_after_mm':to_mm_dict(height_after),
        'right_left_height_center_difference_after_mm':delta_h*1000,
        'local_X_shift_mm':to_mm_dict(delta_x),
        'world_UP_shift_mm':{'right':move_up_right*1000, 'left':move_up_left*1000},
        'local_Y_shift_mm':to_mm_dict(delta_y),
        'finger_X_bounds_after_mm': {'right':bounds_x_mm(right), 'left':bounds_x_mm(left)},
        'finger_robot_UP_extents_after_mm': {
            'right':[1000*v for v in up_extent_after[right]],
            'left':[1000*v for v in up_extent_after[left]]},
        'joint_limits_m':{
            'piper_gripper_joint1':[qmin,.05],
            'piper_gripper_joint2':[-.05,-qmin]},
        'note':('Only custom printed assembly moved, not actual carriage bolt holes. '
                'AABB closed clearance is not a validated collision/contact model. '
                'Prior V5/V6 collision and inertia are still provisional. ') }
    root.insert(0, ET.Comment('V7: 4mm/side outer inset, printed X cross-section symmetrically registered; OEM rail untouched.'))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(root).write(str(OUT),encoding='utf-8',xml_declaration=True)
    view = g.fixed_open_urdf(root,OPEN_Q)
    ET.ElementTree(view).write(str(OPEN),encoding='utf-8',xml_declaration=True)
    ET.parse(str(OUT)); ET.parse(str(OPEN))
    assert len([j for j in root.findall('joint') if j.get('type') in ('revolute','continuous','prismatic')]) == 20
    assert len([j for j in view.findall('joint') if j.get('type') in ('revolute','continuous','prismatic')]) == 18
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    print('\n======= AS2+PiPER UMI V7 =======')
    print('Open width: %.3f -> %.3f mm (TARGET 134.500)' % (old_width*1000,width*1000))
    print('Robot right/left height BEFORE: %.3f / %.3f mm' % (height_before[right]*1000,height_before[left]*1000))
    print('Robot right/left height AFTER : %.3f / %.3f mm' % (height_after[right]*1000,height_after[left]*1000))
    print('Robot right UP shift: %+0.3f mm, left UP shift: %+0.3f mm' % (move_up_right*1000,move_up_left*1000))
    span_right = (up_extent_after[right][1]-up_extent_after[right][0])*1000
    span_left = (up_extent_after[left][1]-up_extent_after[left][0])*1000
    print('Finger vertical extents AFTER: right %.3f mm, left %.3f mm' % (span_right, span_left))
    if abs(span_right-span_left) > .5:
        print('WARNING: the two meshes have different robot-Z extents; their centers align but both edges cannot match by translation alone.')
    print('Max open gap %.3f mm; max closed gap %.3f mm' % (gap_open_after*1000,actual_close*1000))
    print('Finger DOF bounds: q1=[%.4f, 0.0500] q2=[-0.0500, %.4f]' % (qmin,-qmin))
    print('ACTUATED (20 DOF):',OUT)
    print('VIEWER OPEN (18 DOF):',OPEN)
    print('MEASUREMENT REPORT:',REPORT)
    print('WARNING: printed-to-carriage offsets are geometric and do not verify hardware mounting holes.')

if __name__=='__main__':
    main()
