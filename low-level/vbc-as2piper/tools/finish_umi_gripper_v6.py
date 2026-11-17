#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V6: align UMI fingers in V5, set open OUTER width, and prevent closed overlap.

Source: V5 actuated URDF. Does not touch V1-V5, source mesh, or RL code.
Makes an actuated 20DOF URDF and fixed-open visual-only URDF.

Frames: all target measurements in the *piper_gripper_base* link frame,
not the AS2 world / camera frame. Gripper base Y is the open/close axis;
base Z is the finger longitudinal direction.

Requirements: Python3.8, numpy, actual STL files.
The 6mm inset is interpreted from the OUTER tips/edges of the printed
fingers to the 142.5mm chassis endpoints on both sides at q fully open.
"""
import json
import math
import re
import struct
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'low-level/resources/robots/as2piper'
SRC = ASSET / 'urdf/as2piper_umi_actuated_v5.urdf'
OUT = ASSET / 'urdf/as2piper_umi_actuated_v6.urdf'
OPEN = ASSET / 'urdf/as2piper_umi_open_pose_v6.urdf'
REPORT = ROOT / 'tools/umi_v6_fit_report.json'

BODY_WIDTH = 0.1425     # 142.5mm overall chassis width (physical measurement)
END_INSET = 0.006       # 6mm from each chassis extreme -> printed outer edge
MIN_INNER_GAP = 0.002  # 2mm safety gap at fully closed command (rigid proxy)
LIMIT_RESOLUTION = 0.0001  # joint limit rounding: 0.1mm per carriage


def fail(msg):
    raise RuntimeError(msg)


def V(s):
    return np.asarray([float(x) for x in s.split()], dtype=np.float64)


def fmt(x):
    return ' '.join('%.12g' % float(v) for v in x)


def rotation(rpy):
    r, p, y = map(float, rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.asarray([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp,   cp*sr,             cp*cr],
    ], dtype=np.float64)


def named(root, tag, name):
    found = root.find("%s[@name='%s']" % (tag, name))
    if found is None:
        fail('Missing %s: %s' % (tag, name))
    return found


def read_stl(path):
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from('<I', data, 80)[0]
        if len(data) == 84 + 50*count:
            dt = np.dtype([('normal', '<f4', (3,)),
                           ('vertices', '<f4', (3, 3)),
                           ('attr', '<u2')])
            return np.frombuffer(data, dtype=dt, count=count, offset=84)['vertices'].astype(np.float64).reshape(-1, 3)
    values = re.findall(rb'\bvertex\s+([-+.0-9eE]+)\s+([-+.0-9eE]+)\s+([-+.0-9eE]+)', data)
    if not values:
        fail('Unsupported STL: ' + str(path))
    return np.asarray([[float(x) for x in row] for row in values], dtype=np.float64)


def origin(element):
    node = element.find('origin')
    if node is None:
        fail('Missing origin on ' + element.tag)
    return node


def finger_joint(root, index):
    return named(root, 'joint', 'piper_gripper_joint%d' % index)


def finger_link(root, index):
    return named(root, 'link', 'piper_gripper_link%d' % index)


def joint_transform(root, index, q):
    j = finger_joint(root, index)
    o = origin(j)
    R = rotation(V(o.get('rpy', '0 0 0')))
    axis = V(j.find('axis').get('xyz'))
    axis_world = R @ axis
    pos = V(o.get('xyz', '0 0 0')) + q * axis_world
    return R, pos, axis_world


def part_points(root, index, name, q, cache):
    Rj, tj, _ = joint_transform(root, index, q)
    part = named(finger_link(root, index), 'visual', 'umi_' + name)
    o = origin(part)
    Rv = rotation(V(o.get('rpy', '0 0 0')))
    tv = V(o.get('xyz', '0 0 0'))
    mesh = part.find('geometry/mesh')
    if mesh is None:
        fail('Missing UMI mesh %s on finger %d' % (name, index))
    path = (ASSET / 'urdf' / mesh.get('filename')).resolve()
    if not path.is_file():
        fail('Missing actual STL: ' + str(path))
    if path not in cache:
        cache[path] = read_stl(path)
    s = V(mesh.get('scale', '1 1 1'))
    raw = cache[path] * s
    return (Rj @ (Rv @ raw.T + tv[:, None]) + tj[:, None]).T


def finger_points(root, index, q, cache):
    return np.concatenate([part_points(root, index, k, q, cache) for k in ('pla', 'tpu')], axis=0)


def move_printed_assembly(root, index, delta_base):
    """Move the whole printed-finger assembly in gripper_base frame.
    Prismatic joints, rails, and gripper chassis remain untouched.
    Collision proxies and *estimated* inertia center follow the printed parts.
    """
    R, _, _ = joint_transform(root, index, 0.0)
    shift_local = R.T @ np.asarray(delta_base, dtype=float)
    link = finger_link(root, index)
    elems = list(link.findall('visual')) + list(link.findall('collision')) + list(link.findall('inertial'))
    if not elems:
        fail('No printed elements to move on finger %d' % index)
    for el in elems:
        o = origin(el)
        o.set('xyz', fmt(V(o.get('xyz', '0 0 0')) + shift_local))


def bounds(points, axis):
    return (float(points[:, axis].min()), float(points[:, axis].max()))


def fixed_open_urdf(actuated, qopen):
    static = deepcopy(actuated)
    for i in (1, 2):
        j = finger_joint(static, i)
        o = origin(j)
        R = rotation(V(o.get('rpy', '0 0 0')))
        axis = V(j.find('axis').get('xyz'))
        o.set('xyz', fmt(V(o.get('xyz')) + qopen[i] * (R @ axis)))
        j.set('type', 'fixed')
        for ch in list(j):
            if ch.tag in ('axis', 'limit', 'mimic', 'dynamics', 'safety_controller'):
                j.remove(ch)
    return static


def main():
    if not SRC.exists():
        fail('V5 source not found: ' + str(SRC))
    root = ET.parse(str(SRC)).getroot()
    dofs = [j for j in root.findall('joint') if j.get('type') in ('revolute', 'continuous', 'prismatic')]
    if len(dofs) != 20:
        fail('V5 source must contain 20 active DOFs, got %d' % len(dofs))
    j1, j2 = finger_joint(root, 1), finger_joint(root, 2)
    if j1.get('type') != 'prismatic' or j2.get('type') != 'prismatic':
        fail('Official two gripper joints must stay prismatic')
    l1, l2 = j1.find('limit'), j2.find('limit')
    o1 = float(l1.get('upper'))
    o2 = float(l2.get('lower'))
    if abs(o1 + o2) > 1e-6 or abs(o1 - .05) > .001:
        fail('Unexpected OEM full-open limits: q1=%f q2=%f. Refusing to guess physical stroke.' % (o1, o2))
    q_open = {1: o1, 2: o2}
    a1 = joint_transform(root, 1, 0)[2]
    a2 = joint_transform(root, 2, 0)[2]
    if not (np.dot(a1, [0, -1, 0]) > .999 and np.dot(a2, [0, -1, 0]) > .999):
        fail('Unexpected slide axes (gripper_base frame): %s, %s. Stop and inspect.' % (fmt(a1), fmt(a2)))

    cache = {}
    p_before = {i: finger_points(root, i, q_open[i], cache) for i in (1, 2)}
    before = {i: {'y': bounds(p_before[i], 1), 'z': bounds(p_before[i], 2)} for i in (1, 2)}

    # Stage 1: use center of each finger's full STL Z extent, not a camera perspective.
    center_z = {i: .5*sum(before[i]['z']) for i in (1, 2)}
    mean_z = .5*(center_z[1] + center_z[2])
    shift_z = {i: mean_z - center_z[i] for i in (1, 2)}
    for i in (1, 2):
        move_printed_assembly(root, i, [0, 0, shift_z[i]])

    # Stage 2: move printed assemblies only (not real slider stroke) to match
    # requested *outer edges* at full-open q. These offsets require mounting
    # hardware validation; report the changes explicitly.
    p = {i: finger_points(root, i, q_open[i], cache) for i in (1, 2)}
    yc = {i: .5*sum(bounds(p[i], 1)) for i in (1, 2)}
    left, right = sorted((1, 2), key=lambda i: yc[i])
    target_half = BODY_WIDTH / 2 - END_INSET
    if target_half <= 0:
        fail('Invalid chassis/inset configuration')
    left_outer = bounds(p[left], 1)[0]
    right_outer = bounds(p[right], 1)[1]
    shift_y = {left: -target_half - left_outer, right: target_half - right_outer}
    for i in (1, 2):
        move_printed_assembly(root, i, [0, shift_y[i], 0])

    p_open = {i: finger_points(root, i, q_open[i], cache) for i in (1, 2)}
    width = bounds(p_open[right], 1)[1] - bounds(p_open[left], 1)[0]
    gap_open = bounds(p_open[right], 1)[0] - bounds(p_open[left], 1)[1]
    if abs(width - (BODY_WIDTH - 2*END_INSET)) > 1e-7:
        fail('Failed full-open outer-width target; got %.5f m' % width)
    if gap_open < MIN_INNER_GAP:
        fail('At MAX open the fingers already overlap: gap %.2f mm' % (gap_open*1e3))

    # Stage 3: with symmetric prismatic commands [q, -q], clearance is
    # gap(q)=gap_full_open - 2*(q_open - q) for these verified axis directions.
    qmin_exact = o1 - (gap_open - MIN_INNER_GAP) / 2
    qmin = max(0.0, math.ceil((qmin_exact-1e-11) / LIMIT_RESOLUTION) * LIMIT_RESOLUTION)
    if qmin >= o1:
        fail('No usable closing stroke after adjusting printed geometry')
    l1.set('lower', '%.6f' % qmin)
    l2.set('upper', '%.6f' % (-qmin))

    p_close = {1: finger_points(root, 1, qmin, cache), 2: finger_points(root, 2, -qmin, cache)}
    actual_min_gap = bounds(p_close[right], 1)[0] - bounds(p_close[left], 1)[1]
    if actual_min_gap + 1e-8 < MIN_INNER_GAP:
        fail('Overlap after limiting close stroke: %.4f mm' % (actual_min_gap*1000))
    zc = {i: .5*sum(bounds(p_open[i], 2)) for i in (1, 2)}
    if abs(zc[1] - zc[2]) > 1e-7:
        fail('Z centers not aligned')

    # TCP: center of the two distal TPU edges (only when gripper is open).
    tip_z = [bounds(part_points(root, i, 'tpu', q_open[i], cache), 2)[1] for i in (1,2)]
    tcp_joint = root.find("joint[@name='piper_umi_tcp_joint']")
    if tcp_joint is not None:
        to = origin(tcp_joint)
        pt = V(to.get('xyz'))
        pt[1] = 0.0
        pt[2] = sum(tip_z) / 2
        to.set('xyz', fmt(pt))

    note = ('V6: printed UMI meshes center-aligned; open external span %.1f mm; '
            'minimum AABB inner clearance %.1f mm; mounted location must be checked.'
            % (1000*width, 1000*actual_min_gap))
    root.insert(0, ET.Comment(note))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(str(OUT), encoding='utf-8', xml_declaration=True)
    fixed = fixed_open_urdf(root, q_open)
    ET.ElementTree(fixed).write(str(OPEN), encoding='utf-8', xml_declaration=True)
    ET.parse(str(OUT)); ET.parse(str(OPEN))
    assert len([j for j in root.findall('joint') if j.get('type') in ('prismatic', 'revolute', 'continuous')]) == 20
    assert len([j for j in fixed.findall('joint') if j.get('type') in ('prismatic', 'revolute', 'continuous')]) == 18

    report = {
        'source': str(SRC),
        'unit': 'mm except joint_q_limits_m and paths',
        'chassis_width_mm': BODY_WIDTH*1000,
        'requested_outer_inset_each_side_mm': END_INSET*1000,
        'full_open_outer_width_mm': width*1000,
        'full_open_inner_gap_mm': gap_open*1000,
        'min_closed_inner_gap_AABB_mm': actual_min_gap*1000,
        'old_finger_z_ranges_mm': {str(i): [x*1000 for x in before[i]['z']] for i in (1,2)},
        'z_shift_mm': {str(i): shift_z[i]*1000 for i in (1,2)},
        'y_shift_printed_geometry_mm': {str(i): shift_y[i]*1000 for i in (1,2)},
        'joint_q_limits_m': {
            'piper_gripper_joint1': [qmin, o1],
            'piper_gripper_joint2': [o2, -qmin]},
        'outer_reference': 'outermost printed-mesh Y at max open; chassis centered Y=0 in piper_gripper_base',
        'IMPORTANT': 'Geometry-fit candidate. Changing the print mount Y offsets does NOT change OEM rail/slider stroke or screw locations; check PLA holes vs moving carriage! Min-gap uses lateral STL AABB, not full dynamic mesh collision. Do not run contact RL without corrected printed collision/inertia and hardware fit.'}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print('\n=== UMI V6 automatic geometric fit ===')
    for i in (1, 2):
        print('Finger%d: old Z=[%.2f, %.2f] mm, Z shift=%+.2f mm, lateral shift=%+.2f mm' %
              (i, before[i]['z'][0]*1000, before[i]['z'][1]*1000, shift_z[i]*1000, shift_y[i]*1000))
    print('Full-open OUTER finger width : %.2f mm (target %.2f)' % (width*1000, (BODY_WIDTH-2*END_INSET)*1000))
    print('Full-open INNER gap         : %.2f mm' % (gap_open*1000))
    print('Min-close INNER gap (AABB)  : %.2f mm (required >= %.2f)' % (actual_min_gap*1000, MIN_INNER_GAP*1000))
    print('Finger1 q range             : [%.4f, %.4f] m' % (qmin, o1))
    print('Finger2 q range             : [%.4f, %.4f] m' % (o2, -qmin))
    print('20 DOF trainable URDF       :', OUT)
    print('18 DOF open visual-only URDF:', OPEN)
    print('Report:', REPORT)
    if max(abs(x) for x in shift_y.values()) > 0.002:
        print('WARNING: lateral bracket offsets exceed 2mm. PLA fastener alignment to actual OEM carriage MUST be rechecked before hardware or contact RL.')
    print('WARNING: Printed collision/inertial proxies inherited from V5; recalculate from physical parts before contact RL.')

if __name__ == '__main__':
    main()
