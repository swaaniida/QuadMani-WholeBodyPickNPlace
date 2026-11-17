#!/usr/bin/env python3
"""Compare AS2-only, center-mounted PiPER, and PiPER directly above the head.

This creates fixed-mount MuJoCo study models from the source AS2+PiPER V7 model.
No rail or adapter mass is included; all AS2/PiPER link inertias are retained.
"""
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / 'models/mount_workspace'
OUT = HERE / 'results/as2_piper_mount_comparison'
SOURCE = MODEL_DIR / 'profiles_plate_forward.xml'
AS2_MESH = HERE.parent / 'low-level/resources/robots/as2piper/meshes/as2/base_link.STL'
SAMPLE_SEED = 20261109
SAMPLE_COUNT = 12000
VOXEL_M = 0.03
HEAD_FRAME = np.array([0.33470532298088074, 0.0, 0.12405480071902275])
ARM_BASES = {
    'center': np.array([0.005, 0.0, 0.10202847822]),
    # User selected the center of the PiPER base underside on the AS2 top mesh.
    # The PiPER base_link origin is 9.25 mm behind that footprint center in x.
    'head': np.array([0.22435, 0.0, 0.1319]),
}
TRAVEL_ARM = np.array([0.0, 0.76, -0.89, 0.0, 0.45, 0.0])


def build_models():
    src = ET.parse(SOURCE).getroot()
    made = {}
    for variant in ('as2_only', 'as2_piper_center', 'as2_piper_head'):
        root = ET.fromstring(ET.tostring(src))
        root.set('model', variant + '_actual_link_inertia')
        base = root.find("./worldbody/body[@name='base_link']")
        interface = base.find("body[@name='as2_assumed_mount_interface']")
        plate = interface.find(".//body[@name='mount_plate']")
        arm = plate.find("body[@name='piper_base_link']")
        plate.remove(arm)
        base.remove(interface)
        if variant == 'as2_only':
            actuator = root.find('actuator')
            for motor in list(actuator):
                if motor.get('joint', '').startswith('piper_'):
                    actuator.remove(motor)
        else:
            location = ARM_BASES['center' if variant.endswith('center') else 'head']
            arm.set('pos', ' '.join('{:.12g}'.format(float(v)) for v in location))
            base.append(arm)
        output = MODEL_DIR / (variant + '_actual_link_inertia.xml')
        ET.ElementTree(root).write(output, encoding='utf-8', xml_declaration=True)
        made[variant] = output
    return made


def setup(model):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[:7] = [0, 0, 0.35, 1, 0, 0, 0]
    for leg in ('FL', 'FR', 'RL', 'RR'):
        for suffix, value in (('hip', 0.0), ('thigh', 0.8), ('calf', -1.6)):
            data.qpos[int(model.joint(leg + '_' + suffix + '_joint').qposadr)] = value
    if 'piper_joint1' in [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]:
        for i, value in enumerate(TRAVEL_ARM, 1):
            data.qpos[int(model.joint('piper_joint{}'.format(i)).qposadr)] = value
        for name, value in (('piper_gripper_joint1', 0.049), ('piper_gripper_joint2', -0.049)):
            data.qpos[int(model.joint(name).qposadr)] = value
    mujoco.mj_forward(model, data)
    lows = []
    for leg in ('FL', 'FR', 'RL', 'RR'):
        bid = int(model.body(leg + '_foot').id)
        for gid in range(model.ngeom):
            if (model.geom_bodyid[gid] == bid and model.geom_contype[gid]
                    and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE):
                lows.append(float(data.geom_xpos[gid, 2] - model.geom_size[gid, 0]))
    if lows:
        data.qpos[2] += 0.002 - min(lows)
    mujoco.mj_forward(model, data)
    return data


def mass_properties(model, data):
    base_id = int(model.body('base_link').id)
    origin = data.xpos[base_id].copy()
    masses = model.body_mass[1:]
    centers = data.xipos[1:]
    total = float(masses.sum())
    com = (masses[:, None] * centers).sum(axis=0) / total
    inertia_world = np.zeros((3, 3))
    for bid in range(1, model.nbody):
        mass = float(model.body_mass[bid])
        if mass <= 0:
            continue
        R = data.ximat[bid].reshape(3, 3)
        local = R @ np.diag(model.body_inertia[bid]) @ R.T
        r = data.xipos[bid] - com
        inertia_world += local + mass * ((r @ r) * np.eye(3) - np.outer(r, r))
    q = data.xquat[base_id]
    Rbase = np.empty(9)
    mujoco.mju_quat2Mat(Rbase, q)
    Rbase = Rbase.reshape(3, 3)
    Ibase = Rbase.T @ inertia_world @ Rbase
    return {
        'mass_kg': total,
        'com_base_xyz_m': (com - origin).tolist(),
        'inertia_about_com_base_kg_m2': Ibase.tolist(),
        'inertia_principal_about_com_kg_m2': np.linalg.eigvalsh(Ibase).tolist(),
    }


def arm_collision(model, data):
    ground_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'ground'))
    minimum = 0.0
    count = 0
    pairs = {}
    for i in range(data.ncon):
        c = data.contact[i]
        if ground_id in (c.geom1, c.geom2):
            continue
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               int(model.geom_bodyid[c.geom1])) or ''
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               int(model.geom_bodyid[c.geom2])) or ''
        if b1.startswith('piper_') or b2.startswith('piper_'):
            if c.dist < -0.0005:
                count += 1
                minimum = min(minimum, float(c.dist))
                pairs['{} <-> {}'.format(b1, b2)] = pairs.get('{} <-> {}'.format(b1, b2), 0) + 1
    return count, minimum, pairs


def sample_workspace(model, data, rng):
    tcp_id = int(model.body('piper_umi_nominal_tcp').id)
    base_id = int(model.body('base_link').id)
    joint_ids = [int(model.joint('piper_joint{}'.format(i)).id) for i in range(1, 7)]
    qadr = [int(model.jnt_qposadr[j]) for j in joint_ids]
    ranges = model.jnt_range[joint_ids]
    lower, upper = ranges[:, 0] + 0.015, ranges[:, 1] - 0.015
    points = []
    rejected_collision = 0
    rejected_ground = 0
    for q in rng.uniform(lower, upper, size=(SAMPLE_COUNT, 6)):
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        collision, _, _ = arm_collision(model, data)
        if collision:
            rejected_collision += 1
            continue
        point_world = data.xpos[tcp_id].copy()
        if point_world[2] < 0.025:
            rejected_ground += 1
            continue
        points.append((point_world - data.xpos[base_id]).tolist())
    return np.asarray(points), rejected_collision, rejected_ground


def voxel_volume(points, origin):
    if len(points) == 0:
        return 0.0
    keys = np.unique(np.floor((points - origin) / VOXEL_M).astype(np.int32), axis=0)
    return float(len(keys) * VOXEL_M ** 3)


def workspace_metrics(points):
    head = points - HEAD_FRAME
    front = head[:, 0] > 0.0
    voxel_body = voxel_volume(points, np.zeros(3))
    voxel_head = voxel_volume(head, np.zeros(3))
    hull = ConvexHull(points) if len(points) >= 4 else None
    return {
        'accepted_tcp_samples': int(len(points)),
        'front_of_as2_head_samples': int(front.sum()),
        'voxel_volume_base_frame_m3': voxel_body,
        'voxel_volume_head_frame_m3': voxel_head,
        'sample_convex_hull_volume_base_frame_m3': float(hull.volume) if hull else 0.0,
        'tcp_bounds_base_frame_m': [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        'tcp_bounds_head_frame_m': [head.min(axis=0).tolist(), head.max(axis=0).tolist()],
        'max_tcp_distance_from_head_m': float(np.linalg.norm(head, axis=1).max()),
        'head_forward_reach_max_m': float(head[:, 0].max()),
    }


def measure_head_mount_surface(as2_only_path):
    """Ray-sample the original AS2 STL beneath the level PiPER base footprint."""
    model = mujoco.MjModel.from_xml_path(str(as2_only_path))
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    for leg in ('FL', 'FR', 'RL', 'RR'):
        for suffix, value in (('hip', 0.0), ('thigh', 0.8), ('calf', -1.6)):
            data.qpos[int(model.joint(leg + '_' + suffix + '_joint').qposadr)] = value
    mujoco.mj_forward(model, data)
    group = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)
    geomid = np.zeros(1, dtype=np.int32)
    center = np.array([0.2151, 0.0, 0.1319])
    xhalf = (0.0400 - (-0.0585)) / 2.0
    xs = np.linspace(center[0] - xhalf, center[0] + xhalf, 25)
    ys = np.linspace(-0.04, 0.04, 17)
    heights = []
    for x in xs:
        for y in ys:
            distance = mujoco.mj_ray(model, data, np.array([x, y, 1.0]),
                                     np.array([0.0, 0.0, -1.0]), group, 1, -1, geomid)
            if distance >= 0:
                heights.append(1.0 - float(distance))
    heights = np.asarray(heights)
    gap = center[2] - heights
    return {
        'sample_count': int(len(heights)),
        'grid_shape': [int(len(xs)), int(len(ys))],
        'as2_surface_z_min_max_m': [float(heights.min()), float(heights.max())],
        'flat_piper_underside_clearance_min_max_mm': [float(gap.min() * 1000), float(gap.max() * 1000)],
        'interpretation': 'negative is geometric overlap; positive is a gap. This samples visual STL only, not hardware fastening pads or structural surfaces.',
    }


def main():
    os.umask(0o002)
    OUT.mkdir(parents=True, exist_ok=True)
    model_paths = build_models()
    rng = np.random.default_rng(SAMPLE_SEED)
    result = {'source': str(SOURCE), 'study_assumptions': {
        'mount_hardware_mass_kg': 0.0,
        'rail_included': False,
        'mounts_are_verified_hardware_datums': False,
        'mount_position_note': 'center is the prior comparison candidate; head is placed from the user-selected PiPER underside-footprint center on the AS2 base mesh, with y forced to zero',
        'head_selected_underside_center_xyz_base_link_m': [0.2151, 0.0, 0.1319],
        'head_piper_base_link_origin_xyz_base_link_m': ARM_BASES['head'].tolist(),
        'piper_base_underside_center_to_base_link_origin_offset_xyz_m': [-0.00925, 0.0, 0.0],
        'piper_pose_rad': TRAVEL_ARM.tolist(),
        'gripper_open_joints_m': [0.049, -0.049],
        'workspace_sampling': 'identical seeded joint samples, URDF joint ranges, MuJoCo collision proxies and ground filtered',
        'workspace_voxel_size_m': VOXEL_M,
    }, 'variants': {}}
    point_sets = {}
    for name, path in model_paths.items():
        model = mujoco.MjModel.from_xml_path(str(path))
        data = setup(model)
        entry = {
            'model_path': str(path), 'nq_nv_nu': [int(model.nq), int(model.nv), int(model.nu)],
            'mass_properties': mass_properties(model, data),
            'standing_arm_non_ground_penetration_count': arm_collision(model, data)[0],
        }
        if name != 'as2_only':
            points, bad_contact, below_ground = sample_workspace(model, data, rng)
            np.savez_compressed(OUT / (name + '_tcp_workspace.npz'), tcp_base=points,
                                tcp_head=points - HEAD_FRAME)
            entry['workspace'] = workspace_metrics(points)
            entry['workspace_rejected_arm_collision_samples'] = bad_contact
            entry['workspace_rejected_ground_samples'] = below_ground
            point_sets[name] = points
        result['variants'][name] = entry
        print(name, 'mass', entry['mass_properties']['mass_kg'],
              'COM', entry['mass_properties']['com_base_xyz_m'],
              'DOFs', entry['nq_nv_nu'], flush=True)
    result['head_mount_footprint_surface_check'] = measure_head_mount_surface(model_paths['as2_only'])
    center = result['variants']['as2_piper_center']['workspace']
    head = result['variants']['as2_piper_head']['workspace']
    result['workspace_change_head_vs_center'] = {
        'voxel_volume_base_frame_ratio': head['voxel_volume_base_frame_m3'] / center['voxel_volume_base_frame_m3'],
        'head_forward_reach_gain_m': head['head_forward_reach_max_m'] - center['head_forward_reach_max_m'],
        'head_forward_reach_ratio': head['head_forward_reach_max_m'] / center['head_forward_reach_max_m'],
        'warning': 'sampled workspace volume depends on sampling resolution, collision proxies, and joint range; not an exact continuous workspace',
    }
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(OUT / 'workspace_points.npz', **{
        name: points for name, points in point_sets.items()})
    write_report(result)
    render_workspace(point_sets)
    print(json.dumps(result['workspace_change_head_vs_center'], indent=2))


def write_report(result):
    c = result['variants']['as2_only']['mass_properties']
    m = result['variants']['as2_piper_center']['mass_properties']
    h = result['variants']['as2_piper_head']['mass_properties']
    wc = result['variants']['as2_piper_center']['workspace']
    wh = result['variants']['as2_piper_head']['workspace']
    lines = [
        '# AS2 단독·중앙 PiPER·머리 위 PiPER 비교', '',
        'AS2와 PiPER 원본 URDF 링크 질량·관성을 사용했다. 레일/어댑터 질량은 제외했다. 장착 좌표는 기존 연구 MJCF의 후보값이며, 실제 체결면 기준으로 확인되지 않았다.',
        '**중요:** head 위치는 사용자가 지정한 PiPER 밑면 중심 `(0.2151, 0, 0.1319) m`를 반영했다. 베이스 링크 원점은 mesh 바운딩박스 중심 오프셋 `(-0.00925, 0, 0) m`를 보정해 `(0.22435, 0, 0.1319) m`로 설정했다. PiPER 밑면 footprint의 평면 높이를 유지해 STL을 25×17 ray grid로 검사하면 간격/관통이 나타난다(상세값은 JSON). 실제 PiPER 접촉 패드와 AS2 하중 지지부/체결 홀은 별도 검증이 필요하다. 중앙 후보는 이전 비교 후보이다. CoM/관성과 작업공간 결과는 기구 배치 비교용이며 보행 안정성 증거로 사용할 수 없다.',
        '저장소에는 AS2 공식 보행 생성기/균형 제어기가 없고, 확인된 Unitree control plugin은 관절 명령 전달기다. 동적 보행 및 외란 비교는 공식 제어기를 재현할 수 있는 실행 artifact와 검증된 장착 좌표가 확보된 뒤 수행한다.', '',
        '| 구성 | 총 질량 (kg) | CoM x/y/z, base 기준 (m) | 관성 주값 (kg·m²) |', '|---|---:|---:|---:|',
        '| AS2 only | {:.3f} | {:.3f}/{:.3f}/{:.3f} | {} |'.format(c['mass_kg'], *c['com_base_xyz_m'], ', '.join('{:.3f}'.format(v) for v in c['inertia_principal_about_com_kg_m2'])),
        '| AS2 + center PiPER | {:.3f} | {:.3f}/{:.3f}/{:.3f} | {} |'.format(m['mass_kg'], *m['com_base_xyz_m'], ', '.join('{:.3f}'.format(v) for v in m['inertia_principal_about_com_kg_m2'])),
        '| AS2 + head PiPER | {:.3f} | {:.3f}/{:.3f}/{:.3f} | {} |'.format(h['mass_kg'], *h['com_base_xyz_m'], ', '.join('{:.3f}'.format(v) for v in h['inertia_principal_about_com_kg_m2'])), '',
        '| 작업공간 지표 | Center | Head | 변화 |', '|---|---:|---:|---:|',
        '| 샘플 voxel 부피 (m³) | {:.3f} | {:.3f} | {:.2f}x |'.format(wc['voxel_volume_base_frame_m3'], wh['voxel_volume_base_frame_m3'], wh['voxel_volume_base_frame_m3']/wc['voxel_volume_base_frame_m3']),
        '| 머리 기준 최대 전방 도달 (m) | {:.3f} | {:.3f} | {:+.3f} |'.format(wc['head_forward_reach_max_m'], wh['head_forward_reach_max_m'], wh['head_forward_reach_max_m']-wc['head_forward_reach_max_m']),
        '| 머리 앞 TCP 표본 수 | {} | {} | {:+d} |'.format(wc['front_of_as2_head_samples'], wh['front_of_as2_head_samples'], wh['front_of_as2_head_samples']-wc['front_of_as2_head_samples']), '',
        '작업공간은 동일 seed의 관절 샘플을 URDF 관절 범위에서 뽑고, MuJoCo collision 및 지면 아래 TCP를 제외했다. 3 cm voxel 점유 부피는 연속 reachable volume의 근사치이며, 추출 표본 밀도와 collision proxy에 영향을 받는다.', '',
        '- [3D workspace와 AS2 mesh 시각화](./workspace_3d_comparison.png)',
        '- [workspace point cloud 데이터](./workspace_points.npz)',
        '- [질량·관성·workspace 상세값](./summary.json)',
        '- [고정 mount study MJCF](../../models/mount_workspace/)',
    ]
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def render_workspace(point_sets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from verify_mounts import stl_triangles
    triangles = stl_triangles(AS2_MESH)
    if triangles.ndim == 2:
        triangles = triangles.reshape(-1, 3, 3)
    surface = triangles[::max(1, len(triangles)//1600)]
    colors = {'as2_piper_center': '#159b8a', 'as2_piper_head': '#e77727'}
    fig = plt.figure(figsize=(15, 5.8), constrained_layout=True)
    titles = [('as2_only', 'AS2 only'), ('as2_piper_center', 'AS2 + center PiPER'), ('as2_piper_head', 'AS2 + PiPER above head')]
    for index, (name, title) in enumerate(titles, 1):
        ax = fig.add_subplot(1, 3, index, projection='3d')
        body = Poly3DCollection(surface, facecolor='#8793a0', edgecolor='none', alpha=0.17)
        ax.add_collection3d(body)
        ax.scatter([HEAD_FRAME[0]], [HEAD_FRAME[1]], [HEAD_FRAME[2]],
                   marker='*', s=85, color='#c82f49', label='AS2 head reference')
        if name != 'as2_only':
            pts = point_sets[name]
            if len(pts) > 3000:
                pts = pts[np.linspace(0, len(pts)-1, 3000, dtype=int)]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, alpha=0.13,
                       color=colors[name], label='collision-filtered TCP samples')
            p = ARM_BASES['center' if name.endswith('center') else 'head']
            ax.scatter([p[0]], [p[1]], [p[2]], marker='s', s=45,
                       color=colors[name], label='PiPER base')
        ax.set_title(title, pad=10)
        ax.set_xlim(-0.75, 0.9); ax.set_ylim(-0.75, 0.75); ax.set_zlim(-0.3, 1.0)
        ax.set_xlabel('X forward (m)', labelpad=3); ax.set_ylabel('Y (m)', labelpad=3); ax.set_zlabel('Z (m)', labelpad=3)
        ax.view_init(elev=20, azim=-58)
        ax.set_box_aspect((1.65, 1.5, 1.3))
        ax.legend(loc='upper left', fontsize=7)
    fig.suptitle('AS2 frame: PiPER mount relocation and collision-filtered 3D TCP workspace', fontsize=15)
    fig.savefig(OUT / 'workspace_3d_comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
