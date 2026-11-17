#!/usr/bin/env python3
"""Export static AS2 and articulated PiPER preview meshes from MuJoCo transforms."""
import json
import os
from pathlib import Path
import struct
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / 'models/mount_workspace/as2_piper_head_actual_link_inertia.xml'
OUT = HERE
ARM_Q = [0.0, 0.76, -0.89, 0.0, 0.45, 0.0]


def setup(model):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[:7] = [0, 0, 0.35, 1, 0, 0, 0]
    for leg in ('FL', 'FR', 'RL', 'RR'):
        for suffix, value in (('hip', 0.0), ('thigh', 0.8), ('calf', -1.6)):
            data.qpos[int(model.joint(leg + '_' + suffix + '_joint').qposadr)] = value
    for i, value in enumerate(ARM_Q, 1):
        data.qpos[int(model.joint('piper_joint{}'.format(i)).qposadr)] = value
    for name, value in (('piper_gripper_joint1', .049), ('piper_gripper_joint2', -.049)):
        data.qpos[int(model.joint(name).qposadr)] = value
    mujoco.mj_forward(model, data)
    lows = []
    for leg in ('FL', 'FR', 'RL', 'RR'):
        bid = int(model.body(leg + '_foot').id)
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == bid and model.geom_contype[gid] and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
                lows.append(float(data.geom_xpos[gid, 2] - model.geom_size[gid, 0]))
    data.qpos[2] += .002 - min(lows)
    mujoco.mj_forward(model, data)
    return data


def main():
    os.umask(0o002)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = setup(model)
    base_id = int(model.body('base_link').id)
    arm_id = int(model.body('piper_base_link').id)
    base_origin = data.xpos[base_id].copy()
    arm_origin = data.xpos[arm_id].copy()
    arm_R = data.xmat[arm_id].reshape(3, 3).copy()
    position_chunks = []
    index_chunks = []
    manifest = {'model': str(MODEL), 'coordinate_frame': 'AS2 base_link; x forward, y left, z up', 'base_origin_world_xyz_m': data.xpos[base_id].tolist(), 'arm_pose_rad': ARM_Q, 'meshes': []}
    pos_bytes = 0
    index_bytes = 0
    all_ids = []
    def in_arm_subtree(body_id):
        current = int(body_id)
        while current > 0:
            if current == arm_id:
                return True
            current = int(model.body_parentid[current])
        return False
    for gid in range(model.ngeom):
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_group[gid]) != 2:
            continue
        body_id = int(model.geom_bodyid[gid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ''
        group = 'arm' if in_arm_subtree(body_id) else 'robot'
        # Only the AS2 robot and the PiPER/UMI visual subtree are part of the viewer.
        if group == 'robot' and body_id == 0:
            continue
        mesh_id = int(model.geom_dataid[gid])
        va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(model.mesh_vert[va:va+vn], dtype=np.float64)
        faces = np.asarray(model.mesh_face[fa:fa+fn], dtype=np.uint32)
        Rgeom = data.geom_xmat[gid].reshape(3, 3)
        world = vertices @ Rgeom.T + data.geom_xpos[gid]
        if group == 'robot':
            points = world - base_origin
        else:
            points = (world - arm_origin) @ arm_R
        points = np.asarray(points, dtype='<f4')
        faces = np.asarray(faces.reshape(-1), dtype='<u4')
        pb = points.tobytes()
        ib = faces.tobytes()
        position_chunks.append(pb)
        index_chunks.append(ib)
        rgba = [float(v) for v in model.geom_rgba[gid]]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ('geom_' + str(gid))
        manifest['meshes'].append({
            'name': name, 'group': group, 'positionOffset': pos_bytes,
            'vertexCount': int(len(points)), 'indexOffset': index_bytes,
            'indexCount': int(len(faces)), 'rgba': rgba,
        })
        pos_bytes += len(pb)
        index_bytes += len(ib)
        all_ids.append((name, group, points.min(axis=0).tolist(), points.max(axis=0).tolist()))
    (OUT / 'preview_positions.bin').write_bytes(b''.join(position_chunks))
    (OUT / 'preview_indices.bin').write_bytes(b''.join(index_chunks))
    manifest['binary_layout'] = 'separate position and index streams; positions are float32 xyz, indices are uint32; offsets are byte offsets'
    manifest['position_stream_bytes'] = pos_bytes
    manifest['index_stream_bytes'] = index_bytes
    manifest['meshes'] = manifest['meshes']
    manifest['arm_base_mesh_local_aabb_m'] = [[-0.0585, -0.04, 0.0], [0.04, 0.04, 0.086]]
    manifest['base_underside_center_local_xyz_m'] = [-0.00925, 0.0, 0.0]
    manifest['selected_underside_center_xyz_base_link_m'] = [0.2151, 0.0, 0.1319]
    manifest['selected_piper_base_link_origin_xyz_base_link_m'] = [0.22435, 0.0, 0.1319]
    manifest['mount_pose'] = 'upright; no alignment to local surface normal'
    manifest['warning'] = 'Placement preview only. A click defines PiPER base underside-footprint center. Confirm holes, structural support, collision, cable clearance, mass and inertial effects before fabrication or dynamics.'
    (OUT / 'preview_geometry.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('meshes', len(manifest['meshes']), 'binary MB', round((pos_bytes+index_bytes)/1e6, 2), 'arm root', arm_origin.tolist())
    for row in all_ids:
        if row[1] == 'arm' and 'base_link' in row[0]: print('arm base mesh local bounds', row)


if __name__ == '__main__':
    main()
