"""Check initial TCP, foot contact height, joint limits and local asset paths."""
from pathlib import Path
import ast
import json
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parent
URDF = ROOT / 'resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf'


def vector(value):
    return np.fromstring(value, sep=' ')


def rotation(angles):
    roll, pitch, yaw = angles
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def main():
    tree = ET.parse(URDF).getroot()
    pose = {leg + '_' + joint + '_joint': angle
            for leg in ('FL', 'FR', 'RL', 'RR')
            for joint, angle in [('hip', 0.), ('thigh', .8), ('calf', -1.6)]}
    pose.update({'piper_joint'+str(i): angle
                 for i, angle in enumerate([0., 1.1, -.55, 0., 0., 0.], 1)})
    pose.update(piper_gripper_joint1=.049, piper_gripper_joint2=-.049)
    transforms = {'base_link': np.eye(4)}
    pending = list(tree.findall('joint'))
    while pending:
        ready = [j for j in pending if j.find('parent').get('link') in transforms]
        if not ready:
            raise ValueError('URDF joint tree is disconnected')
        for joint in ready:
            origin = joint.find('origin')
            transform = np.eye(4)
            if origin is not None:
                transform[:3, :3] = rotation(vector(origin.get('rpy', '0 0 0')))
                transform[:3, 3] = vector(origin.get('xyz', '0 0 0'))
            motion = np.eye(4)
            angle = pose.get(joint.get('name'), 0.)
            if joint.get('type') != 'fixed':
                limit = joint.find('limit')
                assert float(limit.get('lower')) <= angle <= float(limit.get('upper'))
                axis = vector(joint.find('axis').get('xyz'))
                if joint.get('type') == 'prismatic':
                    motion[:3, 3] = angle * axis
                else:
                    x, y, z = axis
                    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    motion[:3, :3] = np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*cross@cross
            transforms[joint.find('child').get('link')] = transforms[joint.find('parent').get('link')] @ transform @ motion
            pending.remove(joint)
    foot_bottom = min(transforms[leg+'_foot'][2, 3]-.028 for leg in ('FL', 'FR', 'RL', 'RR'))
    arm = transforms['piper_base_link'][:3, 3]
    tcp = transforms['piper_umi_nominal_tcp']
    radius, pitch, yaw = [.427898468676838, .421686284147847, 0.]
    goal = arm + radius*np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])
    position_error = float(np.linalg.norm(goal-tcp[:3, 3]))
    rotation_error = float(np.max(np.abs(rotation([0., -pitch+2.455216137744035, 0.])-tcp[:3, :3])))
    assert position_error < 1e-6 and rotation_error < 1e-5
    assert abs(foot_bottom + .3244069024246581) < 1e-9
    for mesh in tree.iter('mesh'):
        assert (URDF.parent / mesh.get('filename')).is_file(), mesh.get('filename')
    for source in (ROOT/'legged_gym').rglob('*.py'):
        ast.parse(source.read_text())
    result = {'foot_contact_base_height_m': -foot_bottom, 'arm_base_xyz_m': arm.tolist(),
              'initial_tcp_xyz_base_m': tcp[:3, 3].tolist(), 'initial_goal_position_error_m': position_error,
              'initial_goal_rotation_matrix_max_error': rotation_error,
              'physics_rollout_verified': False, 'full_workspace_ik_collision_verified': False}
    (ROOT/'geometry_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
