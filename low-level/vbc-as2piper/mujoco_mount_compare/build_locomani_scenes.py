#!/usr/bin/env python3
"""Add an assumed wrist camera and a grasp object to the verified mount scenes."""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / 'models/mesh_seated'
OUTPUT_DIR = HERE / 'models/locomani'
CFG = json.loads((HERE / 'locomani_config.json').read_text(encoding='utf-8'))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant in ('profiles_plate', 'plate_only'):
        root = ET.parse(str(SOURCE_DIR / (variant + '.xml'))).getroot()
        root.set('model', 'as2piper_locomani_' + variant)
        gripper = root.find(".//body[@name='piper_gripper_base']")
        if gripper is None:
            raise RuntimeError('Missing piper_gripper_base')
        # Gripper local -X is the dorsal (upper) side; finger separation is Y.
        position = CFG['wrist_camera_position_in_gripper_base_m']
        target = CFG['wrist_camera_lookat_in_gripper_base_m']
        dx, _, dz = [b-a for a,b in zip(position,target)]
        norm = math.hypot(dx,dz)
        camera_up = [dz/norm, 0.0, -dx/norm]
        ET.SubElement(
            gripper,
            'camera',
            name='wrist_camera',
            pos=' '.join(str(x) for x in CFG['wrist_camera_position_in_gripper_base_m']),
            xyaxes='0 1 0 {} {} {}'.format(*camera_up),
            fovy=str(CFG['wrist_camera_vertical_fov_deg']),
        )
        world = root.find('worldbody')
        obj = ET.SubElement(world, 'body', name='target_object', pos=' '.join(str(x) for x in CFG['object_initial_xyz_m']))
        ET.SubElement(obj, 'freejoint', name='target_object_free')
        ET.SubElement(
            obj,
            'geom',
            name='target_object_geom',
            type='sphere',
            size=str(CFG['object_radius_m']),
            mass=str(CFG['object_mass_kg']),
            rgba='0.92 0.18 0.08 1',
            friction='1.0 0.01 0.001',
            contype='1',
            conaffinity='1',
        )
        ET.SubElement(obj, 'site', name='target_object_center', type='sphere', size='0.006', rgba='1 0.9 0.1 1')

        ET.ElementTree(root).write(str(OUTPUT_DIR / (variant + '.xml')), encoding='utf-8', xml_declaration=True)
        print('Generated', OUTPUT_DIR / (variant + '.xml'))


if __name__ == '__main__':
    main()
