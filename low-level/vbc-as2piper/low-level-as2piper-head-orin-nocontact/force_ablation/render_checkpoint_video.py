"""Render recorded PhysX states only; MuJoCo does not integrate physics here."""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'mujoco_mount_compare'))
from build_mount_scenes import Converter, origin_attrs


class ExactURDF(Converter):
    # Preserve every original fixed transform, including the PiPER mounting joint.
    def make_link(self, name, parent, pose, via=None):
        body = ET.SubElement(parent, 'body', {'name': name, **pose})
        if via is not None and via.get('type') != 'fixed':
            ET.SubElement(body, 'joint', self.joint_motor(via))
        link = self.links[name]
        self.add_inertial(body, link)
        for i, visual in enumerate(link.findall('visual')):
            self.add_geom(body, visual, True, name, i)
        for joint in self.children.get(name, []):
            self.make_link(joint.find('child').get('link'), body, origin_attrs(joint.find('origin')), joint)
        return body


def build(folder, meta, j):
    xml_path = folder/('render_%d.xml' % j)
    c = ExactURDF(Path(meta['urdf']), xml_path, {'time_step_s': .002, 'gravity_m_s2': -9.81}, 'replay')
    body = c.make_link('base_link', c.world, {'pos': '0 0 0'})
    ET.SubElement(body, 'freejoint', name='base_free')
    ET.SubElement(c.root.find('visual'), 'global', offwidth='640', offheight='360')
    if meta['terrain'] == 'rough':
        c.world.remove(c.world.find("geom[@name='ground']"))
        ET.SubElement(c.assets, 'mesh', name='actual_physx_terrain', file=str((folder/('terrain_%d.obj' % j)).resolve()))
        ET.SubElement(c.world, 'geom', name='terrain', type='mesh', mesh='actual_physx_terrain', rgba='.53 .58 .60 1', contype='0', conaffinity='0')
    for mesh in c.assets.findall('mesh'):
        mesh.set('file', str((folder/mesh.get('file')).resolve()))
    ET.ElementTree(c.root).write(xml_path, encoding='utf-8', xml_declaration=True)
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    m.vis.headlight.ambient[:] = [.65]*3
    m.vis.headlight.diffuse[:] = [.75]*3
    d = mujoco.MjData(m)
    qids = [int(m.joint(n).qposadr[0]) for n in meta['dof_names']]
    cam = mujoco.MjvCamera()
    cam.distance = 1.75
    cam.azimuth = 130
    cam.elevation = -22
    return m, d, mujoco.Renderer(m, height=360, width=640), cam, qids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True)
    a = p.parse_args()
    folder = Path(a.folder).resolve()
    meta = json.loads((folder/'metadata.json').read_text())
    z = np.load(folder/'states.npz')
    font_path = PROJECT/'unitree_rl_mjlab/reports/stability_flat_force_comparison_20261110/assets/NotoSansKR-Medium.ttf'
    font = ImageFont.truetype(str(font_path), 20)
    small = ImageFont.truetype(str(font_path), 17)
    labels = ['정지: yaw 흔들림 확인', '전진: 목표 대비 과속 확인', '회전: yaw 속도 추종', '전진 + 회전: 복합 동작 / EE 추종']
    models = [build(folder, meta, j) for j in range(4)]
    selected = np.flatnonzero((z['time'] >= 2.) & (z['time'] < 10.))
    max_fk_error = 0.
    output = folder.parent/('B_10000_'+meta['terrain']+'_features.mp4')
    with imageio.get_writer(str(output), fps=25, codec='libx264', quality=8, macro_block_size=1) as writer:
        for count, i in enumerate(selected):
            canvas = Image.new('RGB', (1280, 1080), '#f5f7fa')
            for j, (m, d, renderer, cam, qids) in enumerate(models):
                root = z['root'][i, j]
                d.qpos[:3] = root[:3]
                d.qpos[3:7] = root[[6, 3, 4, 5]]
                d.qpos[qids] = z['q'][i, j]
                mujoco.mj_forward(m, d)
                fk_error = np.linalg.norm(d.body('piper_umi_nominal_tcp').xpos-z['ee'][i, j])
                max_fk_error = max(max_fk_error, float(fk_error))
                assert fk_error < .001, 'URDF replay TCP mismatch: '+str(fk_error)
                cam.lookat[:] = root[:3]+[.1, 0, .13]
                renderer.update_scene(d, camera=cam)
                for key, radius, color in [('target', .013, [1., .18, .1, .85]), ('ee', .009, [.05, .95, .8, 1.])]:
                    g = renderer.scene.geoms[renderer.scene.ngeom]
                    renderer.scene.ngeom += 1
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius]*3), z[key][i, j], np.eye(3).ravel(), np.array(color, dtype=np.float32))
                panel = Image.new('RGB', (640, 540), '#f5f7fa')
                panel.paste(Image.fromarray(renderer.render().copy()), (0, 78))
                draw = ImageDraw.Draw(panel)
                title = ('평지' if meta['terrain']=='flat' else '요철 지면')+' | '+labels[j]
                draw.text((10, 3), title, font=font, fill='#132b40')
                cmd = meta['commands'][j]
                ee = np.linalg.norm(z['target'][i, j]-z['ee'][i, j])*100
                draw.text((10, 31), 'vx 목표 %.2f / 실제 %.2f m/s  |  yaw 목표 %.2f / 실제 %.2f rad/s' % (cmd[0], z['velocity'][i,j],cmd[2],z['yaw'][i,j]),font=small, fill='#253c51')
                draw.text((10, 54), 'EE 오차 %.2f cm | 빨강: 목표 / 청록: 실제 | t=%.2fs' % (ee,z['time'][i]),font=small,fill='#253c51')
                is_yaw = j in [0, 2]
                data = z['yaw'][:, j] if is_yaw else z['velocity'][:, j]
                target = cmd[2] if is_yaw else cmd[0]
                limit = 1.3 if is_yaw else 1.0
                low = -1.3 if is_yaw else -.3
                start = max(selected[0], i-74)
                history = data[start:i+1]
                def yy(v): return 526-62*(np.clip(v,low,limit)-low)/(limit-low)
                draw.text((10, 439), ('yaw 속도' if is_yaw else '전진 속도')+' 최근 3초 | 회색: 목표 / 파랑: 실제',font=small,fill='#253c51')
                draw.line([(12, yy(target)), (626, yy(target))], fill='#98a3af', width=2)
                if len(history)>1:
                    draw.line([(12+614*k/max(1,len(history)-1), yy(v)) for k,v in enumerate(history)],fill='#1479bc',width=3)
                canvas.paste(panel, ((j%2)*640, (j//2)*540))
            writer.append_data(np.asarray(canvas))
            if count == len(selected)//2:
                canvas.save(folder.parent/('B_10000_'+meta['terrain']+'_features.jpg'))
            if count % 50 == 0:
                print(meta['terrain'], 'frame', count, '/', len(selected), flush=True)
    for _, _, renderer, _, _ in models:
        renderer.close()
    (folder/'render_validation.json').write_text(json.dumps({'video':str(output), 'frames':len(selected), 'fps':25, 'max_tcp_fk_error_m':max_fk_error, 'physics':'Recorded PhysX policy rollout; MuJoCo kinematic replay only; no re-simulation', 'warmup_removed_s':2., 'replica_selection':meta['env_ids']},indent=2)+'\n')
    print('Saved', output, flush=True)


if __name__ == '__main__':
    main()
