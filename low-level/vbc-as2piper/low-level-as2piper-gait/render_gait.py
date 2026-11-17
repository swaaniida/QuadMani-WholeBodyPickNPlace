"""Render actual policy evaluation states; never integrate or script motion."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('evaluation');p.add_argument('--seconds',type=float,default=3.)
    p.add_argument('--start-seconds',type=float)
    p.add_argument('--grid',action='store_true');p.add_argument('--stride',type=int,default=1)
    p.add_argument('--fixed-azimuth',action='store_true')
    a=p.parse_args();out=Path(a.evaluation)
    if a.stride<1:p.error('--stride must be positive')
    trace=np.load(str(out/'trajectories.npz'));meta=json.loads((out/'metadata.json').read_text())
    measured=json.loads((out/'policy_evaluation.json').read_text())
    samples=np.load(str(out/'policy_evaluation.npz'))['samples']
    rough='Terrain=trimesh' in measured['conditions']
    if rough and not all((out/('terrain_panel_%d.obj'%i)).exists() for i in range(4)):
        raise ValueError('Actual terrain exports are required for a rough-ground replay.')
    xml=ET.parse(ROOT/'resources/commands/plate_forward.xml').getroot();asset=xml.find('asset')
    for mesh in asset.findall('mesh'):
        mesh.set('file',str((ROOT/'resources/commands'/mesh.get('file')).resolve()))
    ET.SubElement(asset,'texture',name='gait_grid',type='2d',builtin='checker',rgb1='.22 .27 .30',rgb2='.35 .4 .43',width='512',height='512')
    ET.SubElement(asset,'material',name='gait_floor',texture='gait_grid',texrepeat='6 6',reflectance='0')
    ground=xml.find("./worldbody/geom[@name='ground']")
    if rough:
        for geom in xml.findall('.//geom'):
            geom.set('group','5' if geom.get('group')=='3' else '0')
        xml.find('worldbody').remove(ground)
        for panel in range(4):
            ET.SubElement(asset,'mesh',name='terrain_%d'%panel,file=str((out/('terrain_panel_%d.obj'%panel)).resolve()))
            ET.SubElement(xml.find('worldbody'),'geom',name='terrain_%d'%panel,type='mesh',mesh='terrain_%d'%panel,
                          rgba='.43 .47 .42 1',group=str(panel+1),contype='0',conaffinity='0')
    else:ground.set('material','gait_floor')
    if 'box' in trace:
        box=ET.SubElement(xml.find('worldbody'),'body',name='evaluation_box')
        ET.SubElement(box,'freejoint',name='evaluation_box_joint')
        half=meta.get('box_size_m',.1)/2
        ET.SubElement(box,'geom',type='box',size=' '.join([str(half)]*3),rgba='.9 .35 .12 1',mass='1')
    m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));d=mujoco.MjData(m)
    m.vis.headlight.ambient[:]=[.6,.6,.6];m.vis.headlight.diffuse[:]=[.8,.8,.8]
    renderer=mujoco.Renderer(m,height=288,width=512);camera=mujoco.MjvCamera();camera.distance=1.9;camera.elevation=-20
    ids=[int(m.joint(n).qposadr) for n in meta['dof_names']]
    def frame(i,env,title):
        state=trace['root'][i,env];d.qpos[:3]=state[:3];d.qpos[3:7]=state[[6,3,4,5]];d.qpos[ids]=trace['q'][i,env]
        if 'box' in trace:
            b=trace['box'][i,env];adr=int(m.joint('evaluation_box_joint').qposadr)
            d.qpos[adr:adr+3]=b[:3];d.qpos[adr+3:adr+7]=b[[6,3,4,5]]
        mujoco.mj_forward(m,d);R=d.xmat[1].reshape(3,3)
        camera.azimuth=120 if a.fixed_azimuth else 120+np.rad2deg(np.arctan2(R[1,0],R[0,0]));camera.lookat[:]=state[:3]+[.08,0,.15]
        option=mujoco.MjvOption()
        if rough:
            option.geomgroup[:]=0;option.geomgroup[0]=1;option.geomgroup[env+1]=1
        renderer.update_scene(d,camera=camera,scene_option=option);im=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,512,43),fill=(245,245,245));command=trace['commands'][i,env]
        step=int(trace['step'][i]);env_id=meta['video_ids'][env]
        recent=samples[max(0,step-2*a.stride):step,env_id]
        reset_label=' | episode reset' if (recent[:,2]>.5).any() else ''
        draw.text((7,5),title+' | RL policy '+str(int(trace['iteration'][i]))+reset_label,fill='black')
        draw.text((7,22),'vx %.2f / %.2f m/s | yaw %.2f / %.2f rad/s'%(trace['vx'][i,env],command[0],trace['yaw_rate'][i,env],command[2]),fill='black')
        return np.asarray(im)
    end=len(trace['step']);start=max(0,end-int(25*a.seconds))
    if a.start_seconds is not None:
        start=max(0,int(25*a.start_seconds));end=min(end,start+int(25*a.seconds))
        if start>=end:raise ValueError('Requested video interval has no frames')
    with imageio.get_writer(str(out/'learned_gait.mp4'),fps=25/a.stride,codec='libx264',quality=8,macro_block_size=1) as w:
        for i in range(start,end,a.stride):
            if a.grid:
                upper=np.concatenate([frame(i,0,'Stop'),frame(i,1,'Forward')],axis=1)
                lower=np.concatenate([frame(i,2,'Turn'),frame(i,3,'Forward + turn')],axis=1)
                joined=np.concatenate([upper,lower],axis=0)
            else:joined=np.concatenate([frame(i,1,'Forward'),frame(i,3,'Forward + turn')],axis=1)
            w.append_data(joined)
            if i==start+((end-start)//(2*a.stride))*a.stride:Image.fromarray(joined).save(str(out/'learned_gait.jpg'),quality=94)
    renderer.close();print('Rendered actual PhysX policy states; no physical reintegration or scripted joint motion.')


if __name__=='__main__':main()
