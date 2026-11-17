"""Render actual policy evaluation states; never integrate or script motion."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('evaluation');p.add_argument('--seconds',type=float,default=3.);a=p.parse_args();out=Path(a.evaluation)
    trace=np.load(str(out/'trajectories.npz'));meta=json.loads((out/'metadata.json').read_text())
    xml=ET.parse(ROOT/'resources/commands/plate_forward.xml').getroot();asset=xml.find('asset')
    for mesh in asset.findall('mesh'):
        mesh.set('file',str((ROOT/'resources/commands'/mesh.get('file')).resolve()))
    ET.SubElement(asset,'texture',name='gait_grid',type='2d',builtin='checker',rgb1='.22 .27 .30',rgb2='.35 .4 .43',width='512',height='512')
    ET.SubElement(asset,'material',name='gait_floor',texture='gait_grid',texrepeat='6 6',reflectance='0')
    xml.find("./worldbody/geom[@name='ground']").set('material','gait_floor')
    m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));d=mujoco.MjData(m)
    m.vis.headlight.ambient[:]=[.6,.6,.6];m.vis.headlight.diffuse[:]=[.8,.8,.8]
    renderer=mujoco.Renderer(m,height=288,width=512);camera=mujoco.MjvCamera();camera.distance=1.9;camera.elevation=-20
    ids=[int(m.joint(n).qposadr) for n in meta['dof_names']]
    def frame(i,env,title):
        state=trace['root'][i,env];d.qpos[:3]=state[:3];d.qpos[3:7]=state[[6,3,4,5]];d.qpos[ids]=trace['q'][i,env]
        mujoco.mj_forward(m,d);R=d.xmat[1].reshape(3,3)
        camera.azimuth=120+np.rad2deg(np.arctan2(R[1,0],R[0,0]));camera.lookat[:]=state[:3]+[.08,0,.15]
        renderer.update_scene(d,camera=camera);im=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,512,43),fill=(245,245,245));command=trace['commands'][i,env]
        draw.text((7,5),title+' | RL policy '+str(int(trace['iteration'][i])),fill='black')
        draw.text((7,22),'vx %.2f / %.2f m/s | yaw %.2f / %.2f rad/s'%(trace['vx'][i,env],command[0],trace['yaw_rate'][i,env],command[2]),fill='black')
        return np.asarray(im)
    end=len(trace['step']);start=max(0,end-int(25*a.seconds))
    with imageio.get_writer(str(out/'learned_gait.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=1) as w:
        for i in range(start,end):
            joined=np.concatenate([frame(i,1,'Forward'),frame(i,3,'Forward + turn')],axis=1);w.append_data(joined)
            if i==(start+end)//2:Image.fromarray(joined).save(str(out/'learned_gait.jpg'),quality=94)
    renderer.close();print('Rendered actual PhysX policy states; no physical reintegration or scripted joint motion.')


if __name__=='__main__':main()
