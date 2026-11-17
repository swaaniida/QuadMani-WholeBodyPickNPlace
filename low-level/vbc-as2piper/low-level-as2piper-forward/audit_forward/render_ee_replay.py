"""Replay recorded PhysX joint/root states in MuJoCo, without re-simulation."""
import json,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import mujoco
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parent.parent
OUT=Path(os.environ.get('EE_AUDIT_DIR',str(ROOT/'audit_forward/ee_100_20261029')))


def main():
    data=np.load(str(OUT/'trajectories.npz'));meta=json.loads((OUT/'metadata.json').read_text())
    m=mujoco.MjModel.from_xml_path(str(ROOT/'resources/commands/plate_forward.xml'));d=mujoco.MjData(m)
    m.vis.headlight.ambient[:]=[.6,.6,.6];m.vis.headlight.diffuse[:]=[.8,.8,.8]
    renderer=mujoco.Renderer(m,height=360,width=640);cam=mujoco.MjvCamera();cam.azimuth=120;cam.elevation=-20;cam.distance=2.15
    qids=[int(m.joint(n).qposadr) for n in meta['dof_names']]
    def frame(i,env):
        root=data['root'][i,env];d.qpos[:3]=root[:3];d.qpos[3:7]=root[[6,3,4,5]];d.qpos[qids]=data['q'][i,env]
        mujoco.mj_forward(m,d);R=d.xmat[1].reshape(3,3)
        cam.azimuth=120+np.rad2deg(np.arctan2(R[1,0],R[0,0]));cam.lookat[:]=root[:3]+[.10,0,.15]
        renderer.update_scene(d,camera=cam)
        for p,size,color in [(data['target'][i,env],.012,[1,.15,.15,.9]),(data['ee'][i,env],.008,[0,1,1,1])]:
            g=renderer.scene.geoms[renderer.scene.ngeom];renderer.scene.ngeom+=1
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([size]*3),p,np.eye(3).reshape(-1),np.array(color,dtype=np.float32))
        image=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,640,58),fill=(245,245,245))
        draw.text((8,6),'PhysX state replay | env %d | iter %d | %s'%(meta['video_ids'][env],data['iteration'][i],'RANDOM GOAL' if data['random'][i,env] else 'INITIAL HOLD'),fill='black')
        draw.text((8,23),'Red: applied target | Cyan: actual EE | error %.1f cm'%(100*data['position_error'][i,env]),fill='black')
        draw.text((8,40),'Flat height reference; training rough terrain not rendered',fill='black')
        return np.array(image)
    count=len(data['step']);envs=[0,1] if data['q'].shape[1]>1 else [0,0]
    random_up=data['random']&data['upright']
    score=random_up.sum(0)
    if len(score)>1:envs[1]=1+int(np.argmax(score[1:]))
    # Four seconds from each end: enough to show multiple short resets.
    windows=[('initial',0,min(count,100)),('late',max(0,count-100),count)]
    montage=Image.new('RGB',(1280,720),'white')
    for row,(label,start,end) in enumerate(windows):
        with imageio.get_writer(str(OUT/('ee_'+label+'_replay.mp4')),fps=25,codec='libx264',quality=8,macro_block_size=1) as writer:
            for i in range(start,end):
                joined=np.concatenate([frame(i,j) for j in envs],axis=1);writer.append_data(joined)
                if i==(start+end)//2:montage.paste(Image.fromarray(joined),(0,row*360))
    montage.save(str(OUT/'ee_replay_montage.jpg'),quality=94);renderer.close()
    print('Rendered recorded trajectories. No MuJoCo physics integration was used.')


if __name__=='__main__':main()
