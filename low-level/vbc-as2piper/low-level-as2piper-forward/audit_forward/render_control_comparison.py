"""Actual recorded states from nominal stand, fixed arm and PPO tests."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import mujoco
import imageio.v2 as imageio
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'audit_forward/ee_100_20261029'


def main():
    stand=np.load(str(ROOT/'audit_forward/ee_nominal_stand_20261029/tracking.npz'))
    fixed=np.load(str(ROOT/'audit_forward/ee_fixed_base_20261029/trajectories.npz'))
    train=np.load(str(OUT/'trajectories.npz'))
    cases=[('Nominal free base | zero actions | home hold',stand['root'][400,0],stand['q'][400,0],None),
           ('Fixed base | independent spherical pose command',fixed['root'][500,0],fixed['q'][500,0],fixed['target'][500,0]),
           ('PPO test | original randomization | late frame',train['root'][-50,0],train['q'][-50,0],train['target'][-50,0])]
    m=mujoco.MjModel.from_xml_path(str(ROOT/'resources/commands/plate_forward.xml'));d=mujoco.MjData(m)
    m.vis.headlight.ambient[:]=[.6]*3;m.vis.headlight.diffuse[:]=[.8]*3
    renderer=mujoco.Renderer(m,height=360,width=640);camera=mujoco.MjvCamera();camera.distance=2.15;camera.elevation=-20
    names=[leg+'_'+j+'_joint' for leg in ('FL','FR','RL','RR') for j in ('hip','thigh','calf')]+['piper_joint'+str(i) for i in range(1,7)]+['piper_gripper_joint1','piper_gripper_joint2']
    ids=[int(m.joint(n).qposadr) for n in names]
    canvas=Image.new('RGB',(1920,360),'white')
    for i,(label,root,q,target) in enumerate(cases):
        d.qpos[:3]=root[:3];d.qpos[3:7]=root[[6,3,4,5]];d.qpos[ids]=q;mujoco.mj_forward(m,d)
        R=d.xmat[1].reshape(3,3);camera.azimuth=120+np.rad2deg(np.arctan2(R[1,0],R[0,0]));camera.lookat[:]=root[:3]+[.1,0,.15]
        renderer.update_scene(d,camera=camera)
        if target is not None:
            g=renderer.scene.geoms[renderer.scene.ngeom];renderer.scene.ngeom+=1
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.012]*3),target,np.eye(3).reshape(-1),np.array([1,.1,.1,1],dtype=np.float32))
        image=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(image);draw.rectangle((0,0,640,38),fill='white')
        draw.text((8,7),label,fill='black');draw.text((8,23),'Recorded PhysX state; PPO rough terrain omitted',fill='black');canvas.paste(image,(i*640,0))
    canvas.save(str(OUT/'control_comparison.jpg'),quality=94)
    # A separate actual-motion clip shows commanded arm motion with a fixed
    # torso, rather than only the short-reset PPO initialization phase.
    start,end=250,350
    movement=np.linalg.norm(np.diff(fixed['target'][start:end],axis=0),axis=-1).sum(axis=0)
    env=int(np.argmax(movement))
    with imageio.get_writer(str(OUT/'ee_fixed_base_tracking.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=1) as writer:
        for frame in range(start,end):
            root=fixed['root'][frame,env];d.qpos[:3]=root[:3];d.qpos[3:7]=root[[6,3,4,5]];d.qpos[ids]=fixed['q'][frame,env]
            mujoco.mj_forward(m,d);camera.azimuth=120;camera.lookat[:]=root[:3]+[.1,0,.15];renderer.update_scene(d,camera=camera)
            for point,size,color in [(fixed['target'][frame,env],.012,[1,.1,.1,1]),(fixed['ee'][frame,env],.008,[0,1,1,1])]:
                g=renderer.scene.geoms[renderer.scene.ngeom];renderer.scene.ngeom+=1
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([size]*3),point,np.eye(3).reshape(-1),np.array(color,dtype=np.float32))
            image=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(image);draw.rectangle((0,0,640,40),fill='white')
            draw.text((8,7),'Fixed torso | nominal physics | B1 spherical pose command',fill='black')
            error=100*np.linalg.norm(fixed['target'][frame,env]-fixed['ee'][frame,env])
            draw.text((8,24),'Recorded PhysX | Red goal, cyan EE | error %.1f cm'%error,fill='black')
            writer.append_data(np.array(image))
    renderer.close()


if __name__=='__main__':main()
