#!/usr/bin/env python3
"""CPU MuJoCo snapshots; no control commands, training or service changes."""
import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from PIL import Image
from check_wrist_camera import normalized_urdf
from sync_wrist_camera import ROOT,WORK,fk


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf',type=Path,default=ROOT/'low-level-as2piper-head-orin/resources/robots/as2piper/urdf/as2piper_head_umi_orin64_backmount.urdf')
    args=p.parse_args();path=args.urdf.resolve()
    robot=ET.parse(path).getroot()
    model=mujoco.MjModel.from_xml_string(normalized_urdf(path,path.read_bytes()));data=mujoco.MjData(model)
    model.vis.headlight.ambient[:]=[.6,.6,.6];model.vis.headlight.diffuse[:]=[.7,.7,.7]
    for i in range(model.njnt):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i);address=model.jnt_qposadr[i]
        if name=='piper_gripper_joint1':data.qpos[address]=.05
        if name=='piper_gripper_joint2':data.qpos[address]=-.05
    mujoco.mj_forward(model,data)
    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'piper_link6')
    poses=fk(robot,'piper_link6');T=poses['wrist_d435i_color_optical_frame']
    R=data.xmat[body].reshape(3,3);camera=data.xpos[body]+R@T[:3,3]
    close=(camera+data.xpos[body])/2
    out=WORK/'renders';out.mkdir(exist_ok=True);os.chmod(out,0o2775)
    renderer=mujoco.Renderer(model,height=360,width=640)
    try:
        for tag,center,distance in [('whole',np.array([.05,0,0]),1.45),('wrist',close,.42)]:
            for label,azimuth,elevation in [('front',0,-15),('side',90,-15),('top',0,-80)]:
                view=mujoco.MjvCamera();view.lookat[:]=center;view.distance=distance;view.azimuth=azimuth;view.elevation=elevation
                renderer.update_scene(data,camera=view)
                output=out/(tag+'_'+label+'.png');Image.fromarray(renderer.render()).save(str(output));os.chmod(output,0o664)
                print(output,flush=True)
    finally:renderer.close()


if __name__=='__main__':main()
