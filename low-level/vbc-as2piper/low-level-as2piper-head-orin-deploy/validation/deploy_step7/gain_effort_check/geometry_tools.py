"""Offline URDF IK/geometry study. No training or policy/config edits."""
import csv
import json
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import lsq_linear

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
URDF = OUT.parents[2]/'resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf'
sys.path.insert(0,str(ROOT/'mujoco_mount_compare'))
from build_mount_scenes import Converter, origin_attrs


class ExactCollision(Converter):
    def make_link(self,name,parent,pose,via=None):
        body=ET.SubElement(parent,'body',{'name':name,**pose})
        if via is not None and via.get('type')!='fixed':ET.SubElement(body,'joint',self.joint_motor(via))
        link=self.links[name];self.add_inertial(body,link)
        for i,g in enumerate(link.findall('visual')):self.add_geom(body,g,True,name,i)
        for i,g in enumerate(link.findall('collision')):self.add_geom(body,g,False,name,i)
        for j in self.children.get(name,[]):self.make_link(j.find('child').get('link'),body,origin_attrs(j.find('origin')),j)
        return body


path=OUT/'analysis_model.xml'
c=ExactCollision(URDF,path,{'time_step_s':.005,'gravity_m_s2':-9.81},'deployment_geometry')
base=c.make_link('base_link',c.world,{'pos':'0 0 0'});ET.SubElement(base,'freejoint',name='base_free')
for mesh in c.assets.findall('mesh'):mesh.set('file',str((path.parent/mesh.get('file')).resolve()))
ET.ElementTree(c.root).write(path,encoding='utf-8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
legs=['FL','FR','RL','RR'];feet=[m.body(n+'_foot').id for n in legs];tcp=m.body('piper_umi_nominal_tcp').id
arm_ids=np.array([m.joint('piper_joint'+str(i)).id for i in range(1,7)])
aq=m.jnt_qposadr[arm_ids];av=m.jnt_dofadr[arm_ids];alo,ahi=m.jnt_range[arm_ids].T
leg_ids=np.array([m.joint(leg+'_'+joint+'_joint').id for leg in legs for joint in ['hip','thigh','calf']])
lq=m.jnt_qposadr[leg_ids];lv=m.jnt_dofadr[leg_ids];llo,lhi=m.jnt_range[leg_ids].T
neutral=np.array([0,1.1,-.55,0,0,0.]);d.qpos[aq]=neutral;d.qpos[lq]=[0,.8,-1.6]*4
for n,v in [('piper_gripper_joint1',.049),('piper_gripper_joint2',-.049)]:d.qpos[m.joint(n).qposadr]=v
d.qpos[:3]=[0,0,.3244069024246581];d.qpos[3:7]=[1,0,0,0]
mujoco.mj_forward(m,d);foot_targets=d.xpos[feet].copy();foot_targets[:,2]=.028
hull=ConvexHull(foot_targets[:,:2]);jp=np.zeros((3,m.nv));jr=jp.copy()

# Ignore within-rigid-assembly and immediate joint-neighbor collisions only.
parent={name:name for name in c.links}
def find(x):
    while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
    return x
for j in c.joints:
    if j.get('type')=='fixed':parent[find(j.find('child').get('link'))]=find(j.find('parent').get('link'))
adj={frozenset((find(j.find('parent').get('link')),find(j.find('child').get('link')))) for j in c.joints if j.get('type')!='fixed'}


def fk():
    mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)


def collisions():
    mujoco.mj_collision(m,d)
    hits=[]
    for contact in d.contact:
        if contact.dist>=-.001:continue
        a,b=[m.body(m.geom_bodyid[g]).name for g in [contact.geom1,contact.geom2]]
        if a=='world' or b=='world':
            other=b if a=='world' else a
            if other.endswith('_foot'):continue
        else:
            ra,rb=find(a),find(b)
            if ra==rb or frozenset((ra,rb)) in adj:continue
        hits.append([a,b,float(contact.dist)])
    return hits


def leg_ik(height,tilt):
    d.qpos[:3]=[0,0,height];d.qpos[3:7]=[np.cos(tilt/2),0,np.sin(tilt/2),0]
    d.qpos[lq]=[0,.8,-1.6]*4
    for _ in range(80):
        fk();error=(foot_targets-d.xpos[feet]).reshape(-1)
        if np.max(np.abs(error))<.0005:return True
        J=np.zeros((12,12))
        for k,body in enumerate(feet):
            mujoco.mj_jacBody(m,d,jp,jr,body);J[k*3:k*3+3]=jp[:,lv]
        step=np.linalg.solve(J.T@J+.00003*np.eye(12),J.T@error)
        d.qpos[lq]=np.clip(d.qpos[lq]+np.clip(step,-.12,.12),llo+1e-5,lhi-1e-5)
    fk();return bool(np.max(np.abs(foot_targets-d.xpos[feet]))<.002)


def arm_ik(target,beta,starts,yaw=0.):
    tq=np.array([np.cos(yaw/2)*np.cos(beta/2),-np.sin(yaw/2)*np.sin(beta/2),np.cos(yaw/2)*np.sin(beta/2),np.sin(yaw/2)*np.cos(beta/2)]);best=(1e9,1e9,None)
    for initial in starts:
        d.qpos[aq]=np.clip(initial,alo+1e-5,ahi-1e-5)
        for _ in range(75):
            fk();actual=d.xquat[tcp].copy();inv=actual.copy();inv[1:]*=-1
            dq=np.zeros(4);mujoco.mju_mulQuat(dq,tq,inv)
            if dq[0]<0:dq*=-1
            angle=np.zeros(3);mujoco.mju_quat2Vel(angle,dq,1)
            error=target-d.xpos[tcp];pe=np.linalg.norm(error);ae=np.linalg.norm(angle)
            if pe+.25*ae<best[0]+.25*best[1]:best=(pe,ae,d.qpos[aq].copy())
            if pe<.003 and ae<np.deg2rad(2):return True,pe,ae
            mujoco.mj_jacBody(m,d,jp,jr,tcp);J=np.vstack([jp[:,av],.25*jr[:,av]])
            step=J.T@np.linalg.solve(J@J.T+.00008*np.eye(6),np.r_[error,.25*angle])
            d.qpos[aq]=np.clip(d.qpos[aq]+np.clip(step,-.15,.15),alo+1e-5,ahi-1e-5)
    d.qpos[aq]=best[2];fk();return False,best[0],best[1]


