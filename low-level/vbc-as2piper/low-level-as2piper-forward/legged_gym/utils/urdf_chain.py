"""Batched URDF FK in xyzw quaternions; no simulator or NumPy dependency."""
import math
import xml.etree.ElementTree as ET
import torch


def multiply(a,b):
    av,aw=a[...,:3],a[...,3:];bv,bw=b[...,:3],b[...,3:]
    return torch.cat((aw*bv+bw*av+torch.cross(av,bv,dim=-1),aw*bw-(av*bv).sum(-1,keepdim=True)),-1)


def rotate(q,v):
    uv=torch.cross(q[...,:3],v,dim=-1)
    return v+2*(q[...,3:]*uv+torch.cross(q[...,:3],uv,dim=-1))


def inverse(q):
    return torch.cat((-q[...,:3],q[...,3:]),-1)


def rpy_quaternion(rpy):
    r,p,y=[v/2 for v in rpy];cr,sr=math.cos(r),math.sin(r);cp,sp=math.cos(p),math.sin(p);cy,sy=math.cos(y),math.sin(y)
    return [sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy]


class URDFChain:
    def __init__(self,path,base,tip,joint_names,device='cpu'):
        robot=ET.parse(path).getroot();parents={j.find('child').get('link'):j for j in robot.findall('joint')}
        chain=[];current=tip
        while current!=base:
            j=parents[current];chain.append(j);current=j.find('parent').get('link')
        self.steps=[];self.joint_names=joint_names
        for j in reversed(chain):
            o=j.find('origin');xyz=[float(v) for v in (o.get('xyz','0 0 0') if o is not None else '0 0 0').split()]
            rpy=[float(v) for v in (o.get('rpy','0 0 0') if o is not None else '0 0 0').split()]
            axis=j.find('axis');axis=[float(v) for v in axis.get('xyz').split()] if axis is not None else [1,0,0]
            index=-1 if j.get('type')=='fixed' else joint_names.index(j.get('name'))
            self.steps.append((torch.tensor(xyz,device=device),torch.tensor(rpy_quaternion(rpy),device=device),
                               torch.tensor(axis,device=device),index,j.get('type')))

    def __call__(self,q):
        p=torch.zeros(q.shape[0],3,device=q.device,dtype=q.dtype)
        rot=torch.zeros(q.shape[0],4,device=q.device,dtype=q.dtype);rot[:,3]=1
        for xyz,orientation,axis,index,kind in self.steps:
            xyz=xyz.to(q).expand(q.shape[0],-1);orientation=orientation.to(q).expand(q.shape[0],-1)
            p=p+rotate(rot,xyz);rot=multiply(rot,orientation)
            if index<0:continue
            if kind=='prismatic':p=p+rotate(rot,axis.to(q).expand_as(p)*q[:,index:index+1])
            else:
                a=q[:,index:index+1]/2
                delta=torch.cat((axis.to(q).expand_as(p)*torch.sin(a),torch.cos(a)),-1)
                rot=multiply(rot,delta)
        return p,rot/rot.norm(dim=-1,keepdim=True)
