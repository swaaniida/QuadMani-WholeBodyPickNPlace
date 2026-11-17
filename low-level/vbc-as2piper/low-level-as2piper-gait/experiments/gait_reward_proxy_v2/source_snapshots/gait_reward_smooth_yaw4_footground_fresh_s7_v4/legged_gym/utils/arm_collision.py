"""Batched IK and conservative arm-vs-body OBB rejection, not a goal bank."""
import json
import torch
from .urdf_chain import URDFChain,rotate


class ArmKinematics(torch.nn.Module):
    def __init__(self,asset):
        super().__init__()
        chain=URDFChain(asset,'piper_base_link','piper_umi_nominal_tcp',
                        ['piper_joint'+str(i) for i in range(1,7)])
        offsets=[];rotations=[];axes=[];p=torch.zeros(3);R=torch.eye(3)
        for xyz,quat,axis,index,kind in chain.steps:
            mat=rotate(quat.expand(3,-1),torch.eye(3)).T
            p=p+R@xyz;R=R@mat
            if index>=0:
                offsets.append(p);rotations.append(R);axes.append(axis)
                p=torch.zeros(3);R=torch.eye(3)
        self.register_buffer('offsets',torch.stack(offsets));self.register_buffer('rotations',torch.stack(rotations))
        self.register_buffer('axes',torch.stack(axes));self.register_buffer('tip_p',p);self.register_buffer('tip_R',R)
        skew=[]
        for a in axes:
            x,y,z=a;skew.append(torch.tensor([[0.,-z,y],[z,0.,-x],[-y,x,0.]]))
        K=torch.stack(skew);self.register_buffer('K',K);self.register_buffer('K2',K@K)

    def forward(self,q):
        n=q.shape[0];p=torch.zeros((n,3),device=q.device);R=torch.eye(3,device=q.device).expand(n,3,3)
        points=torch.zeros((n,6,3),device=q.device);frames=torch.zeros((n,6,3,3),device=q.device)
        axis=torch.zeros((n,6,3),device=q.device)
        for j in range(6):
            p=p+(R@self.offsets[j].view(1,3,1)).squeeze(-1);R=R@self.rotations[j]
            points[:,j]=p;axis[:,j]=(R@self.axes[j].view(1,3,1)).squeeze(-1)
            angle=q[:,j,None,None]
            R=R@(torch.eye(3,device=q.device)+torch.sin(angle)*self.K[j]+(1-torch.cos(angle))*self.K2[j])
            frames[:,j]=R
        tip=p+(R@self.tip_p.view(1,3,1)).squeeze(-1);orientation=R@self.tip_R
        linear=torch.cross(axis,tip[:,None,:]-points,dim=-1).transpose(1,2)
        J=torch.cat((linear,axis.transpose(1,2)),dim=1)
        return tip,orientation,J,points,frames


def box_collision(center,rotation,half,obstacle_center,obstacle_half):
    """Full 15-axis separating-axis test, arm OBB against body-aligned boxes."""
    # [batch, arm geom, obstacle, axis, xyz]
    n,g=center.shape[:2];h=obstacle_center.shape[0]
    R=rotation[:,:,None].expand(n,g,h,3,3)
    eye=torch.eye(3,device=center.device).expand(n,g,h,3,3)
    arm_axes=R.transpose(-1,-2)
    cross=torch.cross(arm_axes[:,:,:, :,None,:].expand(n,g,h,3,3,3),
                      eye[:,:,:,None,:,:].expand(n,g,h,3,3,3),dim=-1).reshape(n,g,h,9,3)
    axes=torch.cat((eye,arm_axes,cross),dim=-2)
    delta=center[:,:,None,:]-obstacle_center[None,None,:,:]
    distance=(axes*delta[:,:,:,None,:]).sum(-1).abs()
    radius_a=((axes@R).abs()*half[None,:,None,None,:]).sum(-1)
    radius_b=(axes.abs()*obstacle_half[None,None,:,None,:]).sum(-1)
    return (~(distance>radius_a+radius_b+1e-7).any(-1)).reshape(n,-1).any(-1)


class ArmCollisionGuard:
    def __init__(self,asset,geometry,device):
        self.fk=torch.jit.script(ArmKinematics(asset)).to(device)
        data=json.loads(geometry.read_text())
        def value(key,dtype=torch.float):return torch.tensor(data[key],device=device,dtype=dtype)
        self.link_index=value('link_index',torch.long);self.center=value('center');self.half=value('half')
        self.obstacle_center=value('obstacle_center');self.obstacle_half=value('obstacle_half')
        self.low=value('joint_low');self.high=value('joint_high')

    def solve(self,target,orientation,seed,iterations=12):
        q=seed.clone();identity=torch.eye(6,device=q.device)
        for _ in range(iterations):
            p,R,J,_,_=self.fk(q)
            # Spatial orientation error; target/R columns are basis vectors.
            w=.5*torch.cross(R.transpose(1,2),orientation.transpose(1,2),dim=-1).sum(1)
            error=torch.cat((target-p,w),dim=-1)
            dq=(J.transpose(1,2)@torch.linalg.solve(J@J.transpose(1,2)+.0025*identity,error[:,:,None])).squeeze(-1)
            q=torch.max(torch.min(q+dq.clamp(-.18,.18),self.high),self.low)
        p,R,_,_,_=self.fk(q)
        angle_cos=((R*orientation).sum(dim=(1,2))-1)/2
        failed=(p-target).norm(dim=-1)>.012
        failed|=angle_cos<.9961947  # 5 degrees
        return q,failed

    def collides(self,q,body_arm_offset):
        _,_,_,points,frames=self.fk(q)
        R=frames[:,self.link_index]
        p=points[:,self.link_index]+(R@self.center[None,:,:,None]).squeeze(-1)+body_arm_offset
        return box_collision(p,R,self.half,self.obstacle_center,self.obstacle_half)
