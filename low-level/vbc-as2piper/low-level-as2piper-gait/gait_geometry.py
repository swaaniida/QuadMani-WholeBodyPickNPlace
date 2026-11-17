"""CPU-only checks against recorded simulator collision surfaces."""
import numpy as np


def distances(points,triangles,return_closest=False):
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    ab=b-a;ac=c-a;n=np.cross(ab,ac);nn=(n*n).sum(-1)
    good=nn>1e-16;a,b,c,ab,ac,n,nn=[z[good] for z in [a,b,c,ab,ac,n,nn]]
    aa=(ab*ab).sum(-1);bb=(ab*ac).sum(-1);cc=(ac*ac).sum(-1);denom=aa*cc-bb*bb
    result=[];closest=[]
    for offset in range(0,len(points),32):
        p=points[offset:offset+32,None,:];ap=p-a
        d0=(ap*ab).sum(-1);d1=(ap*ac).sum(-1)
        u=(cc*d0-bb*d1)/np.maximum(denom,1e-20);v=(aa*d1-bb*d0)/np.maximum(denom,1e-20)
        inside=(u>=0)&(v>=0)&(u+v<=1)
        d=np.where(inside,(ap*n).sum(-1)**2/nn,np.inf)
        if return_closest:
            nearest=p-((ap*n).sum(-1)/nn)[...,None]*n
        for first,last in [(a,b),(b,c),(c,a)]:
            edge=last-first;t=np.clip(((p-first)*edge).sum(-1)/np.maximum((edge*edge).sum(-1),1e-20),0,1)
            candidate=first+t[...,None]*edge
            distance=((p-candidate)**2).sum(-1)
            if return_closest:nearest=np.where((distance<d)[...,None],candidate,nearest)
            d=np.minimum(d,distance)
        index=d.argmin(-1);rows=np.arange(len(index))
        result.append(np.sqrt(d[rows,index]))
        if return_closest:closest.append(nearest[rows,index])
    result=np.concatenate(result)
    return (result,np.concatenate(closest)) if return_closest else result


def box_distances(points,poses,size,return_closest=False):
    """Point to solid oriented cube; quaternion xyzw, size is edge length."""
    delta=points-poses[:,:3];xyz=poses[:,3:6];w=poses[:,6:7]
    t=2*np.cross(-xyz,delta)
    local=delta+w*t+np.cross(-xyz,t)
    clipped=np.clip(local,-size/2,size/2)
    distance=np.linalg.norm(local-clipped,axis=-1)
    if not return_closest:return distance
    t=2*np.cross(xyz,clipped)
    closest=clipped+w*t+np.cross(xyz,t)+poses[:,:3]
    return distance,closest
