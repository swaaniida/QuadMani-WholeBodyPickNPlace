"""Pure geometry/reporting for independent OBB branch analysis; no model imports."""
import csv
import json
from pathlib import Path
import numpy as np
from official_viser_helpers import matrix_to_wxyz_position

BRANCHES=('diffusion','obb_camera','obb_world')
COLORS={'diffusion':(235,45,45),'obb_camera':(35,100,245),'obb_world':(30,190,75)}


def ground_basis(plane):
    p=np.asarray(plane,float);p=p/np.linalg.norm(p[:3]);z=p[:3]
    seed=np.array([1.,0,0]) if abs(z[0])<.95 else np.array([0.,1,0])
    x=seed-z*(seed@z);x/=np.linalg.norm(x);y=np.cross(z,x)
    T=np.eye(4);T[:3,:3]=np.column_stack((x,y,z));T[:3,3]=-p[3]*z
    return T


def generate_obb(points,plane,depth,compute_obb,top_down_pose,num_yaws=10,
                 offsets_cm=(-8,-6,-4,-2,0)):
    """Use official OBB/PCA and base-pose helpers in two canonical frames.

    Official helper approaches -canonical Z. Camera canonical +Z=-camera Z,
    giving +camera Z approach. World canonical +Z=plane free-space normal.
    """
    camera=np.eye(4);camera[:3,:3]=np.diag([1.,-1.,-1.])
    frames={'obb_camera':camera,'obb_world':ground_basis(plane)}
    result={};metadata={}
    for tag,F in frames.items():
        pc=(np.asarray(points)-F[:3,3])@F[:3,:3]
        center,half,R=compute_obb(pc,mode='pca')
        base=top_down_pose(center,half,R,z_offset=0.)
        top=float(center[2]+np.abs(R[2])@half)
        poses=[];sweep=[]
        for yaw in np.linspace(0,2*np.pi,num_yaws,endpoint=False):
            c,s=np.cos(yaw),np.sin(yaw);Y=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            for offset in offsets_cm:
                T=np.eye(4);T[:3,:3]=Y@base[:3,:3]
                T[:3,3]=[center[0],center[1],top+offset/100.]
                # Official graspmoe converts fingertip anchor to gripper base.
                T[:3,3]-=T[:3,:3]@np.array([0.,0.,depth])
                poses.append(F@T);sweep.append((float(yaw),float(offset)))
        result[tag]=np.asarray(poses,dtype=np.float32)
        metadata[tag]=dict(T_camera_canonical=F.tolist(),sweep_yaw_rad_offset_cm=sweep)
    return result,metadata


def analyze(grasps,confidences,tags,vertices,plane,rmse):
    g=np.asarray(grasps);c=np.asarray(confidences);tags=np.asarray(tags)
    p=np.asarray(plane,float);p/=np.linalg.norm(p[:3])
    distances=np.einsum('nji,j->ni',g[:,:3,:3],p[:3])@np.asarray(vertices).T+g[:,:3,3]@p[:3,None]+p[3]
    vi=distances.argmin(axis=1);d=distances[np.arange(len(g)),vi];pen=np.maximum(0,-d)
    status=np.where(d<0,'penetrating',np.where(d<=rmse,'contact_like','safe'))
    shallow=np.lexsort((-c,pen));confidence=np.argsort(-c,kind='stable')
    sr=np.empty(len(g),int);sr[shallow]=np.arange(1,len(g)+1)
    cr=np.empty(len(g),int);cr[confidence]=np.arange(1,len(g)+1)
    br=np.empty(len(g),int)
    for tag in BRANCHES:
        ids=np.flatnonzero(tags==tag);order=ids[np.argsort(-c[ids],kind='stable')];br[order]=np.arange(1,len(ids)+1)
    # Three explicit definitions; representatives may legitimately coincide.
    least=int(min(range(len(g)),key=lambda i:(pen[i],-d[i],-c[i])))
    highlights=dict(least_penetration=least,highest_confidence=int(confidence[0]),
                    shallow_high_confidence=int(shallow[0]))
    rows=[]
    for i in shallow:
        q,xyz=matrix_to_wxyz_position(g[i]);v=np.asarray(vertices)[vi[i]]
        row=dict(shallow_rank=int(sr[i]),confidence_rank=int(cr[i]),branch_tag=str(tags[i]),
                 candidate_index=int(i),branch_rank=int(br[i]),confidence=float(c[i]),
                 penetration_mm=float(pen[i]*1000),clearance_mm=float(d[i]*1000),
                 collision_status=str(status[i]),minimum_signed_distance_m=float(d[i]),
                 minimum_vertex_index=int(vi[i]),minimum_vertex_local_xyz_mm=json.dumps((v*1000).tolist()),
                 minimum_region_heuristic='proximal_z_lt_90mm' if v[2]<.09 else 'distal_z_ge_90mm',
                 x=float(xyz[0]),y=float(xyz[1]),z=float(xyz[2]),qx=float(q[1]),qy=float(q[2]),qz=float(q[3]),qw=float(q[0]),
                 pose_summary=f'xyz={np.round(xyz,4).tolist()}; qxyzw={np.round(q[[1,2,3,0]],4).tolist()}',
                 pose_4x4=json.dumps(g[i].tolist()),rotation_3x3=json.dumps(g[i,:3,:3].tolist()))
        rows.append(row)
    summary=[]
    for tag in ('all',)+BRANCHES:
        ids=np.arange(len(g)) if tag=='all' else np.flatnonzero(tags==tag)
        if not len(ids):continue
        best=int(ids[np.argmax(c[ids])]);pp=pen[ids]*1000
        summary.append(dict(branch=tag,count=len(ids),safe=int(np.sum(status[ids]=='safe')),
            contact_like=int(np.sum(status[ids]=='contact_like')),penetrating=int(np.sum(status[ids]=='penetrating')),
            confidence_top1=float(c[best]),confidence_top1_index=best,
            confidence_top1_penetration_mm=float(pen[best]*1000),
            penetration_min_mm=float(pp.min()),penetration_mean_mm=float(pp.mean()),penetration_max_mm=float(pp.max())))
    return rows,summary,highlights


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def table(rows,columns):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(f'{r[k]:.4g}' if isinstance(r[k],float) else str(r[k]) for k in columns)+' |' for r in rows])
