"""Geometry and temporal styling for recorded poses; never generates predictions."""
import numpy as np


def history_data(rows,max_grippers=200):
    if not rows:raise ValueError('Pose history is empty')
    poses=np.array([[[float(r[f't{a}{b}']) for b in range(4)] for a in range(4)] for r in rows],dtype=np.float64)
    times=np.array([float(r['completed_elapsed_s']) for r in rows],dtype=np.float64)
    if not np.isfinite(poses).all() or not np.isfinite(times).all() or np.any(np.diff(times)<0):
        raise ValueError('Expected finite, chronologically ordered recorded poses')
    elapsed=times-times[0]
    fraction=elapsed/elapsed[-1] if elapsed[-1]>0 else np.linspace(0,1,len(rows))
    if len(rows)==1:fraction[:]=1
    # Color and alpha encode time, independently of model confidence.
    colors=np.rint((1-fraction[:,None])*np.array([205,230,248])+fraction[:,None]*np.array([15,70,150])).astype(np.uint8)
    opacity=(.06+.79*fraction).astype(np.float32)
    positions=poses[:,:3,3].astype(np.float32)
    keys=[r.get('stability_segment',f'{r.get("session","")}:{r.get("generation","")}') for r in rows]
    keep=np.array([a==b for a,b in zip(keys[:-1],keys[1:])],dtype=bool)
    lines=np.stack((positions[:-1],positions[1:]),axis=1)[keep] if len(rows)>1 else np.empty((0,2,3),np.float32)
    line_colors=np.stack((colors[:-1],colors[1:]),axis=1)[keep] if len(rows)>1 else np.empty((0,2,3),np.uint8)
    indices=np.unique(np.linspace(0,len(rows)-1,min(max(1,int(max_grippers)),len(rows)),dtype=int))
    starts=np.r_[0,np.flatnonzero(~keep)+1]
    ends=np.r_[np.flatnonzero(~keep),len(rows)-1]
    return dict(poses=poses,positions=positions,elapsed=elapsed,fraction=fraction,colors=colors,opacity=opacity,
                lines=lines,line_colors=line_colors,gripper_indices=indices,starts=starts,ends=ends)


def direction_arrows(lines,colors,limit=12):
    """Two-stroke arrowheads on observed connecting lines; no pose interpolation."""
    result=[];result_colors=[]
    valid=np.flatnonzero(np.linalg.norm(lines[:,1]-lines[:,0],axis=1)>1e-6)
    if len(valid):valid=valid[np.unique(np.linspace(0,len(valid)-1,min(limit,len(valid)),dtype=int))]
    for idx in valid:
        a,b=lines[idx];delta=b-a;length=np.linalg.norm(delta);direction=delta/length
        reference=np.eye(3)[np.argmin(np.abs(direction))]
        side=np.cross(direction,reference);side/=np.linalg.norm(side)
        tip=a+.75*delta;size=min(.012,length*.23)
        for sign in (-1,1):
            result.append([tip,tip-size*direction+sign*.5*size*side])
            result_colors.append([colors[idx,1],colors[idx,1]])
    return np.asarray(result,dtype=np.float32).reshape(-1,2,3),np.asarray(result_colors,dtype=np.uint8).reshape(-1,2,3)


def sweep_gripper_mesh(params):
    """Explicitly simplified fallback using official sweep dimensions (not CAD)."""
    import trimesh
    width,height,depth=map(float,params['extents_open'])
    cx,cy,cz=map(float,params['offset_open']);thickness=.008
    parts=[]
    for x in (cx-width/2-thickness/2,cx+width/2+thickness/2):
        mesh=trimesh.creation.box(extents=(thickness,height,depth))
        mesh.apply_translation((x,cy,cz));parts.append(mesh)
    palm=trimesh.creation.box(extents=(width+2*thickness,height,thickness))
    palm.apply_translation((cx,cy,cz-depth/2-thickness/2));parts.append(palm)
    return trimesh.util.concatenate(parts)
