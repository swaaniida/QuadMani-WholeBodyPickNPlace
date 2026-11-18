"""Offline camera motion for visualization only. Never changes selection ICP/scores."""
import hashlib
import json
from pathlib import Path
import numpy as np
from camera_motion import RGBDOdometry,read_rgbd,collect_frames


def reconstruct(path,rows,progress=None,cancel=None):
    path=Path(path);items=collect_frames(path,rows)
    signature=hashlib.sha256(json.dumps([(fid,name,(path/name).stat().st_size,(path/name).stat().st_mtime_ns) for fid,name in items]).encode()).hexdigest()
    cache=path/'replay_camera_motion.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as d:
            if str(d['signature'])==signature:return {k:d[k].copy() for k in d.files}
    first=read_rgbd(path/items[0][1]);engine=RGBDOdometry(first)
    poses=[];valid=[];segments=[];statuses=[];segment=0;previous_ok=True
    for i,(fid,name) in enumerate(items):
        if cancel is not None and cancel.is_set():raise RuntimeError('Replay reconstruction cancelled')
        if progress:progress(i,len(items))
        if i==0:T=np.eye(4);status='reference'
        elif fid.rsplit('-',1)[0]!=items[0][0].rsplit('-',1)[0]:T=None;status='stream_changed'
        else:
            try:
                frame=read_rgbd(path/name)
                if frame['frame_id']!=fid:raise ValueError('Frame ID mismatch')
                T,metrics=engine.step(frame);status=metrics['status']
            except Exception as exc:T=None;status=f'error:{exc}'
        ok=T is not None
        if ok and not previous_ok:segment+=1
        poses.append(T if ok else np.full((4,4),np.nan));valid.append(ok);statuses.append(status);segments.append(segment)
        previous_ok=ok
    result=dict(frame_ids=np.asarray([fid for fid,_ in items]),T_world_camera=np.asarray(poses),valid=np.asarray(valid),
                segments=np.asarray(segments),status=np.asarray(statuses),signature=np.asarray(signature))
    np.savez_compressed(cache,**result)
    if progress:progress(len(items),len(items))
    return result


def trajectory_lines(poses,valid,segments):
    """Archived/processed frame adjacency, NOT consecutive raw camera frame numbers."""
    ids=np.flatnonzero(valid)
    if len(ids)<2:return np.empty((0,2,3),np.float32)
    connected=(np.diff(ids)==1)&(np.diff(np.asarray(segments)[ids])==0)
    positions=np.asarray(poses)[ids,:3,3]
    return np.stack((positions[:-1],positions[1:]),axis=1)[connected].astype(np.float32)
