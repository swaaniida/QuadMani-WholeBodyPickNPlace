"""Replay helpers kept separate so the legacy viewer is never overwritten."""

import numpy as np

from camera_motion import transform_points

def sample_times(rows):
    # Prefer capture time, never inference completion time when capture timestamps exist.
    for key,scale in (('host_timestamp_s',1.),('camera_timestamp_ms',.001),('completed_elapsed_s',1.)):
        try:values=np.array([float(row[key]) for row in rows])*scale
        except (KeyError,TypeError,ValueError):continue
        if np.isfinite(values).all() and np.all(np.diff(values)>=0):return values-values[0]
    raise ValueError('No finite chronological timestamps in poses.csv')

def sample_at_time(times,t):
    return max(0,min(len(times)-1,int(np.searchsorted(times,t,side='right')-1)))

def loop_hold(times):
    intervals=np.diff(times);positive=intervals[intervals>0]
    return float(np.median(positive)) if len(positive) else .5

def advance_playback(cursor,dt,speed,duration,loop,end_hold=.5):
    if duration<=0:return 0.,False
    next_time=cursor+max(0.,dt)*speed
    if not loop and next_time>=duration:return duration,False
    # Last pose stays for the same typical interval as other recorded poses.
    cycle=duration+max(.001,end_hold)
    if loop and next_time>=cycle:return next_time%cycle,True
    return next_time,True

def playback_style(index,count,fraction=None):
    if fraction is None:fraction=index/max(count-1,1) if count>1 else 1.
    fraction=float(np.clip(fraction,0,1))
    # Visible starting opacity plus a conspicuous yellow -> orange -> dark-red ramp.
    stops=np.array([[255,245,185],[255,100,0],[120,0,15]],float)
    segment=min(int(fraction*2),1);weight=fraction*2-segment
    color=np.rint(stops[segment]*(1-weight)+stops[segment+1]*weight).astype(np.uint8)
    return float(.25+.75*fraction**.7),color

def fuse_clouds(clouds,colors,voxel=.003):
    points=np.concatenate(clouds);rgb=np.concatenate(colors)
    valid=np.isfinite(points).all(axis=1);points=points[valid];rgb=rgb[valid]
    if not len(points):raise ValueError('No valid reconstructed object points')
    _,inverse=np.unique(np.floor(points/voxel).astype(np.int64),axis=0,return_inverse=True)
    count=np.bincount(inverse)
    means=lambda array:np.stack([np.bincount(inverse,weights=array[:,j])/count for j in range(3)],axis=1)
    return means(points).astype(np.float32),np.rint(means(rgb)).clip(0,255).astype(np.uint8)

def calibrated_frustum(T,K,width,height,scale=.10):
    pixels=np.array([[0,0,1],[width-1,0,1],[width-1,height-1,1],[0,height-1,1]],float)
    corners=(pixels@np.linalg.inv(K).T)*scale
    local=np.concatenate((np.stack((np.zeros_like(corners),corners),axis=1),
                          np.stack((corners,np.roll(corners,-1,axis=0)),axis=1)))
    return transform_points(local.reshape(-1,3),T).reshape(-1,2,3).astype(np.float32)
