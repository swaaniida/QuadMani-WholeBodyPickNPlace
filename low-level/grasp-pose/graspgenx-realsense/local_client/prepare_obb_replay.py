"""Export every completed grasp capture from a local recording; CPU only."""
import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
import zipfile
import cv2
import numpy as np
from view_temporal import latest_recording


def export_recording(session,output):
    session=Path(session);rows=list(csv.DictReader((session/'comparison.csv').open(newline='')))
    if not rows:raise ValueError('Recording has no completed grasp attempts')
    manifest=dict(session_name=session.name,source_session=str(session),frames=[])
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for i,row in enumerate(rows):
            with np.load(session/row['candidate_file'],allow_pickle=False) as f:
                data={k:f[k].copy() for k in ('frame_id','point_cloud','colors','ground_plane')}
                data['plane_rmse_m']=f['plane_rmse_m'].copy() if 'plane_rmse_m' in f else np.array(np.nan)
            with np.load(session/row['frame_file'],allow_pickle=False) as f:
                if str(f['frame_id'])!=str(data['frame_id']):raise ValueError('Frame-ID mismatch')
                depth=f['depth_m'];K=f['K'];mask=f['instance_mask']>0
                bg=(cv2.dilate(mask.astype(np.uint8),np.ones((11,11),np.uint8))==0)&np.isfinite(depth)&(depth>.08)&(depth<4.)
                v,u=np.where(bg)
                if len(v)>30000:
                    ids=np.linspace(0,len(v)-1,30000,dtype=int);v=v[ids];u=u[ids]
                dz=depth[v,u]
                data['background_points']=np.column_stack(((u-K[0,2])*dz/K[0,0],(v-K[1,2])*dz/K[1,1],dz)).astype(np.float32)
                rgb=cv2.imdecode(f['rgb_jpeg'],cv2.IMREAD_COLOR)[:,:,::-1]
                data['background_colors']=rgb[v,u];data['rgb_jpeg']=f['rgb_jpeg'].copy()
            name=f'inputs/{i:06d}.npz';buf=BytesIO();np.savez_compressed(buf,**data);z.writestr(name,buf.getvalue())
            manifest['frames'].append(dict(index=i,frame_id=str(data['frame_id']),input_file=name,
                capture_utc=row.get('capture_utc',''),camera_timestamp_ms=row.get('camera_timestamp_ms','')))
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    print(f'SESSION: {session}\nEXPORTED: {len(rows)} captures -> {Path(output).resolve()}',flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session',type=Path,nargs='?')
    p.add_argument('--output',type=Path,default=Path('obb-replay-input.zip'));a=p.parse_args()
    export_recording(a.session if a.session else latest_recording(),a.output)
