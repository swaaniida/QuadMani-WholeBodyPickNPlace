"""Bounded background RGB/depth/mask archive, indexed by original frame ID."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import queue
import re
import threading
import time
import zipfile
import cv2
import numpy as np


def save_npz_fast(path,**arrays):
    # Same standard lossless NPZ format; level 1 avoids expensive default level 6.
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.partial')
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for name,value in arrays.items():
                with archive.open(name+'.npy','w') as file:
                    np.lib.format.write_array(file,np.asanyarray(value),allow_pickle=False)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def timestamp_fields(frame):
    host=getattr(frame,'host_timestamp_s',None)
    suffix=frame.frame_id.rsplit('-',1)[-1]
    return dict(frame_number=int(suffix) if suffix.isdigit() else None,
                frame_number_source='frame_id suffix (RealSense color frame number; replay sequence in dataset tests)',
                camera_timestamp_ms=frame.camera_timestamp_ms,host_timestamp_s=host,
                capture_utc=datetime.fromtimestamp(host,timezone.utc).isoformat() if host is not None else None)


def frame_stats(snapshot):
    f=snapshot['frame'];r=snapshot['response']
    valid=(snapshot['filtered']>0)&np.isfinite(f.depth_m)&(f.depth_m>0)
    return dict(frame_id=f.frame_id,**timestamp_fields(f),source=getattr(f,'source','realsense_usb'),
        session=r['session'],generation=r['generation'],tracking_fps=snapshot['actual_fps'],
        mask_area_pixels=int(np.count_nonzero(snapshot['mask'])),mask_valid=bool(r['tracking_valid']),
        tracking_status=r['status'],grasp_allowed=bool(r['grasp_allowed']),
        valid_depth_pixels=int(np.count_nonzero(valid)),mask_depth_ms=r['mask_depth_ms'],
        tracking_rtt_ms=snapshot['tracking_rtt_ms'],sam_init_ms=r['metrics'].get('sam_init_ms'),
        vos_ms=r['metrics'].get('vos_ms'),archive_status='queued',frame_file='')


class FrameArchive:
    def __init__(self,path):
        self.path=Path(path)
        self.queue=queue.Queue(maxsize=4)
        self.rows=[]
        self.seen=set()
        self.lock=threading.Lock()
        self.files_condition=threading.Condition()
        self.writing=False
        self.priority_waiters=0
        self.closing=False
        self.dropped=0
        self.errors=[]
        self.thread=threading.Thread(target=self._run,name='rgbd-archive',daemon=True)
        self.thread.start()

    def offer(self,snapshot):
        if self.closing:return
        fid=snapshot['frame'].frame_id
        with self.lock:
            if fid in self.seen:return
            self.seen.add(fid)
            row=frame_stats(snapshot)
            self.rows.append(row)
        try:self.queue.put_nowait((snapshot,row))
        except queue.Full:
            try:
                _,old=self.queue.get_nowait()
                old['archive_status']='dropped_backlog'
                self.dropped+=1
                self.queue.task_done()
            except queue.Empty:pass
            try:self.queue.put_nowait((snapshot,row))
            except queue.Full:
                row['archive_status']='dropped_backlog'
                self.dropped+=1

    def save(self,snapshot,priority=False):
        f=snapshot['frame']
        safe=re.sub('[^a-zA-Z0-9_.-]','_',f.frame_id)[:96]
        relative=Path('frames')/(safe+'_'+hashlib.sha256(f.frame_id.encode()).hexdigest()[:10]+'.npz')
        target=self.path/relative
        with self.files_condition:
            if priority:self.priority_waiters+=1
            try:
                while self.writing or (not priority and self.priority_waiters):self.files_condition.wait()
                self.writing=True
            finally:
                if priority:self.priority_waiters-=1
        try:
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                ok,jpeg=cv2.imencode('.jpg',f.bgr,[cv2.IMWRITE_JPEG_QUALITY,90])
                if not ok:raise IOError('RGB archive encoding failed')
                metadata=dict(frame_stats(snapshot),intrinsics=f.intrinsics,color_intrinsics=f.color_intrinsics,
                              depth_scale=f.depth_scale,filters=vars(snapshot['config']),
                              rgb_encoding='JPEG90 bytes',mask_encoding='uint8 background=0 instance=1',
                              frame='camera_optical',units='meter')
                # One file per Frame ID avoids many small NFS/NTFS metadata writes.
                save_npz_fast(target,rgb_jpeg=jpeg,depth_raw=f.depth_raw,depth_m=f.depth_m,K=f.K,
                              depth_scale=f.depth_scale,instance_mask=(snapshot['mask']>0).astype(np.uint8),
                              filtered_mask=(snapshot['filtered']>0).astype(np.uint8),frame_id=f.frame_id,
                              metadata_json=json.dumps(metadata))
        finally:
            with self.files_condition:
                self.writing=False
                self.files_condition.notify_all()
        return relative.as_posix()

    def _run(self):
        while not self.closing or not self.queue.empty():
            try:snapshot,row=self.queue.get(timeout=.1)
            except queue.Empty:continue
            try:
                row['frame_file']=self.save(snapshot)
                row['archive_status']='saved'
            except Exception as exc:
                row['archive_status']='write_error'
                self.errors.append(dict(frame_id=row['frame_id'],error=str(exc)))
            finally:self.queue.task_done()

    def close(self):
        self.closing=True
        self.thread.join()  # Drain at most four outstanding frames; never discard pose context.
