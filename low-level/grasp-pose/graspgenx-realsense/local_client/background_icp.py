"""Bounded, continuous background-only ICP and supporting-plane RANSAC on CPU.

Current -> previous transforms; immutable Frame-ID results. No GPU/model imports.
Latest-frame queue prevents growing latency. Keyframes are local re-anchors, not SLAM.
"""
from collections import OrderedDict,deque
import csv
from dataclasses import asdict
import json
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import queue
import threading
import time
import cv2
import numpy as np
from pointcloud import deproject
from temporal_selection import TemporalConfig,pose_change,capture_time


def background_cloud(snapshot,cfg):
    f=snapshot['frame'];mask=(snapshot['mask']>0).astype(np.uint8)
    k=cfg.mask_dilation_pixels
    excluded=cv2.dilate(mask,np.ones((k,k),np.uint8))>0
    valid=(~excluded)&np.isfinite(f.depth_m)&(f.depth_m>.08)&(f.depth_m<4.)
    v,u=np.where(valid)
    if len(v)>cfg.max_background_points:
        ids=np.linspace(0,len(v)-1,cfg.max_background_points,dtype=int);v=v[ids];u=u[ids]
    z=f.depth_m[v,u];rays=getattr(f,'unit_rays',None)
    if rays is not None:return np.ascontiguousarray(rays[v,u]*z[:,None],dtype=np.float32)
    return np.column_stack(((u-f.K[0,2])*z/f.K[0,0],(v-f.K[1,2])*z/f.K[1,1],z)).astype(np.float32)


def estimate_support_plane(points,object_points,cfg,o3d):
    """Largest verified supporting surface. Without calibrated up, 'ground'
    is a geometric supporting-plane hypothesis, never a gravity measurement.
    """
    if len(points)<cfg.plane_min_inliers or len(object_points)<10:return None,dict(status='insufficient_points')
    if len(points)>cfg.max_plane_points:points=points[np.linspace(0,len(points)-1,cfg.max_plane_points,dtype=int)]
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(.012)
    initial=len(cloud.points);options=[]
    center=np.median(object_points,axis=0)
    for _ in range(4):
        if len(cloud.points)<cfg.plane_min_inliers:break
        plane,inliers=cloud.segment_plane(distance_threshold=cfg.plane_distance_m,ransac_n=3,num_iterations=150)
        xyz=np.asarray(cloud.points);plane=np.asarray(plane,float);plane/=np.linalg.norm(plane[:3])
        if plane[:3]@center+plane[3]<0:plane=-plane
        n=plane[:3];obj_distance=np.asarray(object_points)@n+plane[3]
        support=xyz[inliers];rmse=float(np.sqrt(np.mean((support@n+plane[3])**2)))
        normal_ok=True
        if cfg.ground_up_camera is not None:
            up=np.asarray(cfg.ground_up_camera,float);up/=np.linalg.norm(up)
            normal_ok=bool(n@up>=np.cos(np.deg2rad(cfg.ground_max_tilt_deg)))
        # Object projection must fall inside the observed supporting surface footprint.
        seed=np.array([1.,0,0]) if abs(n[0])<.9 else np.array([0.,1,0])
        x=np.cross(n,seed);x/=np.linalg.norm(x);y=np.cross(n,x)
        footprint=np.column_stack((support@x,support@y)).astype(np.float32)
        hull=cv2.convexHull(footprint)
        inside=cv2.pointPolygonTest(hull,(float(center@x),float(center@y)),True)>=-.03
        ok=(len(inliers)>=cfg.plane_min_inliers and len(inliers)/max(initial,1)>=cfg.plane_min_fraction
            and rmse<=cfg.plane_max_rmse_m and np.percentile(obj_distance,5)>=-cfg.plane_distance_m
            and .003<=np.median(obj_distance)<=cfg.plane_max_object_height_m
            and plane[3]>0 and inside and normal_ok)
        if ok:
            options.append((len(inliers),plane,rmse))
            if len(inliers)>.56*initial:break  # Remaining points cannot form an equally large ambiguous plane.
        cloud=cloud.select_by_index(inliers,invert=True)
    if not options:return None,dict(status='no_verified_support_plane')
    options.sort(key=lambda item:-item[0]);count,plane,rmse=options[0]
    if len(options)>1 and options[1][0]>.8*count and abs(options[1][1][:3]@plane[:3])<.95:
        return None,dict(status='ambiguous_support_planes')
    return plane,dict(status='calibrated_up_support' if cfg.ground_up_camera is not None else 'dominant_support_hypothesis',
                      inliers=count,fraction=count/max(initial,1),rmse_m=rmse)


class BackgroundICP:
    def __init__(self,path,config=None):
        import open3d as o3d
        self.o3d=o3d;self.cfg=config or TemporalConfig();self.path=path
        self.queue=queue.Queue(maxsize=1);self.condition=threading.Condition()
        self.results=OrderedDict();self.rows=[];self.drops=0;self.stop=threading.Event()
        self.previous=None;self.keyframes=deque(maxlen=self.cfg.max_keyframes)
        self.accepted=0;self.processed=0;self.reference=0;self.first=None;self.last_completed=None
        self.thread=threading.Thread(target=self._run,name='background-icp',daemon=True);self.thread.start()

    def offer(self,snapshot):
        if self.stop.is_set():return
        try:self.queue.put_nowait(snapshot)
        except queue.Full:
            try:self.queue.get_nowait();self.queue.task_done();self.drops+=1
            except queue.Empty:pass
            try:self.queue.put_nowait(snapshot)
            except queue.Full:self.drops+=1

    def lookup(self,frame_id,wait_s=0.):
        deadline=time.perf_counter()+wait_s
        with self.condition:
            while frame_id not in self.results and not self.stop.is_set():
                remaining=deadline-time.perf_counter()
                if remaining<=0:break
                self.condition.wait(remaining)
            return self.results.get(frame_id)

    def _cloud(self,points):
        c=self.o3d.geometry.PointCloud(self.o3d.utility.Vector3dVector(points))
        c=c.voxel_down_sample(self.cfg.voxel_m)
        if len(c.points)>self.cfg.max_background_points:
            ids=np.linspace(0,len(c.points)-1,self.cfg.max_background_points,dtype=int);c=c.select_by_index(ids)
        c.estimate_normals(self.o3d.geometry.KDTreeSearchParamHybrid(radius=self.cfg.voxel_m*3,max_nn=30))
        return c

    def _register(self,source,target,initial):
        o=self.o3d;c=self.cfg
        reg=o.pipelines.registration.registration_icp(source,target,c.icp_correspondence_m,initial,
            o.pipelines.registration.TransformationEstimationPointToPlane(),
            o.pipelines.registration.ICPConvergenceCriteria(max_iteration=20))
        T=reg.transformation
        normals=np.asarray(target.normals)
        eigen=float(np.linalg.eigvalsh(normals.T@normals/max(len(normals),1))[0])
        finite=np.isfinite(T).all() and np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-3) and np.linalg.det(T[:3,:3])>.999
        ok=(finite and reg.fitness>=c.icp_min_fitness and reg.inlier_rmse<=c.icp_max_rmse_m and eigen>=c.min_normal_eigenvalue)
        return T,dict(fitness=float(reg.fitness),rmse_m=float(reg.inlier_rmse),normal_min_eigenvalue=eigen),bool(ok)

    def process(self,snapshot):
        start=time.perf_counter();f=snapshot['frame'];c=self.cfg
        clock,stamp=capture_time(f)
        result=dict(capture_clock=clock,capture_time_s=stamp,frame_id=f.frame_id,capture_monotonic=f.ready_monotonic,camera_timestamp_ms=f.camera_timestamp_ms,
                    host_timestamp_s=getattr(f,'host_timestamp_s',None),valid=False,T_world_camera=None,
                    plane=None,plane_metrics=dict(status='not_estimated'),status='tracking_invalid',reference=self.reference)
        if not snapshot['response']['tracking_valid']:return result
        bg=background_cloud(snapshot,c)
        obj,_,_=deproject(f.depth_m,f.K,snapshot['mask']>0,getattr(f,'unit_rays',None))
        pre=time.perf_counter();cloud=self._cloud(bg)
        plane_start=time.perf_counter();plane,pm=estimate_support_plane(bg,obj,c,self.o3d)
        result.update(plane=plane,plane_metrics=pm,background_points=len(cloud.points),preprocess_ms=(pre-start)*1000,
                      plane_ms=(time.perf_counter()-plane_start)*1000,status='insufficient_background')
        if len(cloud.points)<c.min_icp_points:return result
        registration_start=time.perf_counter()
        if self.previous is None:
            # The reference transform is a convention, not a measured motion.
            T=np.eye(4);quality=dict(fitness=1.,rmse_m=0.);ok=True;status='reference'
        else:
            previous=self.previous
            if clock!=previous['clock'] or stamp<=previous['time']:result['status']='nonmonotonic_capture';return result
            if f.depth_m.shape!=previous['shape'] or not np.allclose(f.K,previous['K'],atol=1e-5):
                result['status']='calibration_changed';return result
            relative,quality,ok=self._register(cloud,previous['cloud'],np.eye(4))
            mm,deg=pose_change(np.eye(4),relative)
            ok=ok and mm<=c.icp_max_step_m*1000 and deg<=c.icp_max_step_deg
            T=previous['pose']@relative;status='relative' if ok else 'icp_rejected'
            if ok and self.accepted%c.keyframe_realign_every==0:
                # Prefer an older overlapping anchor, reducing long relative chains.
                candidates=sorted(self.keyframes,key=lambda k:k['count'])
                reanchored=False
                for key in candidates:
                    if np.linalg.norm(key['pose'][:3,3]-T[:3,3])>.5:continue
                    initial=np.linalg.inv(key['pose'])@T
                    rel,q,verified=self._register(cloud,key['cloud'],initial)
                    corrected=key['pose']@rel;delta,angle=pose_change(T,corrected)
                    if verified and delta<=c.max_keyframe_correction_m*1000 and angle<=c.max_keyframe_correction_deg:
                        T=corrected;quality=q;status='keyframe';reanchored=True;break
                if not reanchored:ok=False;status='keyframe_rejected'
        result.update(**quality,valid=ok,status=status,icp_ms=(time.perf_counter()-registration_start)*1000)
        if ok:
            self.accepted+=1
            self.previous=dict(cloud=cloud,pose=T.copy(),time=stamp,clock=clock,K=f.K.copy(),shape=f.depth_m.shape,count=self.accepted)
            if not self.keyframes or self.accepted%c.keyframe_interval==0:self.keyframes.append(dict(self.previous))
            result['T_world_camera']=T
        return result

    def _run(self):
        try:
            from threadpoolctl import threadpool_limits
            limiter=threadpool_limits(limits=2)
        except ImportError:limiter=None
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:snapshot=self.queue.get(timeout=.05)
                except queue.Empty:continue
                start=time.perf_counter()
                try:r=self.process(snapshot)
                except Exception as exc:r=dict(frame_id=snapshot['frame'].frame_id,valid=False,T_world_camera=None,plane=None,
                                                plane_metrics=dict(status='error'),status=f'error:{exc}')
                done=time.perf_counter();r['processing_ms']=(done-start)*1000
                r['capture_to_icp_ms']=(done-snapshot['frame'].ready_monotonic)*1000
                r['completed_monotonic']=done
                if self.first is None:self.first=start
                self.processed+=1;r['icp_processing_fps']=1000/r['processing_ms'] if r['processing_ms'] else 0
                r['icp_throughput_fps']=self.processed/max(done-self.first,1e-8)
                r['icp_only_fps']=1000/r['icp_ms'] if r.get('icp_ms',0)>0 and r['status']!='reference' else None
                with self.condition:
                    self.results[r['frame_id']]=r
                    while len(self.results)>self.cfg.result_cache_size:self.results.popitem(last=False)
                    self.condition.notify_all()
                row={k:v for k,v in r.items() if k not in ('T_world_camera','plane','plane_metrics')}
                row.update(plane_status=r['plane_metrics']['status'],queue_drops=self.drops)
                row.update({f't{a}{b}':float(r['T_world_camera'][a,b]) if r['valid'] else None for a in range(4) for b in range(4)})
                self.rows.append(row)
                self.queue.task_done()
        finally:
            if limiter is not None:limiter.restore_original_limits()

    def close(self):
        self.stop.set();self.thread.join()
        if self.rows:
            fields=list(dict.fromkeys(k for r in self.rows for k in r))
            with (self.path/'icp_frames.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(self.rows)
        poses=[(r['frame_id'],r) for r in self.rows if r['valid']]
        if poses:np.savez_compressed(self.path/'continuous_icp.npz',frame_ids=[fid for fid,_ in poses],
                          T_world_camera=np.array([[[r[f't{a}{b}'] for b in range(4)] for a in range(4)] for _,r in poses]))
        (self.path/'icp_summary.json').write_text(json.dumps(dict(processed=self.processed,accepted=self.accepted,
            queue_drops=self.drops,config=asdict(self.cfg),note='CPU processing of tracked frames; measured throughput is not camera FPS'),indent=2))
