"""CPU RGB-D camera odometry for offline fixed-object visualization.

T_reference_camera maps a camera point into the first grasp frame. Original
camera-relative grasps and RGB-D files are never changed. No IMU is assumed.
"""
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
import time
import cv2
import numpy as np

VERSION='rgbd-camera-v2'


def transform_points(points,T):
    return np.asarray(points)@T[:3,:3].T+T[:3,3]


def compose_camera_pose(T_reference_previous,T_previous_current):
    return np.asarray(T_reference_previous)@np.asarray(T_previous_current)


def read_rgbd(path):
    with np.load(path,allow_pickle=False) as d:
        rgb=cv2.imdecode(d['rgb_jpeg'],cv2.IMREAD_COLOR)[:,:,::-1].copy()
        depth=d['depth_m'].astype(np.float32).copy();K=d['K'].astype(np.float64)
        fid=str(d['frame_id'])
        meta=json.loads(str(d['metadata_json'])) if 'metadata_json' in d else {}
    if rgb.shape[:2]!=depth.shape or K.shape!=(3,3):raise ValueError('RGB/depth/K shape mismatch')
    depth[~np.isfinite(depth)|(depth<=.08)|(depth>4.)]=0
    return dict(rgb=rgb,depth=depth,K=K,frame_id=fid,distortion_coeffs=meta.get('color_intrinsics',{}).get('coeffs',[]))


def collect_frames(path,pose_rows):
    path=Path(path);entries={}
    log=path/'tracking_frames.csv'
    if log.exists():
        with log.open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                name=row.get('frame_file','')
                if name and (path/name).exists():entries[row['frame_id']]=name
    for row in pose_rows:
        if not (path/row['frame_file']).exists():raise FileNotFoundError(row['frame_file'])
        entries[row['frame_id']]=row['frame_file']
    items=list(entries.items())
    prefixes={fid.rsplit('-',1)[0] for fid,_ in items}
    if len(prefixes)==1 and all(fid.rsplit('-',1)[-1].isdigit() for fid,_ in items):
        items.sort(key=lambda item:int(item[0].rsplit('-',1)[-1]))
    anchor=next(i for i,(fid,_) in enumerate(items) if fid==pose_rows[0]['frame_id'])
    return items[anchor:]


class RGBDOdometry:
    def __init__(self,frame):
        os.environ.setdefault('OMP_NUM_THREADS','2')
        if np.any(frame.get('distortion_coeffs',[])):raise ValueError('Rectified RGB-D required; nonzero distortion')
        try:import open3d as o3d
        except ImportError as exc:
            raise RuntimeError('Install CPU visualization dependency: python -m pip install open3d==0.19.0') from exc
        self.o3d=o3d
        h,w=frame['depth'].shape;K=frame['K']
        self.K=K;self.shape=(h,w)
        scale=min(1.,640/w,480/h)
        self.processing_size=(max(1,round(w*scale)),max(1,round(h*scale)))
        width,height=self.processing_size
        scaled=K.copy();scaled[0]*=width/w;scaled[1]*=height/h
        self.intrinsic=o3d.camera.PinholeCameraIntrinsic(width,height,float(scaled[0,0]),float(scaled[1,1]),float(scaled[0,2]),float(scaled[1,2]))
        self.option=o3d.pipelines.odometry.OdometryOption()
        self.option.depth_min=.08;self.option.depth_max=4.;self.option.depth_diff_max=.07
        self.jacobian=o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        self.previous=self.prepare(frame)
        self.pose=np.eye(4);self.guess=np.eye(4)

    def prepare(self,frame):
        o=self.o3d
        rgb_array=frame['rgb'];depth_array=frame['depth']
        if depth_array.shape[::-1]!=self.processing_size:
            rgb_array=cv2.resize(rgb_array,self.processing_size,interpolation=cv2.INTER_AREA)
            depth_array=cv2.resize(depth_array,self.processing_size,interpolation=cv2.INTER_NEAREST)
        rgb=o.geometry.Image(np.ascontiguousarray(rgb_array,dtype=np.uint8))
        depth=o.geometry.Image(np.ascontiguousarray(depth_array,dtype=np.float32))
        pair=o.geometry.RGBDImage.create_from_color_and_depth(rgb,depth,depth_scale=1.,depth_trunc=4.,convert_rgb_to_intensity=True)
        cloud=o.geometry.PointCloud.create_from_rgbd_image(pair,self.intrinsic).voxel_down_sample(.012)
        return pair,cloud

    def step(self,frame):
        start=time.perf_counter()
        if np.any(frame.get('distortion_coeffs',[])):
            return None,dict(status='nonzero_distortion',fitness=0.,rmse_m=None,elapsed_ms=0.)
        if frame['depth'].shape!=self.shape or not np.allclose(frame['K'],self.K,rtol=0,atol=1e-5):
            return None,dict(status='calibration_changed',fitness=0.,rmse_m=None,elapsed_ms=0.)
        current=self.prepare(frame);source=current[1];target=self.previous[1];o=self.o3d
        if min(len(source.points),len(target.points))<300:
            return None,dict(status='insufficient_depth',fitness=0.,rmse_m=None,elapsed_ms=(time.perf_counter()-start)*1000)
        # Source=current, target=previous: result is T_previous_current.
        ok,relative,info=o.pipelines.odometry.compute_rgbd_odometry(current[0],self.previous[0],self.intrinsic,
                            self.guess,self.jacobian,self.option)
        method='rgbd'
        if not ok or not np.isfinite(relative).all():relative=None
        def quality(T):
            if T is None:return False,0.,None
            translation=np.linalg.norm(T[:3,3])
            angle=np.degrees(np.arccos(np.clip((np.trace(T[:3,:3])-1)/2,-1,1)))
            evaluation=o.pipelines.registration.evaluate_registration(source,target,.025,T)
            valid=(evaluation.fitness>=.35 and evaluation.inlier_rmse<=.018 and translation<=.25 and angle<=30
                   and np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-3)
                   and np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-3))
            return bool(valid),float(evaluation.fitness),float(evaluation.inlier_rmse)
        valid,fitness,rmse=quality(relative)
        if not valid:
            # Geometric refinement fallback; no guessed transform is accepted without validation.
            target.estimate_normals(o.geometry.KDTreeSearchParamHybrid(radius=.04,max_nn=30))
            normals=np.asarray(target.normals)
            supported_normals=np.linalg.eigvalsh(normals.T@normals/max(len(normals),1))[0]>.003
            reg=o.pipelines.registration.registration_icp(source,target,.05,self.guess,
                o.pipelines.registration.TransformationEstimationPointToPlane(),
                o.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
            relative=reg.transformation;valid,fitness,rmse=quality(relative);method='icp'
            if valid and not supported_normals:valid=False
            if valid:
                information=o.pipelines.registration.get_information_matrix_from_point_clouds(source,target,.025,relative)
                if np.linalg.cond(information)>1e9:valid=False
        metrics=dict(status=method if valid else 'rejected',fitness=fitness,rmse_m=rmse,elapsed_ms=(time.perf_counter()-start)*1000)
        if not valid:return None,metrics
        self.pose=compose_camera_pose(self.pose,relative)
        self.previous=current;self.guess=relative
        return self.pose.copy(),metrics


def build_camera_motion(path,pose_rows,progress=None):
    path=Path(path);items=collect_frames(path,pose_rows)
    signature=hashlib.sha256(json.dumps(dict(version=VERSION,inputs=[(fid,name,(path/name).stat().st_size,(path/name).stat().st_mtime_ns)
                                    for fid,name in items]),sort_keys=True).encode()).hexdigest()
    folder=path/'camera_motion';folder.mkdir(exist_ok=True)
    cached=folder/'trajectory.npz';meta_file=folder/'report.json'
    if cached.exists() and meta_file.exists() and json.loads(meta_file.read_text()).get('signature')==signature:
        with np.load(cached,allow_pickle=False) as data:result={k:data[k].copy() for k in data.files}
        export_aligned_grasps(path,pose_rows,result)
        if progress:progress(len(items),len(items),'cached')
        return result
    poses=[];valid=[];metrics=[];last_valid=-1;segments=[];segment=0
    first=read_rgbd(path/items[0][1]);assert first['frame_id']==items[0][0]
    engine=RGBDOdometry(first)
    for i,(fid,name) in enumerate(items):
        if progress:progress(i,len(items),fid)
        if i==0:
            T=np.eye(4);m=dict(status='reference',fitness=1.,rmse_m=0.,elapsed_ms=0.)
        elif fid.rsplit('-',1)[0]!=items[0][0].rsplit('-',1)[0]:
            T=None;m=dict(status='camera_stream_changed',fitness=0.,rmse_m=None,elapsed_ms=0.)
        else:
            try:
                frame=read_rgbd(path/name)
                if frame['frame_id']!=fid:raise ValueError('Archive Frame ID mismatch')
                T,m=engine.step(frame)
            except Exception as exc:T=None;m=dict(status=f'error:{exc}',fitness=0.,rmse_m=None,elapsed_ms=0.)
        if T is not None:
            if i and last_valid!=i-1:segment+=1
            last_valid=i
        poses.append(T if T is not None else np.full((4,4),np.nan));valid.append(T is not None)
        metrics.append(dict(frame_id=fid,**m));segments.append(segment)
    result=dict(frame_ids=np.asarray([fid for fid,_ in items]),T_reference_camera=np.asarray(poses),
                valid=np.asarray(valid,dtype=bool),segments=np.asarray(segments,dtype=np.int32),
                status=np.asarray([m['status'] for m in metrics]))
    with tempfile.NamedTemporaryFile(dir=folder,prefix='trajectory_',suffix='.partial',delete=False) as handle:
        temporary=Path(handle.name);np.savez_compressed(handle,**result)
    temporary.replace(cached)
    report=dict(signature=signature,version=VERSION,reference_frame_id=items[0][0],frames=len(items),valid=sum(valid),
        frame='first grasp camera optical frame; metric; not gravity aligned',
        convention='T_reference_gripper = T_reference_camera @ T_camera_gripper',
        sensor='Intel RealSense D435i RGB-D; IMU was not recorded and is not used',
        assumption='Target and background remain stationary; no loop closure; drift is possible',
        validation=dict(min_fitness=.35,max_rmse_m=.018,max_step_translation_m=.25,max_step_rotation_deg=30,
                        icp_fallback_requires_nonplanar_normals=True),frames_detail=metrics)
    meta_file.write_text(json.dumps(report,indent=2),encoding='utf-8')
    export_aligned_grasps(path,pose_rows,result)
    if progress:progress(len(items),len(items),'complete')
    return result


def export_aligned_grasps(path,pose_rows,motion):
    from official_viser_helpers import matrix_to_wxyz_position
    path=Path(path);lookup={str(fid):i for i,fid in enumerate(motion['frame_ids'])}
    records=[];previous=None
    for row in pose_rows:
        i=lookup.get(row['frame_id']);ok=i is not None and bool(motion['valid'][i])
        record=dict(frame_id=row['frame_id'],reference_frame_id=str(motion['frame_ids'][0]),
                    camera_alignment_valid=ok,status=str(motion['status'][i]) if i is not None else 'missing_frame',
                    C=row['C'],R=row['R'],P=row['P'],Y=row['Y'],S=row['S'])
        for key in ('x_m','y_m','z_m','qx','qy','qz','qw','position_delta_mm','rotation_delta_deg'):record['reference_'+key]=None
        for a in range(4):
            for b in range(4):record[f'reference_t{a}{b}']=None
        if ok:
            camera_grasp=np.array([[float(row[f't{a}{b}']) for b in range(4)] for a in range(4)])
            T=motion['T_reference_camera'][i]@camera_grasp
            q,p=matrix_to_wxyz_position(T);q=np.asarray(q,float);q/=np.linalg.norm(q)
            record.update(reference_x_m=p[0],reference_y_m=p[1],reference_z_m=p[2],
                          reference_qx=q[1],reference_qy=q[2],reference_qz=q[3],reference_qw=q[0])
            record.update({f'reference_t{a}{b}':T[a,b] for a in range(4) for b in range(4)})
            segment=(row.get('stability_segment'),int(motion['segments'][i]))
            if previous is not None and previous[0]==segment:
                record['reference_position_delta_mm']=float(np.linalg.norm(p-previous[1])*1000)
                record['reference_rotation_delta_deg']=float(2*np.degrees(np.arccos(np.clip(abs(q@previous[2]),0,1))))
            previous=(segment,p.copy(),q.copy())
        else:previous=None
        records.append(record)
    target=path/'camera_motion/aligned_grasps.csv'
    with target.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
