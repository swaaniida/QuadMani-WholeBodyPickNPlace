"""Continuous single-flight grasp scheduling and top-1-only session recording.

No model or inference RPC is implemented here. Uses the existing TrackingWorker.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import threading
import cv2
import numpy as np
from grasp_ranking import rerank_result
from frame_archive import FrameArchive, frame_stats, timestamp_fields
from official_viser_helpers import matrix_to_wxyz_position
from pointcloud import deproject, downsample


def cloud_from_snapshot(snapshot):
    f=snapshot['frame']
    pc,v,u=deproject(f.depth_m,f.K,snapshot['filtered']>0,f.unit_rays)
    colors=f.bgr[v,u,::-1].copy()
    cfg=snapshot['config']
    return downsample(pc,colors,cfg.voxel_m,cfg.max_points)


class Top1Session:
    def __init__(self, output, settings=None):
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.path=Path(output)/'auto_sessions'/f'{stamp}_{uuid.uuid4().hex[:6]}'
        self.path.mkdir(parents=True,exist_ok=False)
        (self.path/'samples').mkdir()
        self.epoch=time.perf_counter()
        self.first_request=None
        self.rows=[]
        self.requests=0
        self.failures=0
        self.closed=False
        self.settings=dict(settings or {})
        self.archive=FrameArchive(self.path)
        self.segment=0
        self.segment_lock=threading.Lock()
        self.was_valid=False
        self.previous=None
        (self.path/'session.json').write_text(json.dumps(dict(
            created_utc=stamp,settings=self.settings,formula='S=0.6C+0.1R+0.25P+0.05Y',
            frame='camera_optical',pose='T_camera_gripper_base; not robot TCP',
            stored_predictions='Exactly one maximum-S pose per successful nonempty inference',
            context='One frames/<FrameID>.npz per tracked frame: RGB JPEG90 bytes, lossless raw+meter depth, binary masks, calibration; pose records: top1 and object cloud only',
            history='Camera-relative samples; no camera motion compensation or world trajectory',
            clock='All throughput and client duration values use this process monotonic clock'),indent=2),encoding='utf-8')
        self.event('created')

    def event(self,name,**details):
        with (self.path/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(event=name,elapsed_s=time.perf_counter()-self.epoch,**details))+'\n')

    def offer_frame(self,snapshot):
        if self.closed:return
        with self.segment_lock:
            valid=bool(snapshot['response']['tracking_valid'])
            if self.was_valid and not valid:self.segment+=1
            self.was_valid=valid
            snapshot['_recording_segment']=self.segment
        self.archive.offer(snapshot)

    def break_stability(self):
        with self.segment_lock:self.segment+=1

    def requested(self,when,frame_id):
        if self.first_request is None:self.first_request=when
        self.requests+=1
        self.event('request',frame_id=frame_id)

    def record(self,completed):
        if self.closed:raise RuntimeError('Recording session is closed')
        received=time.perf_counter()
        snapshot=completed['snapshot'];raw=completed['result'];f=snapshot['frame']
        if raw['frame_id']!=f.frame_id:raise ValueError('Result/snapshot frame mismatch')
        if not len(raw['grasps']):
            self.event('empty_grasps',frame_id=f.frame_id)
            return None
        start=time.perf_counter()
        points,colors=cloud_from_snapshot(snapshot)
        result=rerank_result(raw,points)
        rank_ms=(time.perf_counter()-start)*1000
        T=np.asarray(result['grasps'][0],dtype=np.float32)
        C=float(result['confidences'][0]);S=float(result['selection_scores'][0])
        R,P,Y=map(float,result['orientation_scores'][0])
        wxyz,pos=matrix_to_wxyz_position(T)
        wxyz=np.asarray(wxyz,dtype=np.float64)
        wxyz=wxyz/np.linalg.norm(wxyz)
        index=len(self.rows)
        file=f'samples/{index:06d}.npz'
        write_start=time.perf_counter()
        # Only the chosen pose is persisted; all other model predictions are dropped.
        np.savez_compressed(self.path/file,pose=T,confidence=C,score=S,orientation_scores=[R,P,Y],
            point_cloud=points,colors=colors,K=f.K,frame_id=f.frame_id,
            original_candidate_index=result['original_candidate_indices'][0])
        # Always preserve context for every stored pose even if frame archive fell behind.
        self.archive.offer(snapshot)
        frame_file=self.archive.save(snapshot,priority=True)
        write_ms=(time.perf_counter()-write_start)*1000
        done=time.perf_counter()
        response=snapshot['response'];metrics=response['metrics']
        segment=f'{response["session"]}:{response["generation"]}:{snapshot.get("_recording_segment",0)}'
        delta_mm=delta_deg=pose_dt_ms=None
        if self.previous is not None and self.previous['segment']==segment:
            previous=self.previous
            delta_mm=float(np.linalg.norm(T[:3,3]-previous['pose'][:3,3])*1000)
            # Quaternion sign is arbitrary: q and -q represent the same rotation.
            delta_deg=float(2*np.degrees(np.arccos(np.clip(abs(previous['quaternion']@wxyz),0,1))))
            pose_dt_ms=(f.ready_monotonic-previous['frame_ready'])*1000
        record_elapsed=done-self.epoch
        elapsed=done-self.first_request if self.first_request is not None else 0
        instant_fps=1/(record_elapsed-self.rows[-1]['completed_elapsed_s']) if self.rows and record_elapsed>self.rows[-1]['completed_elapsed_s'] else None
        tracking_stats=frame_stats(snapshot)
        row=dict(index=index,frame_id=f.frame_id,session=response['session'],generation=response['generation'],
            file=file,completed_elapsed_s=done-self.epoch,requested_elapsed_s=completed['requested_at']-self.epoch,
            source=getattr(f,'source','realsense_usb'),**timestamp_fields(f),frame_file=frame_file,
            x_m=float(pos[0]),y_m=float(pos[1]),z_m=float(pos[2]),
            qx=float(wxyz[1]),qy=float(wxyz[2]),qz=float(wxyz[3]),qw=float(wxyz[0]),
            C=C,R=R,P=P,Y=Y,S=S,candidate_count=len(raw['grasps']),
            selected_original_index=int(result['original_candidate_indices'][0]),
            candidate_count_generated=result.get('candidate_count_generated'),
            candidate_count_after_threshold=result.get('candidate_count_after_threshold'),
            candidate_count_after_filter=result.get('candidate_count_after_filter',len(raw['grasps'])),
            requested_num_grasps=self.settings.get('num_grasps'),
            mask_area_pixels=tracking_stats['mask_area_pixels'],mask_valid=tracking_stats['mask_valid'],
            valid_depth_pixels=tracking_stats['valid_depth_pixels'],point_count=len(points),
            pose_generation_fps=(index+1)/elapsed if elapsed>0 else 0,
            instantaneous_pose_fps=instant_fps,
            end_to_end_latency_ms=(done-f.ready_monotonic)*1000,
            stability_segment=segment,position_delta_mm=delta_mm,rotation_delta_deg=delta_deg,pose_dt_ms=pose_dt_ms,
            acquisition_ms=f.acquisition_ms,alignment_ms=f.alignment_ms,
            frame_encode_ms=snapshot.get('frame_encode_ms'),tracking_rtt_ms=snapshot['tracking_rtt_ms'],
            sam_init_ms=metrics.get('sam_init_ms'),vos_ms=metrics.get('vos_ms'),
            mask_depth_ms=response['mask_depth_ms'],grasp_infer_ms=result['server_timing']['infer_ms'],
            grasp_rpc_rtt_ms=result['grasp_rpc_rtt_ms'],
            result_poll_rtt_ms=completed.get('poll_rtt_ms'),
            request_to_result_ms=(received-completed['requested_at'])*1000,
            pointcloud_and_ranking_ms=rank_ms,record_write_ms=write_ms,
            request_to_record_ms=(done-completed['requested_at'])*1000,
            capture_to_record_ms=(done-f.ready_monotonic)*1000,
            tracking_fps=snapshot['actual_fps'])
        # Full SE(3), not just quaternion, is also available in the CSV.
        row.update({f't{r}{c}':float(T[r,c]) for r in range(4) for c in range(4)})
        with (self.path/'poses.csv').open('a',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row))
            if not self.rows:writer.writeheader()
            writer.writerow(row)
        self.rows.append(row)
        self.previous=dict(segment=segment,pose=T.copy(),quaternion=wxyz.copy(),frame_ready=f.ready_monotonic)
        return row

    def fps(self):
        if not self.rows or self.first_request is None:return 0.
        elapsed=self.rows[-1]['completed_elapsed_s']-(self.first_request-self.epoch)
        return len(self.rows)/elapsed if elapsed>0 else 0.

    def stats(self):
        rows=self.rows;n=len(rows)
        elapsed=(rows[-1]['completed_elapsed_s']-(self.first_request-self.epoch)) if n and self.first_request is not None else 0
        intervals=np.diff([r['completed_elapsed_s'] for r in rows])
        latency={}
        if rows:
            for key in rows[0]:
                if key.endswith('_ms'):
                    values=[r[key] for r in rows if r[key] is not None]
                    if values:latency[key]=dict(mean=float(np.mean(values)),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)))
        infer=latency.get('grasp_infer_ms',{}).get('mean',0)
        return dict(poses=n,requests=self.requests,failures=self.failures,
            first_request_to_last_record_s=elapsed,
            completed_pose_fps=n/elapsed if elapsed>0 else 0,
            steady_completion_fps=(n-1)/float(np.sum(intervals)) if n>1 and np.sum(intervals)>0 else None,
            model_only_fps_estimate=1000/infer if infer>0 else None,
            completion_interval_ms=dict(p50=float(np.percentile(intervals,50)*1000),p95=float(np.percentile(intervals,95)*1000)) if n>1 else None,
            latency=latency,
            fps_definition='completed_pose_fps = saved top1 count / (last record completion - first auto request); includes pauses/loss between them; excludes initial click wait and idle after last result',
            model_only_note='reciprocal mean server inference time, not end-to-end measured throughput',
            windows_hardware_tested=self.settings.get('windows_hardware_tested',False),
            tracking_frames=len(self.archive.rows),archive_backlog_drops=self.archive.dropped,
            archive_errors=self.archive.errors,
            archived_frames=sum(bool(r['frame_file']) for r in self.archive.rows),
            unarchived_frames=sum(not bool(r['frame_file']) for r in self.archive.rows),
            candidate_count_unknown_records=sum(r['candidate_count_generated'] is None for r in rows))

    def close(self):
        if not self.closed:
            self.archive.close()
            # Include frames archived on demand for a top1 after a queue drop.
            pose_by_frame={r['frame_id']:r for r in self.rows}
            for row in self.archive.rows:
                if row['frame_id'] in pose_by_frame:
                    row['frame_file']=pose_by_frame[row['frame_id']]['frame_file']
                    row['archive_status']='saved_pose_context'
            for filename,rows in (
                ('tracking_frames.csv',self.archive.rows),
                ('frames_with_poses.csv',[dict(r,**{k:v for k,v in pose_by_frame.get(r['frame_id'],{}).items() if k not in r},
                                               pose_available=r['frame_id'] in pose_by_frame) for r in self.archive.rows])):
                if rows:
                    fields=list(dict.fromkeys(k for r in rows for k in r))
                    with (self.path/filename).open('w',newline='',encoding='utf-8') as handle:
                        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            self.event('closed')
            (self.path/'summary.json').write_text(json.dumps(self.stats(),indent=2),encoding='utf-8')
            self.closed=True
        return self.stats()


class ContinuousGrasp:
    """One outstanding job; no backlog; next request uses a distinct fresh frame."""
    def __init__(self,session,enabled=True):
        self.session=session
        self.enabled=enabled
        self.inflight=False
        self.revision=None
        self.last_frame=None
        self.started=None
        self.last_error=None
        self.retry_after=0.
        self.failure_streak=0

    def tick(self,worker):
        error=worker.take_grasp_error() if hasattr(worker,'take_grasp_error') else None
        if error is not None:
            self.cancel(error['message'],break_stability=False)
            self.last_error=error['message']
            self.failure_streak+=1
            self.retry_after=time.perf_counter()+min(4.,.25*2**min(self.failure_streak,4))
        with worker.lock:
            revision=worker.revision
            pending=worker.grasp_pending is not None
        latest,status,busy=worker.state()
        if self.inflight and revision!=self.revision:
            self.cancel('selection changed')
        if self.inflight and not busy and not pending and latest is None and 'error' in status.lower():
            self.cancel(status)
        if not self.enabled or self.inflight or busy or pending or latest is None or time.perf_counter()<self.retry_after:return False
        f=latest['frame']
        if (not latest['response']['grasp_allowed'] or time.perf_counter()-f.ready_monotonic>2
                or f.frame_id==self.last_frame):return False
        try:
            worker.request_grasp()
        except ValueError:return False
        self.inflight=True
        self.revision=revision
        self.last_frame=f.frame_id
        self.last_error=None
        now=time.perf_counter()
        if self.started is None:self.started=now
        self.session.requested(now,f.frame_id)
        return True

    def accept(self,completed):
        self.inflight=False
        self.failure_streak=0
        self.last_error=None
        try:
            return self.session.record(completed)
        except Exception as exc:
            self.session.failures+=1
            self.session.event('record_error',message=str(exc))
            self.enabled=False  # Stop scheduling if persistence/validation fails.
            self.last_error=str(exc)
            raise

    def cancel(self,reason,break_stability=True):
        if self.inflight:
            self.session.failures+=1
            self.session.event('cancelled',reason=reason)
        self.inflight=False
        self.last_frame=None
        if break_stability:self.session.break_stability()

    def toggle(self):
        self.enabled=not self.enabled
        self.session.break_stability()
        self.session.event('enabled' if self.enabled else 'paused')
