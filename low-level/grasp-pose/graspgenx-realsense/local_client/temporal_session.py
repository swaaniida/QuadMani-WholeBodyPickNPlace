"""Opt-in recording adapter; legacy Top1Session and model code stay intact."""
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from auto_grasp import Top1Session,cloud_from_snapshot
from frame_archive import frame_stats,timestamp_fields
from official_viser_helpers import matrix_to_wxyz_position
from pointcloud import deproject
from background_icp import BackgroundICP,background_cloud,estimate_support_plane
from temporal_selection import TemporalConfig,select_candidates,pose_change,capture_time,same_capture_sequence


class TemporalSession(Top1Session):
    def __init__(self,output,settings=None):
        self.config=TemporalConfig.load((settings or {}).get('temporal_config'))
        # Fail before opening an archive if CPU registration support is missing.
        import open3d
        import trimesh
        self.vertices=np.asarray(trimesh.load(Path(__file__).parent/'assets/franka_umi_vis_mesh.obj',force='mesh',process=False).vertices)
        super().__init__(output,settings)
        (self.path/'all_candidates').mkdir()
        self.icp=BackgroundICP(self.path,self.config)
        self.previous_new=None;self.previous_baseline=None;self.previous_time=None;self.previous_generation=None;self.previous_clock=None
        self.comparisons=[];self.icp_closed=False;self.duration_started=None
        meta=json.loads((self.path/'session.json').read_text())
        meta.update(formula='S=0.6C+0.4T; C only when temporal unavailable',
            baseline='S=0.6C+0.1R+0.25P+0.05Y; unfiltered original baseline',
            temporal='exp(-0.5*(SO3_angle/rotation_sigma)^2); rotation only; position excluded',
            config=asdict(self.config),temporal_age_limit='none; legacy max_temporal_gap_s ignored',collision_geometry='Official franka_umi_vis_mesh.obj, static OPEN; mesh vertices vs solid ground half-space; strict minimum distance > plane RMSE; no fixed additional margin',
            no_ground_policy='No Valid Grasp; never silently pass unchecked candidates',
            history='Continuous background ICP/keyframes, exact Frame-ID transforms; no interpolation',
            stored_predictions='All identical raw model candidates, both selections, original camera-frame poses unchanged')
        (self.path/'session.json').write_text(json.dumps(meta,indent=2))

    def offer_frame(self,snapshot):
        super().offer_frame(snapshot)
        if not self.closed:self.icp.offer(snapshot)

    def record(self,completed):
        start=time.perf_counter();snapshot=completed['snapshot'];f=snapshot['frame'];raw=completed['result']
        if raw['frame_id']!=f.frame_id:raise ValueError('Grasp and RGB-D frame mismatch')
        points,colors=cloud_from_snapshot(snapshot)
        if not len(points):self.event('no_valid_object_cloud',frame_id=f.frame_id);return None
        registration=self.icp.lookup(f.frame_id,self.config.icp_wait_s)
        T=registration.get('T_world_camera') if registration else None
        reliable=bool(registration and registration['valid'])
        plane=registration.get('plane') if registration else None
        pm=registration['plane_metrics'] if registration else dict(status='not_processed')
        plane_fallback_ms=0.
        if plane is None:
            # Exact requested capture, never the latest camera frame or plane.
            then=time.perf_counter()
            bg=background_cloud(snapshot,self.config)
            obj,_,_=deproject(f.depth_m,f.K,snapshot['mask']>0,getattr(f,'unit_rays',None))
            plane,pm=estimate_support_plane(bg,obj,self.config,self.icp.o3d)
            plane_fallback_ms=(time.perf_counter()-then)*1000
        generation=(snapshot['response']['session'],snapshot['response']['generation'])
        clock,stamp=capture_time(f)
        same=same_capture_sequence(clock,generation,stamp,self.previous_clock,self.previous_generation,self.previous_time)
        previous=self.previous_new if same else None
        choice=select_candidates(raw['grasps'],raw['confidences'],points,self.vertices,plane,self.config,
            previous_world=previous,T_world_camera=T,icp_reliable=reliable,
            ground_uncertainty_m=pm.get('rmse_m',0.))
        new=choice['selected'];baseline=choice['baseline_selected'];g=np.asarray(raw['grasps'])
        world_baseline=T@g[baseline] if reliable and baseline is not None else None
        world_new=T@g[new] if reliable and new is not None else None
        baseline_delta=pose_change(self.previous_baseline if same else None,world_baseline)
        new_delta=pose_change(previous,world_new)
        attempt=len(self.comparisons);candidate_file=f'all_candidates/{attempt:06d}.npz'
        # Preserve every original SE(3) and C: baseline and new algorithm share input.
        np.savez_compressed(self.path/candidate_file,grasps=g,confidences=raw['confidences'],
            original_indices=np.arange(len(g)),frame_id=f.frame_id,point_cloud=points,colors=colors,
            baseline_scores=choice['baseline']['scores'],baseline_components=choice['baseline']['components'],temporal_scores=choice['temporal'],scores=choice['scores'],
            collision_valid=choice['valid'],clearance_m=choice['clearance_m'],
            baseline_index=-1 if baseline is None else baseline,new_index=-1 if new is None else new,
            T_world_camera=np.full((4,4),np.nan) if T is None else T,
            ground_plane=np.full(4,np.nan) if plane is None else plane,plane_rmse_m=pm.get('rmse_m',np.nan),
            required_clearance_m=pm.get('rmse_m',np.nan))
        self.archive.offer(snapshot);frame_file=self.archive.save(snapshot,priority=True)
        now=time.perf_counter()
        row=dict(attempt=attempt,frame_id=f.frame_id,frame_file=frame_file,candidate_file=candidate_file,
            **timestamp_fields(f),status=choice['status'],ground_status=pm['status'],
            plane_rmse_m=pm.get('rmse_m'),required_clearance_m=pm.get('rmse_m'),
            candidate_count=len(g),candidate_count_generated=raw.get('candidate_count_generated'),
            candidate_count_after_threshold=raw.get('candidate_count_after_threshold'),after_ground_count=int(choice['valid'].sum()),
            baseline_index=baseline,new_index=new,baseline_confidence=float(raw['confidences'][baseline]) if baseline is not None else None,
            baseline_score=float(choice['baseline']['scores'][baseline]) if baseline is not None else None,
            baseline_ground_valid=bool(choice['valid'][baseline]) if baseline is not None else None,
            confidence=float(raw['confidences'][new]) if new is not None else None,
            temporal_score=float(choice['temporal'][new]) if new is not None and choice['temporal_used'] else None,
            final_score=float(choice['scores'][new]) if new is not None else None,temporal_used=choice['temporal_used'],
            temporal_reason='used' if choice['temporal_used'] else ('icp_unreliable_or_missing' if not reliable else 'no_previous_valid_top1_or_selection_changed'),
            icp_status=registration['status'] if registration else 'frame_not_ready_or_dropped',
            icp_fitness=registration.get('fitness') if registration else None,
            icp_rmse_m=registration.get('rmse_m') if registration else None,
            icp_ms=registration.get('icp_ms') if registration else None,
            icp_processing_ms=registration.get('processing_ms') if registration else None,
            icp_fps=registration.get('icp_throughput_fps') if registration else None,
            icp_queue_drops=self.icp.drops,plane_fallback_ms=plane_fallback_ms,
            baseline_delta_mm=baseline_delta[0],baseline_delta_deg=baseline_delta[1],
            new_delta_mm=new_delta[0],new_delta_deg=new_delta[1],
            grasp_infer_ms=raw['server_timing']['infer_ms'],tracking_fps=snapshot['actual_fps'],
            postprocess_ms=(now-start)*1000,end_to_end_ms=(now-f.ready_monotonic)*1000,
            request_to_selection_ms=(now-completed['requested_at'])*1000)
        for label,j in (('baseline',baseline),('new',new)):
            position=np.full(3,np.nan);q=np.full(4,np.nan)
            if j is not None:q,position=matrix_to_wxyz_position(g[j])
            row.update({f'{label}_camera_{axis}_m':float(position[k]) if j is not None else None for k,axis in enumerate(('x','y','z'))})
            row.update({f'{label}_{axis}':float(q[k]) if j is not None else None for k,axis in enumerate(('qw','qx','qy','qz'))})
        row['baseline_temporal_score']=float(choice['temporal'][baseline]) if baseline is not None and choice['temporal_used'] else None
        # Only consecutive trusted captures can supply temporal comparisons.
        self.previous_new=world_new;self.previous_baseline=world_baseline
        self.previous_time=stamp;self.previous_clock=clock;self.previous_generation=generation
        if self.duration_started is None:
            self.duration_started=time.perf_counter()
            self.event('recording_started',duration_origin='first_completed_selection')
        row['completed_elapsed_s']=time.perf_counter()-self.epoch
        span=now-self.first_request if self.first_request is not None else 0.
        row['grasp_pose_fps']=(len(self.rows)+int(new is not None))/span if span>0 else 0.
        row['request_to_result_ms']=(start-completed['requested_at'])*1000
        self.comparisons.append(row)
        with (self.path/'comparison.csv').open('a',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(row))
            if attempt==0:writer.writeheader()
            writer.writerow(row)
        if new is None:
            self.event('no_valid_grasp',frame_id=f.frame_id,reason=choice['status'])
            print(f'No Valid Grasp: {choice["status"]}; plane={pm["status"]}',flush=True)
            return None
        order=choice['order'];result=dict(raw)
        for key in ('grasps','confidences','branch_tags'):
            if key in raw:result[key]=np.asarray(raw[key])[order]
        result.update(selection_scores=choice['scores'][order],original_candidate_indices=order,
                      orientation_scores=choice['baseline']['components'][order,1:])
        return self._persist(completed,result,points,colors,(time.perf_counter()-start)*1000,row)

    def _persist(self,completed,result,points,colors,rank_ms,comparison):
        received=time.perf_counter()
        snapshot=completed['snapshot'];raw=completed['result'];f=snapshot['frame']
        T=np.asarray(result['grasps'][0]).copy()
        C=float(result['confidences'][0]);S=float(result['selection_scores'][0])
        R,P,Y=map(float,result['orientation_scores'][0])
        wxyz,pos=matrix_to_wxyz_position(T)
        wxyz=np.asarray(wxyz,dtype=np.float64)
        wxyz=wxyz/np.linalg.norm(wxyz)
        index=len(self.rows)
        file=f'samples/{index:06d}.npz'
        write_start=time.perf_counter()
        # Primary top-1 remains compatible with the existing session viewer.
        np.savez_compressed(self.path/file,pose=T,confidence=C,score=S,orientation_scores=[R,P,Y],
            point_cloud=points,colors=colors,K=f.K,frame_id=f.frame_id,
            original_candidate_index=result['original_candidate_indices'][0],temporal_score=np.nan if comparison['temporal_score'] is None else comparison['temporal_score'],
            selection_algorithm='confidence_temporal_ground')
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
            request_to_result_ms=comparison['request_to_result_ms'],
            pointcloud_and_ranking_ms=rank_ms,record_write_ms=write_ms,
            request_to_record_ms=(done-completed['requested_at'])*1000,
            capture_to_record_ms=(done-f.ready_monotonic)*1000,
            tracking_fps=snapshot['actual_fps'])
        row.update(camera_relative_delta_mm=row['position_delta_mm'],camera_relative_delta_deg=row['rotation_delta_deg'],
            position_delta_mm=comparison['new_delta_mm'],rotation_delta_deg=comparison['new_delta_deg'],
            pose_delta_frame='continuous_icp_reference',temporal_score=comparison['temporal_score'],temporal_used=comparison['temporal_used'],
            selection_algorithm='confidence_temporal_ground',ground_filtered_count=comparison['after_ground_count'],
            icp_processing_ms=comparison['icp_processing_ms'],compensated_delta_mm=comparison['new_delta_mm'],
            compensated_delta_deg=comparison['new_delta_deg'])
        # Full SE(3), not just quaternion, is also available in the CSV.
        row.update({f't{r}{c}':float(T[r,c]) for r in range(4) for c in range(4)})
        with (self.path/'poses.csv').open('a',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row))
            if not self.rows:writer.writeheader()
            writer.writerow(row)
        self.rows.append(row)
        self.previous=dict(segment=segment,pose=T.copy(),quaternion=wxyz.copy(),frame_ready=f.ready_monotonic)
        return row

    def close(self):
        if not self.icp_closed:
            self.icp_closed=True;self.icp.close()
        report=super().close()
        def stats(key):
            values=[float(r[key]) for r in self.comparisons if r.get(key) is not None and np.isfinite(r[key])]
            return dict(n=len(values),mean=float(np.mean(values)),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95))) if values else None
        comparison=dict(attempts=len(self.comparisons),valid_new=sum(r['new_index'] is not None for r in self.comparisons),
            temporal_used=sum(r['temporal_used'] for r in self.comparisons),no_valid=sum(r['new_index'] is None for r in self.comparisons),
            baseline_ground_rejected=sum(r['baseline_ground_valid'] is False for r in self.comparisons),
            metrics={key:stats(key) for key in ('baseline_delta_mm','baseline_delta_deg','new_delta_mm','new_delta_deg',
                 'baseline_confidence','confidence','temporal_score','icp_ms','icp_processing_ms','icp_fps','grasp_infer_ms','end_to_end_ms','postprocess_ms')},
            actual_pose_throughput=report['completed_pose_fps'],note='Stability in ICP reference coordinates; blank when exact capture transform unavailable. No live hardware/model benchmark implied.')
        (self.path/'comparison_summary.json').write_text(json.dumps(comparison,indent=2))
        return report
