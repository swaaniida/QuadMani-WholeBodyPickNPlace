"""Tracking SAM extension of the existing RealSense/GraspGenX frontend."""
from grasp_ranking import rerank_result
from auto_grasp import ContinuousGrasp, cloud_from_snapshot
from temporal_session import TemporalSession as Top1Session
import argparse
import faulthandler
from dataclasses import replace
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import webbrowser
import cv2
import numpy as np
from pointcloud import FilterConfig
from realsense_capture import RealSenseCapture
from recording import save_result
from tracking_client import TrackingWorker,csv_row,summary
from visualization import GraspViewer,overlay


def recording_elapsed(session,now=None):
    if session.duration_started is None:return None
    return max(0.,(time.perf_counter() if now is None else now)-session.duration_started)


def reset_recording_timer(session):
    # Keep all recorded poses and timestamps; restart only the duration budget.
    session.duration_start_index=len(session.rows)
    session.event('duration_timer_reset',next_pose_index=session.duration_start_index)


def mask_overlay(snapshot):
    image=snapshot['frame'].bgr.copy()
    mask=snapshot['mask']>0
    retained=snapshot['filtered']>0
    image[mask]=(image[mask]*.6+np.array([0,220,0])*.4).astype(np.uint8)
    removed=mask & ~retained
    image[removed]=(image[removed]*.5+np.array([0,0,255])*.5).astype(np.uint8)
    return image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--debug-startup',action='store_true',help='Log initialization stages and dump Python thread stacks after 10 seconds until first live GUI frame')
    p.add_argument('--temporal-config',type=Path,help='Optional temporal/ICP/ground JSON settings')
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=15557)
    p.add_argument('--serial')
    p.add_argument('--auto-grasp',action='store_true',help='Continuously record maximum-S top1; defer 3D review until Q')
    p.add_argument('--duration',type=float,default=0,help='Seconds after first completed selection attempt (including No Valid Grasp); 0 = until Q')
    p.add_argument('--max-poses',type=int,default=0,help='Stop after this many recorded top1 poses; 0 = unlimited')
    p.add_argument('--no-review',action='store_true',help='Print summary only; open view_session.py later')
    p.add_argument('--fps',type=float,default=15,help='Maximum RGB-D uploads per second')
    p.add_argument('--planner',choices=['graspmoe','diffusion'],default='graspmoe')
    p.add_argument('--num-grasps',type=int,default=200)
    p.add_argument('--topk',type=int,default=-1,help='Temporal selection requires -1: retain all model candidates before ground filtering')
    p.add_argument('--threshold',type=float,default=-1)
    p.add_argument('--text',help='Optional initial GroundingDINO text prompt')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs/tracking_temporal')
    p.add_argument('--viewer-port',type=int,default=8080)
    p.add_argument('--near',type=float,default=.1)
    p.add_argument('--far',type=float,default=2.)
    p.add_argument('--band',type=float,default=0,help='Meters; default 0 preserves the SAM object depth extent')
    p.add_argument('--voxel',type=float,default=.003)
    args=p.parse_args()
    if args.topk!=-1 or args.threshold!=-1:
        p.error('Temporal selection requires --topk -1 --threshold -1; filter ground before scoring all candidates')
    args.auto_grasp=True
    def startup_stage(message):
        if args.debug_startup:print('[STARTUP] '+message,flush=True)
    if args.debug_startup:
        faulthandler.enable()
        faulthandler.dump_traceback_later(10,repeat=True)
    if not 0<args.fps<=30:p.error('--fps must be (0,30]')
    if args.duration<0 or args.max_poses<0:p.error('--duration/--max-poses must be >=0')
    print('Loading CPU ICP support; inference servers remain unchanged',flush=True)
    startup_stage('Importing Open3D')
    try:import open3d
    except ImportError:p.error('Install: python -m pip install -r requirements-temporal.txt')
    startup_stage('Open3D import completed')
    cfg=FilterConfig(near_m=args.near,far_m=args.far,foreground_band_m=args.band,voxel_m=args.voxel)
    startup_stage('Starting RealSense capture thread')
    capture=RealSenseCapture(args.serial).start()
    startup_stage('Starting tracking network thread')
    worker=TrackingWorker(capture,cfg,args.host,args.port,args.fps,args.output,args.planner,
                          args.topk,args.threshold,args.num_grasps).start()
    viewer=None
    auto=None
    auto_armed=args.auto_grasp
    finishing=False
    exit_reason=None

    def start_auto():
        import platform
        session=Top1Session(args.output,dict(planner=args.planner,num_grasps=args.num_grasps,
            candidate_topk=args.topk,tracking_upload_cap_fps=args.fps,
            windows_hardware_tested=platform.system()=='Windows',source='realsense_usb',
            duration_s=args.duration,max_poses=args.max_poses,duration_origin='first_completed_selection',temporal_config=str(args.temporal_config) if args.temporal_config else None))
        worker.frame_observer=session.offer_frame
        print('AUTO top1 session:',session.path,flush=True)
        return ContinuousGrasp(session)

    # Open the recording only after target selection, never on program startup.
    displayed=None
    last_grasp=None
    last_view=None
    last_result_ms=None
    last_sam_ms=None
    notice='Click target | A auto/pause | G single | V 3D | R reset | S save | Q finish/review'
    win='Tracking SAM: matched RGB + aligned depth (click RGB target)'
    controls='Tracking depth filter: green retained / red removed'
    startup_stage('Creating OpenCV windows')
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    cv2.namedWindow('RealSense LIVE',cv2.WINDOW_NORMAL)
    cv2.namedWindow(controls,cv2.WINDOW_NORMAL)
    for name,initial,maximum in [('near mm',int(args.near*1000),5000),('far mm',int(args.far*1000),5000),
                                 ('band mm (0 off)',int(args.band*1000),1000),('voxel mm',int(args.voxel*1000),30)]:
        cv2.createTrackbar(name,controls,initial,maximum,lambda _:None)

    def click(event,x,y,flags,param):
        nonlocal notice,last_grasp,last_view,auto
        if event!=cv2.EVENT_LBUTTONDOWN or displayed is None:return
        frame=displayed['frame']
        if x>=frame.bgr.shape[1] or y>=frame.bgr.shape[0]:return
        worker.frame_observer=None
        if auto is not None:
            auto.cancel('new target selected')
            auto.session.event('exit_reason',reason='new_target_selected')
            auto.session.close()
            auto=None
        worker.select(frame,point=(x,y))
        if auto_armed:auto=start_auto()
        last_grasp=last_view=None
        if viewer is not None:viewer.server.scene.reset()
        notice=f'SAM initializing from capture {frame.frame_id} at ({x},{y})'
        try:cv2.destroyWindow('GRASP on matched captured RGB')
        except cv2.error:pass
    cv2.setMouseCallback(win,click)
    startup_stage('Entering camera GUI event loop')
    try:
        while True:
            cfg=replace(cfg,near_m=cv2.getTrackbarPos('near mm',controls)/1000,
                        far_m=cv2.getTrackbarPos('far mm',controls)/1000,
                        foreground_band_m=cv2.getTrackbarPos('band mm (0 off)',controls)/1000,
                        voxel_m=cv2.getTrackbarPos('voxel mm',controls)/1000)
            if not 0<cfg.near_m<cfg.far_m:
                notice='Depth filter: require 0 < Near < Far; previous valid settings kept'
                cfg=worker.config
            with worker.lock:worker.config=cfg
            live=capture.latest()
            if live is not None:
                cv2.imshow('RealSense LIVE',live.bgr)
                if args.text:
                    worker.select(live,text=args.text)
                    if auto_armed and auto is None:auto=start_auto()
                    args.text=None
            latest,status,busy=worker.state()
            if latest is not None:
                displayed=latest
                f=latest['frame'];r=latest['response']
                if r['metrics']['sam_init_ms'] is not None:last_sam_ms=round(r['metrics']['sam_init_ms'],1)
                rgb=mask_overlay(latest)
                depth_color=cv2.applyColorMap(cv2.convertScaleAbs(f.depth_m,alpha=255/max(cfg.far_m,.1)),cv2.COLORMAP_TURBO)
                depth_color[f.depth_m<=0]=0
                age=(time.perf_counter()-f.ready_monotonic)*1000
                image=np.hstack((rgb,depth_color))
                auto_status='' if auto is None else f' | AUTO {"ON" if auto.enabled else "PAUSED"} top1={len(auto.session.rows)} poseFPS={auto.session.fps():.2f}'
                if auto is not None:
                    elapsed=recording_elapsed(auto.session)
                    auto_status+=(' | Waiting first selection; timer not started' if elapsed is None else f' | recording {elapsed:.1f}s')
                if auto is not None and auto.session.comparisons:
                    selection=auto.session.comparisons[-1]
                    auto_status+=f' | {selection["status"]} | ICP={selection["icp_fps"]} FPS'
                if auto is not None and auto.last_error:auto_status+=' | Grasp waiting: '+auto.last_error
                lines=[f'ID 1 | {status} | tracked={latest["actual_fps"]:.1f} FPS | age={age:.0f} ms',
                       f'depth retained={r["valid_depth_ratio"]:.1%} | SAM={last_sam_ms} ms | VOS={None if r["metrics"]["vos_ms"] is None else round(r["metrics"]["vos_ms"],1)} ms | grasp={last_result_ms} ms',
                       f'CAPTURE {f.frame_id}',
                       notice + (' | GRASP RUNNING; tracking continues' if busy else ''),auto_status]
                for i,line in enumerate(lines):
                    cv2.putText(image,line[:175],(8,20+i*20),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
            else:
                displayed=None
                image=np.zeros((480,1280,3),np.uint8)
                cv2.putText(image,(status+' | '+capture.status)[:160],(10,40),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
            cv2.imshow(win,image)
            if auto is None:
                grasp_error=worker.take_grasp_error()
                if grasp_error is not None:notice='Grasp skipped; tracking continues: '+grasp_error['message']
            completed=worker.take_result()
            if completed is not None:
                try:
                    if auto is not None:
                        row=auto.accept(completed)
                        if row is None and auto.session.comparisons:notice='No Valid Grasp: '+auto.session.comparisons[-1]['status']
                        last_result_ms=round(completed['result']['server_timing']['infer_ms'],1)
                        if row is not None:
                            if row['index']==getattr(auto.session,'duration_start_index',0):
                                print(f'First valid pose saved; duration={args.duration}s from first selection',flush=True)
                                auto.session.event('first_valid_pose_saved',duration_s=args.duration)
                            notice=f'Top1 #{row["index"]+1} S={row["S"]:.4f} | Q finishes and reviews'
                            print(f'AUTO top1={len(auto.session.rows)} S={row["S"]:.4f} infer={last_result_ms} ms',flush=True)
                    else:
                        snapshot=completed['snapshot'];result=completed['result'];frame=snapshot['frame']
                        viz_start=time.perf_counter()
                        points,colors=cloud_from_snapshot(snapshot)
                        result=rerank_result(result,points)
                        cloud_ms=(time.perf_counter()-viz_start)*1000
                        overlay_image=overlay(frame.bgr,frame.K,result['grasps'],result['confidences'],
                                              frame.frame_id,frame.color_intrinsics,point_cloud=points,original_indices=result['original_candidate_indices'])
                        cv2.imshow('GRASP on matched captured RGB',overlay_image)
                        if viewer is None:viewer=GraspViewer(args.viewer_port)
                        viewer.show(points,colors,result['grasps'],result['confidences'],frame.frame_id,original_indices=result['original_candidate_indices'])
                        viz_ms=(time.perf_counter()-viz_start)*1000
                        run_id=datetime.now(timezone.utc).strftime('tracking_%Y%m%dT%H%M%S_%fZ')
                        result['network_rtt_ms']=result['grasp_rpc_rtt_ms']
                        result['tracking_metadata']=dict(frame_id=frame.frame_id,
                            session=snapshot['response']['session'],generation=snapshot['response']['generation'],
                            metrics=snapshot['response']['metrics'],tracking_rtt_ms=snapshot['tracking_rtt_ms'])
                        save_ms=save_result(args.output,run_id,frame,points,colors,snapshot['filtered'],result,
                                           overlay_image,snapshot['config'],(0,0,frame.bgr.shape[1],frame.bgr.shape[0]),
                                           'tracking_scene_depth',args.planner)
                        cv2.imwrite(str(args.output/'depth'/f'{run_id}_tracking_mask.png'),snapshot['mask']*255)
                        row=dict(run_id=run_id,frame_id=frame.frame_id,
                                 sam_init_ms=snapshot['response']['metrics']['sam_init_ms'],
                                 vos_ms=snapshot['response']['metrics']['vos_ms'],
                                 mask_depth_ms=snapshot['response']['mask_depth_ms'],
                                 pointcloud_and_downsample_ms=cloud_ms,
                                 grasp_infer_ms=result['server_timing']['infer_ms'],
                                 grasp_rpc_rtt_ms=result['grasp_rpc_rtt_ms'],result_poll_rtt_ms=completed['poll_rtt_ms'],
                                 visualization_ms=viz_ms,save_ms=save_ms,
                                 total_grasp_ms=(time.perf_counter()-completed['requested_at'])*1000,
                                 frame_age_at_result_ms=(time.perf_counter()-frame.ready_monotonic)*1000,
                                 grasps=len(result['grasps']))
                        csv_row(args.output/'grasp_requests.csv',row)
                        print(json.dumps(row,indent=2),flush=True)
                        print(json.dumps(summary(args.output/'grasp_requests.csv'),indent=2),flush=True)
                        last_result_ms=round(result['server_timing']['infer_ms'],1)
                        last_grasp=(snapshot,result)
                        last_view=('grasp',frame.frame_id)
                        notice=f'{len(result["grasps"])} real grasps saved: {run_id}'
                except Exception as exc:
                    notice=f'Visualization/save error: {exc}'
                    print(notice,flush=True)
            key=cv2.waitKey(10)&0xff
            if args.debug_startup and live is not None:
                faulthandler.cancel_dump_traceback_later()
                startup_stage('First live frame displayed and GUI events processed')
                args.debug_startup=False
            if key in (ord('q'),27):
                if auto is None:exit_reason='user_q';break
                if finishing:
                    exit_reason='user_forced_q'
                    auto.cancel('forced exit while draining')
                    break
                finishing=True
                exit_reason='user_q'
                auto.enabled=False
                notice='Finishing: waiting for the outstanding grasp; Q again forces exit'
            if key==ord('a') and not finishing:
                if auto is None:
                    if busy:notice='Wait for the manual grasp before starting auto'
                    else:
                        auto_armed=True
                        if latest is not None and latest['response']['tracking_valid']:auto=start_auto()
                        if viewer is not None:viewer.close();viewer=None
                        try:cv2.destroyWindow('GRASP on matched captured RGB')
                        except cv2.error:pass
                else:auto.toggle()
                notice='Auto top1 recording; A pause/resume, Q finish and review'
            elif key==ord('g'):
                if auto is not None:
                    notice='Auto session active; use A to pause/resume'
                    continue

                try:worker.request_grasp();notice='Requested grasp of latest fresh matched frame'
                except ValueError as exc:notice=str(exc)
            elif key==ord('r'):
                worker.frame_observer=None
                worker.reset()
                if auto is not None:
                    auto.cancel('tracking reset')
                    auto.session.event('exit_reason',reason='tracking_reset_discarded_from_current_review')
                    auto.session.close()
                    auto=None
                    auto_armed=True
                displayed=last_grasp=last_view=None
                if viewer is not None:viewer.server.scene.reset()
                try:cv2.destroyWindow('GRASP on matched captured RGB')
                except cv2.error:pass
                notice='Reset; previous take excluded; click to start a NEW recording'
            elif key==ord('v'):
                if auto is not None:
                    notice='3D review is deferred; press Q to finish recording'
                    continue
                try:
                    if viewer is None:viewer=GraspViewer(args.viewer_port)
                    if last_grasp is not None:
                        snapshot,result=last_grasp
                        points,colors=cloud_from_snapshot(snapshot)
                        viewer.show(points,colors,result['grasps'],result['confidences'],snapshot['frame'].frame_id,original_indices=result['original_candidate_indices'])
                    elif latest is not None and latest['response']['grasp_allowed']:
                        points,colors=cloud_from_snapshot(latest)
                        viewer.show(points,colors,frame_id=latest['frame'].frame_id)
                    else:raise ValueError('No valid tracked cloud yet')
                    webbrowser.open(f'http://127.0.0.1:{args.viewer_port}')
                except Exception as exc:notice=str(exc)
            elif key==ord('s'):
                try:
                    if latest is None:raise ValueError('No matched frame/mask to save')
                    out=args.output/'captures';out.mkdir(parents=True,exist_ok=True)
                    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                    f=latest['frame']
                    cv2.imwrite(str(out/f'{stamp}_rgb.png'),f.bgr)
                    np.savez_compressed(out/f'{stamp}.npz',depth_m=f.depth_m,depth_raw=f.depth_raw,
                                        K=f.K,depth_scale=f.depth_scale,instance_mask=latest['mask'],
                                        filtered_mask=latest['filtered'],frame_id=f.frame_id)
                    (out/f'{stamp}.json').write_text(json.dumps(dict(frame_id=f.frame_id,
                        camera_timestamp_ms=f.camera_timestamp_ms,host_timestamp_s=f.host_timestamp_s,
                        intrinsics=f.intrinsics,color_intrinsics=f.color_intrinsics,
                        filters=vars(latest['config']),status=latest['response']['status']),indent=2))
                    notice=f'Matched RGB-D + masks saved: {stamp}'
                except Exception as exc:notice=str(exc)
            if auto is not None:
                limit=(args.max_poses>0 and len(auto.session.rows)>=args.max_poses)
                elapsed=recording_elapsed(auto.session)
                expired=(args.duration>0 and elapsed is not None and elapsed>=args.duration)
                if (limit or expired) and not finishing:
                    exit_reason='max_poses_reached' if limit else 'recording_duration_reached'
                    print(f'STOP: {exit_reason}; saved={len(auto.session.rows)}',flush=True)
                    auto.session.event('stop_requested',reason=exit_reason)
                    finishing=True
                    auto.enabled=False
                auto.tick(worker)
                if finishing and not auto.inflight:break
    finally:
        worker.close();capture.close();cv2.destroyAllWindows()
        if viewer is not None:viewer.close()
        print(json.dumps(summary(args.output/'tracking_frames.csv'),indent=2))
        if auto is not None:
            if auto.inflight:auto.cancel('client shutdown')
            worker.frame_observer=None
            auto.session.event('exit_reason',reason=exit_reason or 'exception_or_interrupt')
            report=auto.session.close()
            print('SESSION:',auto.session.path,flush=True)
            print(json.dumps(report,indent=2),flush=True)
    if auto is not None and auto.session.comparisons and not args.no_review:
        from view_temporal import review
        review(auto.session.path,args.viewer_port)



if __name__=='__main__':main()
