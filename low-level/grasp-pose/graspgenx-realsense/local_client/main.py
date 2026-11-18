"""Windows RGB-D frontend: S select, G infer, V view, R reset, Q quit."""
from grasp_ranking import rerank_result
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import webbrowser
import cv2
import numpy as np
from grasp_client import GraspClient, load_sweep_params
from pointcloud import FilterConfig, prepare
from realsense_capture import RealSenseCapture
from recording import append_latency, latency_summary, save_result
from visualization import GraspViewer, overlay


def infer_selected(frame, roi, config, args):
    start = time.perf_counter()
    frame_age = (start-frame.ready_monotonic)*1000
    points, colors, mask, timing = prepare(frame, roi, config)
    # Socket creation, send, receive and close all occur in this single thread.
    client = GraspClient(args.host, args.port, args.timeout_ms)
    try:
        result = client.infer(points, num_grasps=args.num_grasps,
                              topk_num_grasps=args.topk, grasp_threshold=args.threshold,
                              api=args.api, planner=args.planner,
                              sweep_params=load_sweep_params())
    finally:
        client.close()
    return dict(start=start, frame_age=frame_age, frame=frame, roi=roi, config=config,
                points=points, colors=colors, mask=mask, timing=timing, result=result)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5556)
    p.add_argument('--serial')
    p.add_argument('--api', choices=['params', 'name'], default='params')
    p.add_argument('--planner', choices=['graspmoe', 'diffusion'], default='graspmoe')
    p.add_argument('--num-grasps', type=int, default=200)
    p.add_argument('--topk', type=int, default=100)
    p.add_argument('--threshold', type=float, default=-1)
    p.add_argument('--timeout-ms', type=int, default=60000)
    p.add_argument('--viewer-port', type=int, default=8080)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'outputs')
    p.add_argument('--near', type=float, default=0.1)
    p.add_argument('--far', type=float, default=2.0)
    p.add_argument('--band', type=float, default=0.08, help='Foreground depth band in meters; 0 disables')
    p.add_argument('--voxel', type=float, default=0.003)
    p.add_argument('--max-points', type=int, default=12000)
    args = p.parse_args()
    if args.api == 'name' and args.planner != 'diffusion':
        p.error('Official name API only supports diffusion; add --planner diffusion or use --api params')
    return args


def main():
    args = parse_args()
    capture = RealSenseCapture(args.serial).start()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='grasp-infer')
    viewer = None
    future = None
    selected = roi = last = None
    selection_revision = 0
    submitted_revision = 0
    notice = 'S select | G infer | V browser | R reset | Q quit'
    latency_path = args.output / 'benchmarks/latency.csv'
    cfg = FilterConfig(args.near, args.far, args.band, voxel_m=args.voxel, max_points=args.max_points)
    window = 'RealSense LIVE (results use a separate frozen capture)'
    controls = 'Depth filtering (meters shown as mm)'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(controls, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(controls, 650, 260)
    for name, initial, maximum in [('near mm', int(args.near*1000), 5000),
                                  ('far mm', int(args.far*1000), 5000),
                                  ('band mm (0 off)', int(args.band*1000), 1000),
                                  ('voxel mm (0 off)', int(args.voxel*1000), 30),
                                  ('foreground percentile', 10, 100)]:
        cv2.createTrackbar(name, controls, initial, maximum, lambda _: None)
    preview_key = None
    last_preview = None
    try:
        while True:
            live = capture.latest()
            cfg = replace(cfg,
                          near_m=cv2.getTrackbarPos('near mm', controls)/1000,
                          far_m=cv2.getTrackbarPos('far mm', controls)/1000,
                          foreground_band_m=cv2.getTrackbarPos('band mm (0 off)', controls)/1000,
                          voxel_m=cv2.getTrackbarPos('voxel mm (0 off)', controls)/1000,
                          foreground_percentile=cv2.getTrackbarPos('foreground percentile', controls))
            if live is not None:
                rgb = live.bgr.copy()
                depth_img = cv2.applyColorMap(cv2.convertScaleAbs(live.depth_m, alpha=255/max(cfg.far_m, .1)), cv2.COLORMAP_TURBO)
                depth_img[(live.depth_m <= 0) | ~np.isfinite(live.depth_m)] = 0
                valid = np.mean(np.isfinite(live.depth_m) & (live.depth_m > 0))
                cv2.putText(rgb, f'LIVE {live.frame_id} valid-depth={valid:.1%}', (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                image = np.hstack((rgb, depth_img))
            else:
                image = np.zeros((480, 1280, 3), np.uint8)
            cv2.putText(image, capture.status[:150], (8, image.shape[0]-45),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
            cv2.putText(image, notice[:160], (8, image.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            cv2.imshow(window, image)
            if selected is not None and (selected.frame_id, cfg) != preview_key:
                preview_key = (selected.frame_id, cfg)
                try:
                    pc, colors, mask, _ = prepare(selected, roi, cfg)
                    preview = selected.bgr.copy()
                    preview[~mask] = preview[~mask]//4
                    cv2.putText(preview, f'SELECTED {selected.frame_id}: {len(pc)} points', (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
                    cv2.imshow('Selected capture and depth mask', preview)
                    last_preview = (pc, colors, selected.frame_id)
                except ValueError as exc:
                    last_preview = None
                    notice = str(exc)
            if future is not None and future.done():
                try:
                    done = future.result()
                    if submitted_revision != selection_revision:
                        notice = 'Discarded result for reset/replaced selection; press G for current capture'
                    else:
                        frame = done['frame']
                        result = rerank_result(done['result'], done['points'])
                        done['result'] = result
                        viz_start = time.perf_counter()
                        overlay_image = overlay(frame.bgr, frame.K, result['grasps'], result['confidences'],
                                                frame.frame_id, frame.color_intrinsics, point_cloud=done['points'], original_indices=result['original_candidate_indices'])
                        cv2.imshow('GRASP RESULT on its captured RGB (not LIVE)', overlay_image)
                        if viewer is None:
                            viewer = GraspViewer(args.viewer_port)
                        viewer.show(done['points'], done['colors'], result['grasps'], result['confidences'], frame.frame_id, original_indices=result['original_candidate_indices'])
                        viz_ms = (time.perf_counter()-viz_start)*1000
                        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                        save_ms = save_result(args.output, run_id, frame, done['points'], done['colors'],
                                              done['mask'], result, overlay_image, done['config'], done['roi'],
                                              args.api, args.planner)
                        end = time.perf_counter()
                        row = dict(run_id=run_id, frame_id=frame.frame_id, api=args.api, planner=args.planner,
                                   points=len(done['points']), grasps=len(result['grasps']),
                                   acquisition_ms=frame.acquisition_ms, alignment_ms=frame.alignment_ms,
                                   conversion_ms=frame.conversion_ms, **done['timing'],
                                   network_rtt_ms=result['network_rtt_ms'],
                                   server_infer_ms=result['server_timing'].get('infer_ms'),
                                   server_handler_sync_ms=result['server_timing'].get('handler_sync_ms'),
                                   visualization_ms=viz_ms, save_ms=save_ms,
                                   request_total_ms=(end-done['start'])*1000,
                                   frame_age_at_request_ms=done['frame_age'],
                                   capture_to_result_ms=(end-frame.ready_monotonic)*1000)
                        append_latency(latency_path, row)
                        print(json.dumps(row, indent=2), flush=True)
                        print(json.dumps(latency_summary(latency_path), indent=2), flush=True)
                        if len(result['grasps']):
                            print('BEST T_camera_gripper_base:\n', result['grasps'][0],
                                  '\nconfidence=', result['confidences'][0], '\nfinal score=', result['selection_scores'][0], flush=True)
                        last = done
                        notice = f'{len(result["grasps"])} grasps saved: {run_id}; V opens 3D'
                except Exception as exc:
                    notice = f'Inference/save failed: {exc}'
                    print(notice, flush=True)
                finally:
                    future = None
            key = cv2.waitKey(15) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord('s') and live is not None:
                frozen = live
                box = cv2.selectROI('Select target on FROZEN capture; Enter confirm, C cancel', frozen.bgr,
                                    showCrosshair=True, fromCenter=False)
                cv2.destroyWindow('Select target on FROZEN capture; Enter confirm, C cancel')
                if box[2] and box[3]:
                    selected, roi = frozen, tuple(map(int, box))
                    selection_revision += 1
                    last = None
                    try:
                        cv2.destroyWindow('GRASP RESULT on its captured RGB (not LIVE)')
                    except cv2.error:
                        pass
                    if viewer is not None:
                        viewer.server.scene.reset()
                        viewer.info.content = f'New capture {frozen.frame_id}; press V for cloud or G to infer.'
                    notice = f'Selected capture {frozen.frame_id}; tune depth, then G'
            elif key == ord('g'):
                if future is not None:
                    notice = 'One inference already running; request ignored (no queue)'
                elif selected is None:
                    notice = 'Press S and select a captured object first'
                else:
                    submitted_revision = selection_revision
                    future = executor.submit(infer_selected, selected, roi, cfg, args)
                    notice = f'Inferring captured {selected.frame_id} ... LIVE continues'
            elif key == ord('r'):
                selected = roi = last = last_preview = None
                selection_revision += 1
                preview_key = None
                for old_window in ('Selected capture and depth mask', 'GRASP RESULT on its captured RGB (not LIVE)'):
                    try:
                        cv2.destroyWindow(old_window)
                    except cv2.error:
                        pass
                if viewer is not None:
                    viewer.server.scene.reset()
                    viewer.info.content = 'Selection reset; waiting for a new capture.'
                notice = 'Selection reset; S selects a fresh capture'
            elif key == ord('v'):
                if last is not None or last_preview is not None:
                    if viewer is None:
                        viewer = GraspViewer(args.viewer_port)
                    if last is not None:
                        r = last['result']
                        viewer.show(last['points'], last['colors'], r['grasps'], r['confidences'], last['frame'].frame_id, original_indices=r['original_candidate_indices'])
                    else:
                        viewer.show(*last_preview[:2], frame_id=last_preview[2])
                    webbrowser.open(f'http://127.0.0.1:{args.viewer_port}')
                else:
                    notice = 'Select a valid point cloud first with S'
    finally:
        capture.close()
        executor.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()
        if viewer is not None:
            viewer.close()
        print(json.dumps(latency_summary(latency_path), indent=2))
        print('Camera closed. An in-flight network call may finish within --timeout-ms.')


if __name__ == '__main__':
    main()
