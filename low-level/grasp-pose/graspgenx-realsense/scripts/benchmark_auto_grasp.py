"""Actual SAM/AOT + GraspGenX continuous recording on static recorded RGB-D.

Not a Windows/RealSense hardware FPS claim. Uses the exact Windows worker,
continuous scheduler, RGB-D/mask archive and top1 recorder against real servers.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import urllib.request
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_client'))
from auto_grasp import Top1Session,ContinuousGrasp
from grasp_client import GraspClient
from pointcloud import FilterConfig
from tracking_client import TrackingWorker
from tracking_worker_smoke import RecordedCapture,wait_for
from view_session import SessionViewer


def run(args):
    health=GraspClient(port=args.port,timeout_ms=1000)
    try:wait_for(lambda:healthy(health),seconds=120)
    finally:health.close()
    capture=RecordedCapture()
    session=Top1Session(ROOT/'outputs/auto_grasp',dict(source='official real_world/00 repeated static RGB-D replay',
        windows_hardware_tested=False,num_grasps=200,tracking_upload_cap_fps=args.fps,duration_s=args.duration))
    worker=TrackingWorker(capture,FilterConfig(foreground_band_m=0),port=args.port,fps=args.fps,
                          output=session.path/'network')
    worker.frame_observer=session.offer_frame
    auto=ContinuousGrasp(session)
    worker.start();viewer=None
    try:
        wait_for(lambda:worker.state()[0],seconds=90)
        worker.select(capture.latest(),point=(620,450))
        wait_for(lambda:worker.state()[0] and worker.state()[0]['response']['grasp_allowed'] and time.perf_counter()-worker.state()[0]['frame'].ready_monotonic<.5,seconds=90)
        timeout=time.perf_counter()+args.duration+120
        last_print=0.
        while time.perf_counter()<timeout:
            completed=worker.take_result()
            if completed is not None:
                row=auto.accept(completed)
                assert row is not None
                assert row['candidate_count_generated'] is not None,'Server count telemetry missing'
                print('record',row['index']+1,'frame',row['frame_id'],'inference ms',row['grasp_infer_ms'],flush=True)
            if auto.started is not None and time.perf_counter()-auto.started>=args.duration:auto.enabled=False
            auto.tick(worker)
            if auto.started is not None and not auto.enabled and not auto.inflight:break
            if time.perf_counter()-last_print>=5:
                print('progress:',len(session.rows),'poses;',worker.state()[1],flush=True);last_print=time.perf_counter()
            time.sleep(.01)
        else:raise TimeoutError('Automatic recording did not complete')
        assert len(session.rows)>=3
    finally:
        worker.close();worker.frame_observer=None
        session.close()
    with (session.path/'frames_with_poses.csv').open(newline='') as f:frames=list(csv.DictReader(f))
    assert sum(r['pose_available']=='True' for r in frames)==len(session.rows)
    assert len({r['frame_id'] for r in session.rows})==len(session.rows)
    for row in session.rows:
        with np.load(session.path/row['file']) as d:
            assert d['pose'].shape==(4,4) and 'grasps' not in d
        with np.load(session.path/row['frame_file']) as d:
            assert str(d['frame_id'])==row['frame_id']
            assert json.loads(str(d['metadata_json']))['frame_id']==row['frame_id']
            np.testing.assert_array_equal(d['depth_m'],capture.depth)
    try:
        viewer=SessionViewer(session.path,port=18090)
        viewer.show_index(len(session.rows)-1)
        assert urllib.request.urlopen('http://127.0.0.1:18090',timeout=10).status==200
        assert 'C#1' not in viewer.viewer.details.content
        assert viewer.rgb.image.shape[:2]==capture.depth.shape
    finally:
        if viewer is not None:viewer.close()
    report=session.stats()
    report.update(validation='real models + static recorded RGB-D + localhost transport; not Windows hardware',
                  session_path=str(session.path),offline_viewer_http=200,frame_pose_join='passed',depth_lossless='passed')
    (ROOT/'outputs/auto_grasp/latest_benchmark.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


def healthy(client):
    try:return client.health()['status']=='ok'
    except Exception:return False

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--duration',type=float,default=30)
    p.add_argument('--fps',type=float,default=10)
    p.add_argument('--port',type=int,default=5557)
    args=p.parse_args();run(args)
