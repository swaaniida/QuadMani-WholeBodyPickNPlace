"""Protocol fixtures only: no model inference or fake inference user mode."""
import sys,threading,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_client'))
import tracking_client
from tracking_client import TrackingWorker
from tracking_protocol import png_mask
from pointcloud import FilterConfig
from auto_grasp import ContinuousGrasp


class Capture:
    def latest(self):
        now=time.perf_counter()
        return SimpleNamespace(frame_id=f'regression-{int(now*30)}',ready_monotonic=now,
            bgr=np.zeros((12,12,3),np.uint8),depth_raw=np.full((12,12),600,np.uint16),
            depth_m=np.full((12,12),.6,np.float32),depth_scale=.001,K=np.eye(3,dtype=np.float32),
            intrinsics={},color_intrinsics={},camera_timestamp_ms=0,host_timestamp_s=0,
            acquisition_ms=1,alignment_ms=1,conversion_ms=1)


def wait_for(fn):
    end=time.perf_counter()+4
    while time.perf_counter()<end:
        value=fn()
        if value:return value
        time.sleep(.005)
    raise AssertionError('Fixture worker timed out')


@pytest.mark.parametrize('failure', ['busy_submit','failed_poll','timeout_submit'])
def test_grasp_error_never_resets_tracking(monkeypatch,tmp_path,failure):
    actions=[];initialized=[False];tracked=[0]
    class Client:
        def __init__(self,*args,**kwargs):pass
        def close(self):pass
        def request(self,p):
            actions.append(p['action'])
            if p['action']=='reset':initialized[0]=False;return {},.1
            if p['action']=='close':return {},.1
            if p['action']=='grasp_submit':
                if failure=='busy_submit':raise RuntimeError('ValueError: One grasp job is already running; no request queue')
                if failure=='timeout_submit':raise TimeoutError('fixture submit transport timeout')
                return dict(job_id='fixture-job'),.1
            if p['action']=='grasp_poll':raise RuntimeError('ValueError: fixture failed grasp job')
            assert p['action']=='frame'
            if 'point' in p:initialized[0]=True
            tracked[0]+=1
            mask=np.ones((12,12),np.uint8) if initialized[0] else np.zeros((12,12),np.uint8)
            return dict(frame_id=p['frame_id'],session=p['session'],status='tracking' if initialized[0] else 'click_to_initialize',
                grasp_allowed=initialized[0],mask_png=png_mask(mask),filtered_mask_png=png_mask(mask),
                service_ms=1,mask_depth_ms=1,mask_pixels=int(mask.sum()),filtered_pixels=int(mask.sum()),
                metrics=dict(sam_init_ms=1 if 'point' in p else None,vos_ms=1)),.1
    monkeypatch.setattr(tracking_client,'GraspClient',Client)
    capture=Capture();worker=TrackingWorker(capture,FilterConfig(),fps=30,output=tmp_path).start()
    try:
        wait_for(lambda:worker.state()[0])
        worker.select(capture.latest(),point=(4,4))
        wait_for(lambda:worker.state()[0] and worker.state()[0]['response']['grasp_allowed'])
        worker.request_grasp()
        # Reservation covers the time before the network thread acknowledges submission.
        with pytest.raises(ValueError):worker.request_grasp()
        error=wait_for(worker.take_grasp_error)
        assert error['revision']==worker.revision
        at_error=tracked[0]
        wait_for(lambda:tracked[0]>=at_error+3)
        latest,status,busy=worker.state()
        assert initialized[0] and latest['response']['grasp_allowed']
        assert status=='tracking' and not busy
        assert 'reset' not in actions
        assert actions.count('grasp_submit')==1
    finally:worker.close()


def test_submission_and_unconsumed_result_are_both_single_flight(tmp_path):
    worker=TrackingWorker(Capture(),FilterConfig(),output=tmp_path)
    worker.request_grasp()
    assert worker.state()[2] is True
    with pytest.raises(ValueError):worker.request_grasp()
    with worker.lock:
        worker.grasp_pending=None;worker.grasp_busy=False;worker.completed={'fixture':True}
    with pytest.raises(ValueError):worker.request_grasp()
    assert worker.take_result()=={'fixture':True}
    worker.request_grasp()


def test_auto_retries_busy_without_breaking_mask_pose_continuity():
    class Session:
        failures=0
        breaks=0
        def break_stability(self):self.breaks+=1
        def event(self,*args,**kwargs):pass
        def requested(self,*args):pass
    class Worker:
        lock=threading.Lock();revision=0;grasp_pending=None
        error=dict(message='One grasp job is already running',revision=0)
        def state(self):return dict(frame=Capture().latest(),response=dict(grasp_allowed=True)),'tracking',False
        def take_grasp_error(self):value=self.error;self.error=None;return value
        def request_grasp(self):self.grasp_pending=1
    session=Session();worker=Worker();auto=ContinuousGrasp(session)
    auto.inflight=True;auto.revision=0
    assert not auto.tick(worker)
    assert not auto.inflight and session.failures==1 and session.breaks==0
    assert 'already running' in auto.last_error
    assert not auto.tick(worker)  # Backoff prevents retry spam.
    auto.retry_after=0
    assert auto.tick(worker)
