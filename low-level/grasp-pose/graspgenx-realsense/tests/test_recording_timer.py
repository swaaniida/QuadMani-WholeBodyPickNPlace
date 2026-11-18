"""Reset duration without losing recorded data or changing the session clock."""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_client'))
from main_tracking import recording_elapsed,reset_recording_timer


def test_reset_waits_for_first_new_pose_and_preserves_old_rows():
    events=[]
    session=SimpleNamespace(epoch=100.,rows=[dict(completed_elapsed_s=2.)],event=lambda *a,**k:events.append((a,k)))
    assert recording_elapsed(session,110.)==8.
    reset_recording_timer(session)
    assert recording_elapsed(session,110.) is None
    assert len(session.rows)==1 and session.epoch==100.
    session.rows.append(dict(completed_elapsed_s=12.))
    assert recording_elapsed(session,112.)==0.
    assert recording_elapsed(session,115.)==3.
    reset_recording_timer(session)
    assert recording_elapsed(session,116.) is None
    assert len(events)==2
