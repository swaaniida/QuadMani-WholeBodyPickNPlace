"""Replay an existing recording with the real nominal camera/mount/UMI rig."""
import argparse
from pathlib import Path
import time
import webbrowser
import numpy as np
import viser
from view_temporal import ComparisonViewer,latest_recording
from temporal_replay import advance_playback,sample_at_time
from mounted_wrist_rig import MountedWristRig,DEFAULT_RIG


class WristReplay(ComparisonViewer):
    def __init__(self,session,rig,port):
        self.mounted_rig=rig;self.rig_ready=False
        super().__init__(session,port)
        if rig.calibration:
            serial=rig.calibration.get('serial')
            recorded=self.rows[0]['frame_id'].split('-',1)[0]
            if serial and recorded.isdigit() and recorded!=serial:
                self.close();raise ValueError('Device calibration serial does not match this recording')
        with self.controls.add_tab('Actual wrist rig'):
            self.rig_mesh=self.viewer.server.gui.add_checkbox('Camera + mount + actual UMI',initial_value=True)
            self.rig_tcp=self.viewer.server.gui.add_checkbox('Actual nominal TCP axes',initial_value=True)
            xyz=rig.T_camera_tcp[:3,3]*1000
            self.viewer.server.gui.add_markdown(f'**CAD nominal TCP, not hand-eye calibrated**\n\nColor optical -> TCP translation (mm): `{xyz.round(3).tolist()}`\n\nActual UMI is static OPEN. GraspGenX candidates remain franka_umi gripper-base poses.')
        for h in (self.rig_mesh,self.rig_tcp):h.on_update(lambda _:setattr(self,'dirty',True))
        self.rig_ready=True;self.dirty=True

    def show(self,index):
        super().show(index)
        if not self.rig_ready or not self.motion_ready.is_set() or self.motion_error:return
        T=self.display_camera(self.rows[index]['frame_id'])
        if T is not None:
            with self.viewer.server.atomic():self.mounted_rig.draw(self.viewer.server,T,self.rig_mesh.value,self.rig_tcp.value)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session',type=Path,nargs='?')
    p.add_argument('--rig',type=Path,default=DEFAULT_RIG);p.add_argument('--calibration',type=Path)
    p.add_argument('--rig-only',action='store_true');p.add_argument('--port',type=int,default=8084);p.add_argument('--no-browser',action='store_true')
    a=p.parse_args()
    calibration=a.calibration or (Path('realsense_calibration.json') if Path('realsense_calibration.json').exists() else None)
    rig=MountedWristRig(a.rig,calibration)
    print('T_color_optical_TCP (nominal CAD):\n',rig.T_camera_tcp,flush=True)
    if a.rig_only:
        server=viser.ViserServer(host='127.0.0.1',port=a.port,label='PiPER UMI D435i nominal rig')
        server.scene.set_up_direction('-y');server.scene.add_frame('/camera_color_optical',axes_length=.07,axes_radius=.0015)
        rig.draw(server,np.eye(4));server.gui.add_markdown('CAD nominal assembly; not hand-eye calibrated. Blue = mount, gray/black = actual UMI. RGB optical X right / Y down / Z forward.')
        @server.on_client_connect
        def connected(client):
            center=rig.T_camera_tcp[:3,3]/2
            client.camera.look_at=center;client.camera.position=center+np.array([.3,-.2,-.35]);client.camera.up_direction=(0,-1,0)
        url=f'http://127.0.0.1:{server.get_port()}';print(url,flush=True)
        if not a.no_browser:webbrowser.open(url)
        try:
            while True:time.sleep(.1)
        except KeyboardInterrupt:pass
        finally:server.stop()
        return
    session=a.session or latest_recording();viewer=WristReplay(session,rig,a.port)
    url=f'http://127.0.0.1:{viewer.viewer.server.get_port()}';print('SESSION:',session,'\n',url,flush=True)
    if not a.no_browser:webbrowser.open(url)
    previous=time.perf_counter();cursor=0.;seek=viewer.seek
    try:
        while True:
            now=time.perf_counter();dt=now-previous;previous=now
            if viewer.seek!=seek:cursor=float(viewer.slider.value);seek=viewer.seek
            if viewer.play.value and viewer.motion_ready.is_set() and viewer.motion is not None and not viewer.motion_error and viewer.viewer.server.get_clients():
                cursor,playing=advance_playback(cursor,dt,float(viewer.speed.value),float(viewer.times[-1]),viewer.loop.value,viewer.hold)
                viewer.play.value=playing;viewer.slider.value=min(cursor,float(viewer.times[-1]))
            index=sample_at_time(viewer.times,viewer.slider.value)
            if index!=viewer.index or viewer.dirty:viewer.dirty=False;viewer.show(index)
            time.sleep(.02)
    except KeyboardInterrupt:pass
    finally:viewer.close()


if __name__=='__main__':main()
