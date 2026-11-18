"""Replay the identical candidate sets: magenta baseline vs green temporal/ground top1."""
import argparse
import base64
from contextlib import nullcontext
import csv
import html
from pathlib import Path
import time
import threading
import webbrowser
import cv2
import numpy as np
from visualization import GraspViewer
from official_viser_helpers import matrix_to_wxyz_position
from camera_motion import read_rgbd,transform_points
from replay_camera import reconstruct,trajectory_lines
from temporal_replay import calibrated_frustum,sample_times,sample_at_time,advance_playback,loop_hold,playback_style,fuse_clouds


def metric_text(value,scale=1.,digits=3):
    try:
        number=float(value)*scale
        if not np.isfinite(number):return '—'
        return f'{number:.{digits}g}'
    except (TypeError,ValueError):return '—'


def preview_html(rgb,frame_id,depth=None,mask=None):
    previews=[('RGB + mask',rgb,'.jpg')]
    if depth is not None:
        valid=np.isfinite(depth)&(depth>0)
        scaled=np.clip(np.nan_to_num(depth,nan=0.,posinf=0.,neginf=0.)/2.*255,0,255).astype(np.uint8)
        color=cv2.applyColorMap(scaled,cv2.COLORMAP_TURBO)[:,:,::-1].copy();color[~valid]=0
        previews.append(('Depth',color,'.jpg'))
    if mask is not None:
        binary=np.repeat(((np.asarray(mask)>0).astype(np.uint8)*255)[:,:,None],3,axis=2)
        previews.append(('Tracking mask',binary,'.png'))
    cards=[]
    for index,(title,image,extension) in enumerate(previews):
        options=[cv2.IMWRITE_JPEG_QUALITY,85] if extension=='.jpg' else []
        ok,encoded=cv2.imencode(extension,image[:,:,::-1],options)
        if not ok:continue
        payload=base64.b64encode(encoded).decode('ascii');mime='image/jpeg' if extension=='.jpg' else 'image/png'
        span='grid-column:1 / -1;' if index==0 else ''
        height=180 if index==0 else 100
        cards.append(f'<div style="{span}min-width:0;text-align:center"><div style="font-weight:600;padding-bottom:3px">'+title+
            f'</div><img style="width:100%;height:{height}px;object-fit:contain;background:#15191f;border-radius:4px" src="data:{mime};base64,{payload}"/></div>')
    return ('<div style="font:11px/1.25 sans-serif"><div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px">'+
            ''.join(cards)+'</div><div style="padding-top:3px">Depth: 0–2 m (farther saturated); black = invalid. Mask: white = object.</div>'+
            f'<div style="overflow-wrap:anywhere;padding-top:3px">{html.escape(str(frame_id))}</div></div>')


def compact_stats_html(row,world=True,error=None):
    escape=lambda value:html.escape(str(value))
    metric=lambda key,scale=1.:metric_text(row.get(key),scale)
    baseline_ok=str(row.get('baseline_ground_valid','')).lower()
    baseline_state='PASS' if baseline_ok=='true' else 'REJECT' if baseline_ok=='false' else '—'
    new_state='PASS' if row.get('new_index') not in ('',None,'-1') else 'REJECT'
    values=[('Confidence',metric('baseline_confidence'),metric('confidence')),
            ('Score',metric('baseline_score'),metric('final_score')),
            ('Temporal',metric('baseline_temporal_score'),metric('temporal_score')),
            ('Δ position / mm',metric('baseline_delta_mm'),metric('new_delta_mm')),
            ('Δ rotation / °',metric('baseline_delta_deg'),metric('new_delta_deg')),
            ('Collision',baseline_state,new_state)]
    tr=lambda cells:'<tr>'+''.join(f'<td style="padding:4px 2px;border-bottom:1px solid #8883">{escape(c)}</td>' for c in cells)+'</tr>'
    compare='<table style="width:100%;border-collapse:collapse">'+tr(('', 'Baseline','Selected'))+''.join(tr(cells) for cells in values)+'</table>'
    timing=[('ICP / FPS',metric('icp_fps')),('ICP processing / ms',metric('icp_processing_ms')),
            ('Grasp inference / ms',metric('grasp_infer_ms')),('End-to-end / ms',metric('end_to_end_ms'))]
    table='<table style="width:100%;border-collapse:collapse">'+''.join(tr(cells) for cells in timing)+'</table>'
    temporal_on=str(row.get('temporal_used','')).lower()=='true'
    reason={'used':'active','no_previous_top1_or_capture_gap':'first pose / capture gap',
            'no_previous_valid_top1_or_selection_changed':'no prior pose / selection changed',
            'icp_unreliable_or_missing':'ICP unavailable'}.get(row.get('temporal_reason'),row.get('temporal_reason',''))
    status=f'ICP: {escape(row.get("icp_status","—"))} · Fitness {metric("icp_fitness")} · RMSE {metric("icp_rmse_m",1000)} mm'
    note=('<div style="color:#d77">'+escape(error or 'Replay pose unavailable')+'</div>') if error or not world else ''
    return ('<div style="font:12px/1.25 sans-serif">'+note+compare+'<div style="height:8px"></div>'+table+
            f'<div style="padding-top:7px">{status}</div><div style="padding-top:5px" title="{escape(row.get("temporal_reason",""))}">Temporal: {"ON" if temporal_on else "OFF"} · {escape(reason)}</div></div>')


def initial_orbit_pivot(object_points,scene_points,plane=None):
    # One initialization per replay; no per-frame following of object/plane noise.
    fallback=(np.min(scene_points,axis=0)+np.max(scene_points,axis=0))/2
    if plane is None:return fallback,np.array([0.,-1.,0.])
    plane=np.asarray(plane,float)/np.linalg.norm(plane[:3])
    center=np.mean(object_points,axis=0)
    projection=center-plane[:3]*(center@plane[:3]+plane[3])
    return (center+projection)/2,plane[:3].copy()


def replay_thread_limit():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()  # CPU thread tuning must not disable camera reconstruction.
    return threadpool_limits(limits=2)


class ComparisonViewer:
    def __init__(self,path,port=8080):
        self.path=Path(path)
        with (self.path/'comparison.csv').open(newline='') as f:self.rows=list(csv.DictReader(f))
        if not self.rows:raise ValueError('No completed selection comparisons')
        self.viewer=GraspViewer(port);gui=self.viewer.server.gui
        # Keep GraspViewer's rendering support; replace its tall default widgets.
        self.viewer.info.visible=False;self.viewer.details.visible=False;self.viewer.mesh_checkbox.visible=False
        gui.main_panel.dock_right();gui.main_panel.set_width(340)
        self.controls=gui.add_panel();self.controls.dock_left();self.controls.set_width(300)
        self.grasp_numbers=[];count=0
        for row in self.rows:
            has_pose=row.get('new_index') not in ('',None,'-1')
            if has_pose:count+=1
            self.grasp_numbers.append(count if has_pose else None)
        self.times=sample_times(self.rows);self.hold=loop_hold(self.times)
        with self.controls.add_tab('Playback & display'):
            gui.add_markdown('**Baseline: magenta · Selected: green**')
            self.viewer.info=gui.add_markdown('Preparing replay…')
            self.slider=gui.add_slider('Time (seconds)',min=0.,max=max(.001,float(self.times[-1])),step=.001,initial_value=0.)
            self.play=gui.add_checkbox('Play',initial_value=True);self.loop=gui.add_checkbox('Loop',initial_value=True)
            self.speed=gui.add_slider('Speed',min=.1,max=4.,step=.1,initial_value=1.)
            self.viewer.mesh_checkbox=gui.add_checkbox('Show grippers',initial_value=True)
            self.show_baseline=gui.add_checkbox('Baseline',initial_value=True)
            self.show_new=gui.add_checkbox('Selected grasp',initial_value=True)
            self.show_ground=gui.add_checkbox('Plane',initial_value=True)
            self.show_rejected=gui.add_checkbox('Rejected candidates',initial_value=False)
            self.rejected_limit=gui.add_slider('Max rejected',min=1,max=30,step=1,initial_value=8,visible=False)
            self.ground_info=gui.add_markdown('Collision check pending…')
            self.reconstruction_status=gui.add_markdown('Restoring camera trajectory…')
        self.orbit_initialized=False;self.orbit_up=np.array([0.,-1.,0.]);self.view_mode='Oblique'
        with self.controls.add_tab('View'):
            self.orbit_status=gui.add_markdown('Center is initialized once after reconstruction.')
            self.pivot_input=gui.add_vector3('Orbit center XYZ (m)',initial_value=(0.,0.,0.),step=.005,disabled=True)
            self.side_button=gui.add_button('Side view (plane level)')
            self.oblique_button=gui.add_button('Oblique view')
            self.side_elevation=gui.add_slider('Side elevation (degrees)',min=0.,max=45.,step=1.,initial_value=5.)
            self.reset_pivot_button=gui.add_button('Reset initial center')
            self.fit_button=gui.add_button('Fit entire scene')
            self.show_pivot=gui.add_checkbox('Show orbit center',initial_value=False)
        @self.pivot_input.on_update
        def change_pivot(event):
            if event.client_id is not None and self.orbit_initialized:self.set_pivot(self.pivot_input.value)
        @self.side_button.on_click
        def side(_):self.set_view('Side')
        @self.oblique_button.on_click
        def oblique(_):self.set_view('Oblique')
        @self.side_elevation.on_update
        def elevation(event):
            if event.client_id is not None and self.view_mode=='Side':self.set_view('Side')
        @self.reset_pivot_button.on_click
        def reset_center(_):
            if self.orbit_initialized:self.set_pivot(self.initial_pivot)
        @self.fit_button.on_click
        def fit(_):
            if self.orbit_initialized:
                self.view_distance=self.full_distance
                for client in self.viewer.server.get_clients().values():self.aim(client)
        @self.show_pivot.on_update
        def pivot_visible(_):self.dirty=True
        @self.viewer.mesh_checkbox.on_update
        def toggle_mesh(_):
            for handle in self.viewer.mesh_handles:handle.visible=self.viewer.mesh_checkbox.value
        self.image=gui.add_html('<div style="height:180px;text-align:center">Preparing RGB…</div>')
        self.viewer.details=gui.add_html('<div>Preparing scores…</div>')
        self.dirty=True;self.seek=0;self.fitted=False;self.index=-1
        with np.load(self.path/self.rows[0]['candidate_file'],allow_pickle=False) as data:
            self.fixed=data['point_cloud'].copy();self.fixed_colors=data['colors'].copy()
        self.camera_ids=[];self.camera_poses=[];self.motion=None;self.lookup={};self.motion_error=None
        self.motion_ready=threading.Event();self.cancel=threading.Event()
        self.loading_modal=gui.add_modal('Reconstruction')
        with self.loading_modal:
            self.loading_title=gui.add_html('<div style="text-align:center;font-size:28px;font-weight:700;padding:24px 0">RECONSTRUCTING...</div>')
            self.loading_progress=gui.add_markdown('Preparing saved RGB-D frames…')
        self.thread=threading.Thread(target=self._reconstruct,name='comparison-replay-motion',daemon=True)
        @self.slider.on_update
        def changed(event):
            if event.client_id is not None:
                self.play.value=False;self.seek+=1;self.dirty=True
        for h in (self.show_baseline,self.show_new,self.show_ground,self.show_rejected,self.rejected_limit):
            h.on_update(lambda _:setattr(self,'dirty',True))
        @self.viewer.server.on_client_connect
        def connected(client):self.aim(client)
        self.show(0)
        self.thread.start()

    def _reconstruct(self):
        try:
            def progress(done,total):
                self.reconstruction_status.content=f'Replay RGB-D reconstruction: {done}/{total}'
                self.loading_progress.content=f'Frames: **{done} / {total}**'
                if done%max(1,total//10)==0:print(f'Replay RGB-D reconstruction {done}/{total}',flush=True)
            with replay_thread_limit():self.motion=reconstruct(self.path,self.rows,progress,self.cancel)
            self.lookup={str(fid):i for i,fid in enumerate(self.motion['frame_ids'])}
            self.camera_ids=list(map(str,self.motion['frame_ids']))
            self.camera_poses=self.motion['T_world_camera']
            clouds=[];colors=[]
            for row in self.rows:
                i=self.lookup.get(row['frame_id'])
                if i is None or not self.motion['valid'][i]:continue
                with np.load(self.path/row['candidate_file'],allow_pickle=False) as data:
                    clouds.append(transform_points(data['point_cloud'],self.camera_poses[i]));colors.append(data['colors'])
            if clouds:self.fixed,self.fixed_colors=fuse_clouds(clouds,colors)
            self.reconstruction_status.content=f'Replay ready: **{int(self.motion["valid"].sum())}/{len(self.motion["valid"])}** camera frames'
            self.fitted=False
        except Exception as exc:
            self.motion_error=str(exc);self.reconstruction_status.content=f'Replay reconstruction unavailable: {exc}'
            self.loading_title.content='<div style="text-align:center;font-size:24px;font-weight:700;padding:20px 0">RECONSTRUCTION FAILED</div>'
            self.loading_progress.content='Error: '+html.escape(str(exc))
            print(f'REPLAY RECONSTRUCTION ERROR: {type(exc).__name__}: {exc}',flush=True)
        finally:self.motion_ready.set();self.dirty=True

    def frame_plane(self,index):
        row=self.rows[index];T=self.display_camera(row['frame_id'])
        if T is None:return None
        with np.load(self.path/row['candidate_file'],allow_pickle=False) as data:plane=data['ground_plane'].copy()
        if not np.isfinite(plane).all() or np.linalg.norm(plane[:3])<1e-8:return None
        transformed=np.linalg.inv(T).T@plane
        return transformed/np.linalg.norm(transformed[:3])

    def initialize_orbit(self):
        if self.orbit_initialized:return
        points=np.vstack((self.fixed,np.zeros((1,3)),self.camera_poses[self.motion['valid'],:3,3]))
        self.pivot,self.orbit_up=initial_orbit_pivot(self.fixed,points,None)
        self.initial_pivot=self.pivot.copy();self.initial_up=self.orbit_up.copy()
        self.full_distance=max(.4,float(np.linalg.norm(np.ptp(points,axis=0))))*1.5
        self.view_distance=self.full_distance
        self.orbit_initialized=True
        self.pivot_input.value=tuple(map(float,self.pivot));self.pivot_input.disabled=False
        self.orbit_status.content='Original scene center. Playback does not move it.'

    def set_pivot(self,point):
        point=np.asarray(point,float)
        if not np.isfinite(point).all():return
        self.pivot=point.copy();self.pivot_input.value=tuple(map(float,point))
        for client in self.viewer.server.get_clients().values():
            # Translate the view without changing its direction or zoom.
            client.camera.position=np.asarray(client.camera.position)+point-np.asarray(client.camera.look_at)
            client.camera.look_at=point
        self.orbit_status.content='Manual fixed orbit center; playback does not move it.'
        self.dirty=True

    def set_view(self,mode):
        if not self.orbit_initialized:return
        if mode=='Side':
            plane=self.frame_plane(int(self.index))
            if plane is None:
                self.orbit_status.content='Current plane unavailable; center/view kept. XYZ remains editable.'
                return
            self.orbit_up=plane[:3].copy()
        self.view_mode=mode
        self.view_distance=max(.4,float(np.linalg.norm(np.ptp(self.fixed,axis=0)))*2.5)
        self.viewer.server.scene.set_up_direction(self.orbit_up)
        for client in self.viewer.server.get_clients().values():self.aim(client)
        self.dirty=True

    def aim(self,client):
        if not self.orbit_initialized:return
        up=self.orbit_up/np.linalg.norm(self.orbit_up)
        seed=np.array([.8,-.5,-1.2])
        horizontal=seed-up*(seed@up)
        if np.linalg.norm(horizontal)<1e-6:horizontal=np.cross(up,[1.,0.,0.])
        horizontal/=np.linalg.norm(horizontal)
        elevation=np.deg2rad(float(self.side_elevation.value) if self.view_mode=='Side' else 30.)
        direction=(seed if self.view_mode=='Oblique' else
                   np.cos(elevation)*horizontal+np.sin(elevation)*up)
        client.camera.position=self.pivot+self.view_distance*direction
        client.camera.look_at=self.pivot;client.camera.up_direction=up

    def display_camera(self,frame_id):
        ready=self.motion_ready.is_set() and not self.motion_error and self.motion is not None
        i=self.lookup.get(frame_id)
        if ready and i is not None and self.motion['valid'][i]:return self.camera_poses[i]
        # Identity is exact for the reference capture by coordinate definition.
        # Never use it as a fallback pose for any later capture.
        if frame_id==self.rows[0]['frame_id']:return np.eye(4)
        return None

    def show(self,index):
        self.index=index;r=self.rows[index];scene=self.viewer.server.scene
        self.rejected_limit.visible=self.show_rejected.value
        with self.viewer.server.atomic():
            scene.reset();self.viewer.mesh_handles=[]
            if not self.motion_ready.is_set() or self.motion_error:
                self.viewer.info.content='RECONSTRUCTION FAILED' if self.motion_error else 'RECONSTRUCTING...'
                return
            self.initialize_orbit()
            scene.set_up_direction(self.orbit_up)
            if self.show_pivot.value:
                scene.add_point_cloud('/orbit_center',points=self.pivot[None].astype(np.float32),colors=(255,100,20),point_size=.007)
                scene.add_label('/orbit_center/label','Orbit center',position=self.pivot)
            with np.load(self.path/r['candidate_file'],allow_pickle=False) as d:
                data={k:d[k].copy() for k in d.files}
            T=self.display_camera(r['frame_id']);world=T is not None
            scene.add_frame('/reference',axes_length=.1,axes_radius=.002)
            if self.fixed is not None:
                scene.add_gaussian_splats('/fixed',centers=self.fixed,
                    covariances=np.broadcast_to(np.eye(3)*1e-6,(len(self.fixed),3,3)),rgbs=self.fixed_colors,
                    opacities=np.full((len(self.fixed),1),.25,np.float32))
            if world:
                points=transform_points(data['point_cloud'],T)
                scene.add_point_cloud('/observed',points=points,colors=data['colors'],point_size=.003)
            if self.motion_ready.is_set() and self.motion is not None and not self.motion_error:
                lines=trajectory_lines(self.camera_poses,self.motion['valid'],self.motion['segments'])
                if len(lines):scene.add_line_segments('/cameras/path',points=lines,colors=(175,195,220),thickness=2.,thickness_units='screen')
                ids=np.flatnonzero(self.motion['valid'])
                if len(ids):
                    scene.add_point_cloud('/cameras/positions',points=self.camera_poses[ids,:3,3],colors=(195,210,230),point_size=.003)
                    scene.add_label('/cameras/START','START',position=self.camera_poses[ids[0],:3,3])
            frame=read_rgbd(self.path/r['frame_file']);image=frame['rgb'].copy()
            with np.load(self.path/r['frame_file'],allow_pickle=False) as d:mask=d['instance_mask']>0
            image[mask]=np.rint(image[mask]*.7+np.array([0,255,80])*.3).astype(np.uint8)
            self.image.content=preview_html(image,r['frame_id'],depth=frame['depth'],mask=mask)
            if world:scene.add_line_segments('/current_camera',points=calibrated_frustum(T,frame['K'],image.shape[1],image.shape[0]),colors=(0,65,255),thickness=4.,thickness_units='screen')
            for kind,key,color,visible in [('baseline','baseline_index',(230,0,210),self.show_baseline.value),('new','new_index',(0,180,55),self.show_new.value)]:
                j=int(data[key]);
                if j<0 or not visible or not world:continue
                pose=T@data['grasps'][j];q,p=matrix_to_wxyz_position(pose)
                clearance=float(data['clearance_m'][j]);ground_ok=bool(data['collision_valid'][j])
                state='PASS' if ground_ok else ('REJECT' if np.isfinite(clearance) else 'UNVERIFIED')
                if kind=='baseline' and not ground_ok and np.isfinite(clearance):color=(240,25,25)
                opacity,_=playback_style(index,len(self.rows))
                if self.viewer.mesh is not None:
                    h=scene.add_mesh_simple('/'+kind+'/CAD',vertices=self.viewer.mesh.vertices,faces=self.viewer.mesh.faces,
                        wxyz=q,position=p,color=color,opacity=opacity,visible=self.viewer.mesh_checkbox.value)
                    self.viewer.mesh_handles.append(h)
                scene.add_frame('/'+kind+'/axes',wxyz=q,position=p,axes_length=.055 if kind=='new' else .04,axes_radius=.001)
                label=f'Grasp {self.grasp_numbers[index]}' if kind=='new' else f'Baseline {index+1} | {state}'
                scene.add_label('/'+kind+'/label',label,position=p+([0,-.025,0] if kind=='baseline' else [0,.025,0]))
            plane_known=bool(np.isfinite(data['ground_plane']).all())
            rejected=np.flatnonzero((~data['collision_valid'].astype(bool))&np.isfinite(data['clearance_m']))
            self.ground_info.content=(f'Pass: {int(data["collision_valid"].sum())} / {len(data["grasps"])} | Reject: {len(rejected)}') if plane_known else 'Plane unavailable; collision unverified.'
            if world and plane_known and self.show_rejected.value and self.viewer.mesh is not None:
                order=rejected[np.argsort(-data['confidences'][rejected],kind='stable')][:int(self.rejected_limit.value)]
                for j in order:
                    color=(240,25,25)
                    local_pose=data['grasps'][j];pose=T@local_pose;q,p=matrix_to_wxyz_position(pose)
                    if not (self.show_baseline.value and j==int(data['baseline_index'])):
                        handle=scene.add_mesh_simple(f'/rejected/{j}/CAD',vertices=self.viewer.mesh.vertices,faces=self.viewer.mesh.faces,
                            wxyz=q,position=p,color=color,opacity=.22,visible=self.viewer.mesh_checkbox.value)
                        self.viewer.mesh_handles.append(handle)
                    scene.add_label(f'/rejected/{j}/label','REJECT',position=p)
            if world and plane_known and self.show_ground.value:
                plane=data['ground_plane'];n=plane[:3];center=np.mean(data['point_cloud'],axis=0)
                center=center-n*(center@n+plane[3]);seed=np.array([1,0,0]) if abs(n[0])<.9 else np.array([0,1,0])
                x=np.cross(n,seed);x/=np.linalg.norm(x);y=np.cross(n,x)
                corners=center+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])@np.stack((x,y))*.25
                scene.add_mesh_simple('/ground/plane',vertices=transform_points(corners,T),faces=np.array([[0,1,2],[0,2,3]]),color=(210,170,65),opacity=.35,side='double')
            stage='READY' if self.motion_ready.is_set() and not self.motion_error else ('RECONSTRUCTION FAILED — reference only' if self.motion_error else 'RECONSTRUCTING — reference capture visible')
            self.viewer.info.content=f'**Grasp {self.grasp_numbers[index] or "—"} · Sample {index+1}/{len(self.rows)}**\n\n{r["status"]} · {r["candidate_count"]} → {r["after_ground_count"]} candidates'
            if world:
                q,position=matrix_to_wxyz_position(T)
                scene.add_frame('/camera_axes',wxyz=q,position=position,axes_length=.07,axes_radius=.002)
                scene.add_point_cloud('/camera_highlight',points=position[None],colors=(0,65,255),point_size=.012)
                scene.add_label('/camera_label','Camera',position=position)
            self.viewer.details.content=compact_stats_html(r,world,self.motion_error)
            if self.loading_modal is not None:
                self.loading_modal.close();self.loading_modal=None
            if not self.fitted:
                for client in self.viewer.server.get_clients().values():self.aim(client)
                self.fitted=True

    def close(self):
        self.cancel.set();self.thread.join(timeout=1);self.viewer.close()


def review(path,port=8080,open_browser=True):
    v=ComparisonViewer(path,port);url=f'http://127.0.0.1:{v.viewer.server.get_port()}'
    print('Comparison replay:',url,flush=True)
    if open_browser:webbrowser.open(url)
    previous=time.perf_counter();cursor=0.;seek=v.seek
    try:
        while True:
            now=time.perf_counter();dt=now-previous;previous=now
            if v.seek!=seek:cursor=float(v.slider.value);seek=v.seek
            if v.play.value and v.motion_ready.is_set() and v.motion is not None and not v.motion_error and v.viewer.server.get_clients():
                cursor,playing=advance_playback(cursor,dt,float(v.speed.value),float(v.times[-1]),v.loop.value,v.hold)
                v.play.value=playing;v.slider.value=min(cursor,float(v.times[-1]))
            index=sample_at_time(v.times,v.slider.value)
            if index!=v.index or v.dirty:v.dirty=False;v.show(index)
            time.sleep(.02)
    except KeyboardInterrupt:pass
    finally:v.close()


def latest_recording(output_root=None):
    root=Path(output_root) if output_root is not None else Path(__file__).parent/'outputs'
    candidates=sorted(root.glob('tracking*/auto_sessions/*/comparison.csv'),
                      key=lambda p:p.stat().st_mtime,reverse=True)
    for candidate in candidates:
        with candidate.open(newline='') as f:
            if next(csv.DictReader(f),None) is not None:return candidate.parent
    raise FileNotFoundError(f'No temporal recording with comparison rows found under {root}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session',type=Path,nargs='?',help='Defaults to latest temporal recording');p.add_argument('--port',type=int,default=8080);p.add_argument('--no-browser',action='store_true')
    args=p.parse_args()
    try:session=args.session if args.session is not None else latest_recording()
    except FileNotFoundError as exc:p.error(str(exc))
    print('SESSION:',session,flush=True)
    review(session,args.port,not args.no_browser)
