"""Time-ordered three-branch replay on the existing fixed-object camera replay."""
import argparse
import csv
from html import escape
import json
from pathlib import Path
import time
import webbrowser
import zipfile
import numpy as np
from view_temporal import ComparisonViewer,latest_recording
from temporal_replay import advance_playback,sample_at_time
from official_viser_helpers import visualize_x_grasp,matrix_to_wxyz_position
from obb_branch_analysis import BRANCHES,COLORS,ground_basis


class BranchReplay(ComparisonViewer):
    def __init__(self,session,analysis,port):
        self.analysis=Path(analysis);m=json.loads((self.analysis/'manifest.json').read_text())
        if Path(session).name!=m['session_name']:raise ValueError('Analysis belongs to another recording; specify its original session path')
        self.results={r['frame_id']:r for r in m['frames']}
        self.mounted_rig=None
        try:
            from mounted_wrist_rig import MountedWristRig,DEFAULT_RIG
            if DEFAULT_RIG.exists():
                calibration=Path('realsense_calibration.json')
                self.mounted_rig=MountedWristRig(calibration=calibration if calibration.exists() else None)
        except ImportError:
            pass  # Optional wrist-rig package is independent of branch analysis.
        self.branch_ready=False
        super().__init__(session,port)
        if self.mounted_rig is not None and self.mounted_rig.calibration:
            serial=self.mounted_rig.calibration.get('serial')
            recorded=self.rows[0]['frame_id'].split('-',1)[0]
            if serial and recorded.isdigit() and serial!=recorded:
                self.close();raise ValueError('Device calibration serial does not match this recording')
        self.show_baseline.value=False;self.show_new.value=False;self.show_rejected.value=False
        self.show_baseline.visible=False;self.show_new.visible=False;self.show_rejected.visible=False
        with self.controls.add_tab('OBB branches'):
            self.branch_check={tag:self.viewer.server.gui.add_checkbox(tag,initial_value=True) for tag in BRANCHES}
            self.order_mode=self.viewer.server.gui.add_dropdown('Sort',options=('Shallow penetration','Confidence'),initial_value='Shallow penetration')
            self.rank=self.viewer.server.gui.add_slider('Inspect rank',min=1,max=300,step=1,initial_value=1)
            self.show_all=self.viewer.server.gui.add_checkbox('Show all candidates',initial_value=False)
            self.per_branch=self.viewer.server.gui.add_slider('Shown per branch',min=1,max=50,step=1,initial_value=5)
            self.inspect_only=self.viewer.server.gui.add_checkbox('Only inspected candidate',initial_value=False)
            self.mesh_all=self.viewer.server.gui.add_checkbox('Mesh for every candidate',initial_value=False)
            self.labels=self.viewer.server.gui.add_checkbox('Label top 10 per branch',initial_value=False)
            self.background=self.viewer.server.gui.add_checkbox('Background point cloud',initial_value=True)
            self.actual_rig=self.viewer.server.gui.add_checkbox('Actual camera + UMI nominal TCP',initial_value=True,visible=self.mounted_rig is not None)
            self.viewer.server.gui.add_markdown('Red diffusion / Blue camera OBB / Green world OBB. Default: top 5 per branch plus highlighted candidates. Inspect rank searches ALL enabled candidates. Use Only inspected candidate to isolate one pose. Thick outlines: branch confidence leaders. Highlights may coincide.')
        for h in [*self.branch_check.values(),self.order_mode,self.rank,self.show_all,self.per_branch,self.inspect_only,self.mesh_all,self.labels,self.background,self.actual_rig]:
            h.on_update(lambda _:setattr(self,'dirty',True))
        @self.rank.on_update
        def inspect_rank(event):
            if event.client_id is not None:self.play.value=False
        self.branch_ready=True;self.dirty=True

    def show(self,index):
        super().show(index)
        if not self.branch_ready or not self.motion_ready.is_set() or self.motion_error:return
        entry=self.results.get(self.rows[index]['frame_id']);T=self.display_camera(self.rows[index]['frame_id'])
        if not entry or entry['status']!='ok':
            self.viewer.details.content='<b>Branch analysis unavailable:</b> '+escape(entry.get('error','not analyzed') if entry else 'not analyzed')
            return
        if T is None:
            self.viewer.details.content='Camera registration unavailable for this exact capture; no substitute transform used.';return
        with np.load(self.analysis/entry['result_file'],allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
        scene=self.viewer.server.scene;g=d['grasps'];c=d['confidences'];tags=d['branch_tags'];clear=d['clearance_m']*1000
        order=d['shallow_order'] if self.order_mode.value=='Shallow penetration' else d['confidence_order']
        active=[int(i) for i in order if self.branch_check[str(tags[i])].value]
        limit=max(1,len(active))
        if self.rank.max!=limit:self.rank.max=limit
        if self.rank.value>limit:self.rank.value=limit
        inspect=active[int(self.rank.value)-1] if active else None
        hi=json.loads(str(d['highlights_json']));special={}
        for label,i in hi.items():special.setdefault(i,[]).append(label)
        if inspect is not None:special.setdefault(inspect,[]).append('INSPECT')
        leaders={int(ids[np.argmax(c[ids])]) for tag in BRANCHES if len(ids:=np.flatnonzero(tags==tag))}
        shown=[];seen={tag:0 for tag in BRANCHES}
        for i in active:
            tag=str(tags[i]);seen[tag]+=1
            if self.inspect_only.value:
                if i==inspect:shown.append(i)
            elif self.show_all.value or seen[tag]<=int(self.per_branch.value) or i in leaders or i in special:
                shown.append(i)
        with self.viewer.server.atomic():
            if self.mounted_rig is not None and self.actual_rig.value:
                self.mounted_rig.draw(self.viewer.server,T)
            if self.background.value:
                bg=d['background_points']@T[:3,:3].T+T[:3,3]
                scene.add_point_cloud('/branches/background',points=bg,colors=(145,145,145),point_size=.0015)
            world=T@ground_basis(d['ground_plane']);q,p=matrix_to_wxyz_position(world)
            scene.add_frame('/branches/world_axis',wxyz=q,position=p,axes_length=.1,axes_radius=.0015)
            scene.add_line_segments('/branches/top_down',points=np.array([[p,p-world[:3,2]*.12]]),colors=(20,180,60),line_width=5)
            scene.add_label('/branches/world_axis/label','World top-down (-normal)',position=p)
            branch_seen={tag:0 for tag in BRANCHES}
            for i in shown:
                pose=T@g[i];color=COLORS[str(tags[i])];name=f'/branches/candidate_{i}'
                branch_seen[str(tags[i])]+=1
                visualize_x_grasp(self.viewer.server,name,pose,color=color,
                    width=float(self.viewer.params['extents_open'][0]),depth=float(self.viewer.params['fingertip_depth']),
                    linewidth=5 if i in leaders or i in special else 1)
                q,p=matrix_to_wxyz_position(pose)
                if self.mesh_all.value or i in special or i in leaders:
                    mesh=self.viewer.mesh
                    if mesh is not None:
                        handle=scene.add_mesh_simple(name+'/CAD',vertices=mesh.vertices,faces=mesh.faces,
                            wxyz=q,position=p,color=color,opacity=.75 if i==inspect else .35,side='double',visible=self.viewer.mesh_checkbox.value)
                        self.viewer.mesh_handles.append(handle)
                if i in special or i in leaders or (self.labels.value and branch_seen[str(tags[i])]<=10):
                    rank=int(np.flatnonzero(d['shallow_order']==i)[0])+1
                    confidence_rank=int(np.flatnonzero(d['confidence_order']==i)[0])+1
                    text=f'{tags[i]} #{i} C={c[i]:.3f} P={max(0,-clear[i]):.1f}mm P-rank={rank} C-rank={confidence_rank}'
                    if i in leaders:text+=' | BRANCH C#1'
                    if i in special:text+=' | '+', '.join(special[i])
                    scene.add_label(name+'/label',text,position=p)
                if i in special:scene.add_frame(name+'/axes',wxyz=q,position=p,axes_length=.05,axes_radius=.0015)
        lines=['<b>Same-frame branch comparison</b><table><tr><th>Branch</th><th>N</th><th>Safe</th><th>Contact</th><th>Pen.</th></tr>']
        rmse=float(d['plane_rmse_m'])*1000
        for tag in BRANCHES:
            ids=np.flatnonzero(tags==tag);v=clear[ids]
            lines.append(f'<tr><td>{tag}</td><td>{len(ids)}</td><td>{np.sum(v>rmse)}</td><td>{np.sum((v>=0)&(v<=rmse))}</td><td>{np.sum(v<0)}</td></tr>')
        lines.append('</table>')
        if inspect is not None:
            i=inspect;lines.append(f'<p>Inspect #{i}: {tags[i]}<br>C={c[i]:.6f}<br>Clearance={clear[i]:.3f} mm<br>Penetration={max(0,-clear[i]):.3f} mm<br>xyz={g[i,:3,3].round(4).tolist()}</p>')
        self.viewer.details.content=''.join(lines)
        self.viewer.info.content=f'**OBB replay {index+1}/{len(self.rows)} · showing {len(shown)}/{len(g)} candidates**\n\n{self.rows[index]["frame_id"]}'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session',type=Path,nargs='?')
    p.add_argument('--results',type=Path,default=Path('obb-branch-results.zip'));p.add_argument('--port',type=int,default=8083)
    p.add_argument('--no-browser',action='store_true');a=p.parse_args()
    session=a.session or latest_recording()
    if a.results.suffix.lower()=='.zip':
        analysis=session/'obb_branch_analysis'
        if analysis.exists():
            # Keep earlier result sets; never mix files from different analyses.
            analysis=session/('obb_branch_analysis_'+str(time.time_ns()))
        analysis.mkdir()
        with zipfile.ZipFile(a.results) as z:
            for member in z.infolist():
                target=(analysis/member.filename).resolve()
                if not target.is_relative_to(analysis.resolve()):raise ValueError('Unsafe ZIP path')
            z.extractall(analysis)
    else:analysis=a.results
    viewer=BranchReplay(session,analysis,a.port);url=f'http://127.0.0.1:{viewer.viewer.server.get_port()}'
    print('SESSION:',session,'\nREPLAY:',url,flush=True)
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
