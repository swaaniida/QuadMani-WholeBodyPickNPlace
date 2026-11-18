"""Read-only cached-pose geometry inspection. No model or service requests."""
import argparse
import json
from pathlib import Path
import time
import webbrowser
import numpy as np
import trimesh
import viser
from official_viser_helpers import matrix_to_wxyz_position
from temporal_selection import collision_clearance

HERE=Path(__file__).resolve().parent


def load_data(path):
    with np.load(path,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    for key in ('grasps','confidences','point_cloud','ground_plane'):
        if key not in data:raise ValueError(f'Missing {key}; use all_candidates/*.npz')
    if not np.isfinite(data['ground_plane']).all():raise ValueError('Capture has no verified plane')
    return data


def verify(data,mesh):
    g=data['grasps'];plane=data['ground_plane'].astype(float)
    plane/=np.linalg.norm(plane[:3])
    world=np.einsum('nij,vj->nvi',g[:,:3,:3],mesh.vertices)+g[:,None,:3,3]
    brute=(world@plane[:3]+plane[3]).min(axis=1)
    fast=collision_clearance(g,mesh.vertices,plane)
    error=float(np.max(np.abs(brute-fast)))
    if error>1e-9:raise AssertionError(f'Clearance mismatch: {error} m')
    return dict(candidates=len(g),max_clearance_error_m=error,mesh_vertices=len(mesh.vertices),
                mesh_faces=len(mesh.faces),mesh_bounds_m=mesh.bounds.tolist(),
                candidates_checked=[dict(index=int(i),confidence=float(data['confidences'][i]),
                    signed_clearance_mm=float(fast[i]*1000),penetration_mm=float(max(0,-fast[i])*1000))
                    for i in dict.fromkeys([int(np.argmax(fast)),int(np.argmin(fast)),0,1,5]) if i<len(g)])


def transform(points,T):return np.asarray(points)@T[:3,:3].T+T[:3,3]


def box_lines(extents,offset):
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    edges=[(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(corners[i]!=corners[j])==1]
    p=corners*np.asarray(extents)/2+offset
    return p[np.asarray(edges)]


def draw(server,data,mesh,params,index,local=False):
    scene=server.scene;scene.reset();scene.set_up_direction('-y')
    T=data['grasps'][index];plane=data['ground_plane'].astype(float)
    plane/=np.linalg.norm(plane[:3])
    if local:
        view=np.linalg.inv(T);plane=T.T@plane;pose=np.eye(4)
    else:view=np.eye(4);pose=T
    v=transform(mesh.vertices,pose);dist=v@plane[:3]+plane[3]
    faces=mesh.faces;bad=(dist[faces]<0).any(axis=1)
    q,pos=matrix_to_wxyz_position(pose)
    scene.add_frame('/grasp',wxyz=q,position=pos,axes_length=.07,axes_radius=.0015)
    scene.add_frame('/camera',wxyz=matrix_to_wxyz_position(view)[0],position=view[:3,3],axes_length=.08)
    scene.add_point_cloud('/object',points=transform(data['point_cloud'],view),
                          colors=data.get('colors',np.full((len(data['point_cloud']),3),150,dtype=np.uint8)),point_size=.002)
    for label,selected,color in [('clear',~bad,(160,170,185)),('crossing',bad,(240,35,35))]:
        if selected.any():scene.add_mesh_simple('/mesh/'+label,vertices=v,faces=faces[selected],color=color,opacity=.85,side='double')
    below=dist<0
    if below.any():scene.add_point_cloud('/below',points=v[below],colors=(255,0,0),point_size=.003)
    j=int(np.argmin(dist));projection=v[j]-dist[j]*plane[:3]
    scene.add_line_segments('/minimum_distance',points=np.array([[v[j],projection]]),colors=(255,20,20),line_width=4)
    scene.add_label('/minimum_distance/label',f'min clearance {dist[j]*1000:.2f} mm',position=v[j])
    for key,color in [('open',(40,130,255)),('mid',(255,160,30))]:
        lines=box_lines(params['extents_'+key],params['offset_'+key])
        scene.add_line_segments('/sweep/'+key,points=transform(lines.reshape(-1,3),pose).reshape(-1,2,3),colors=color,line_width=3)
    tip=transform(np.array([[0,0,params['fingertip_depth']]]),pose)[0]
    scene.add_point_cloud('/fingertip',points=tip[None],colors=(255,255,0),point_size=.004)
    scene.add_label('/fingertip/label','Official fingertip +Z 177 mm (not robot TCP)',position=tip)
    center=transform(data['point_cloud'],view).mean(axis=0)
    center-=plane[:3]*(center@plane[:3]+plane[3])
    seed=np.array([1.,0,0]) if abs(plane[0])<.9 else np.array([0.,1,0])
    x=np.cross(plane[:3],seed);x/=np.linalg.norm(x);y=np.cross(plane[:3],x)
    corners=center+.35*np.array([-x-y,x-y,x+y,-x+y])
    scene.add_mesh_simple('/plane',vertices=corners,faces=np.array([[0,1,2],[0,2,3]]),color=(210,160,60),opacity=.35,side='double')
    return center


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidates',type=Path,nargs='?',default=HERE/'example_results/gripper_frame_verification.npz')
    parser.add_argument('--port',type=int,default=8082)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args();data=load_data(args.candidates)
    mesh=trimesh.load(HERE/'assets/franka_umi_vis_mesh.obj',force='mesh',process=False)
    params=json.loads((HERE/'assets/franka_umi_sweep.json').read_text())
    print(json.dumps(verify(data,mesh),indent=2),flush=True)
    if args.check_only:return
    server=viser.ViserServer(host='127.0.0.1',port=args.port,label='Gripper frame verification')
    server.gui.add_markdown('Axes: X red / Y green / Z blue. Open sweep: blue; Mid: orange.\n\nRed mesh: triangles with at least one penetrating vertex; red dots: below-plane vertices. CAD is static OPEN; Mid box describes another joint state.')
    slider=server.gui.add_slider('Candidate index (zero-based)',min=0,max=len(data['grasps'])-1,step=1,initial_value=0)
    local=server.gui.add_checkbox('Inspect in gripper local frame',initial_value=False)
    info=server.gui.add_markdown('')
    def redraw():
        with server.atomic():center=draw(server,data,mesh,params,int(slider.value),local.value)
        i=int(slider.value);d=collision_clearance(data['grasps'][i:i+1],mesh.vertices,data['ground_plane'])[0]
        info.content=f'Frame: {data.get("frame_id", "unknown")}\n\nCandidate {i}; C={data["confidences"][i]:.6f}; clearance={d*1000:.3f} mm; penetration={max(0,-d)*1000:.3f} mm'
        return center
    def aim(client):
        T=data['grasps'][int(slider.value)]
        center=transform(data['point_cloud'],np.linalg.inv(T) if local.value else np.eye(4)).mean(axis=0)
        client.camera.look_at=center;client.camera.position=center+np.array([.35,-.25,-.4]);client.camera.up_direction=(0,-1,0)
    slider.on_update(lambda _:redraw())
    @local.on_update
    def change_frame(_):
        redraw()
        for client in server.get_clients().values():aim(client)
    server.on_client_connect(aim);redraw()
    url=f'http://127.0.0.1:{server.get_port()}';print(url,flush=True)
    if not args.no_browser:webbrowser.open(url)
    try:
        while True:time.sleep(.2)
    except KeyboardInterrupt:pass
    finally:server.stop()


if __name__=='__main__':main()
