"""CPU render one checkpoint's recorded PhysX state; no physics integration."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_checkpoint_video import build, PROJECT


def line(scene, a, b, width, color):
    if np.linalg.norm(np.asarray(a)-b)<1e-6:
        return
    g=scene.geoms[scene.ngeom]
    scene.ngeom+=1
    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32))
    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,width,np.asarray(a),np.asarray(b))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--folder',required=True)
    p.add_argument('--index',type=int,required=True)
    p.add_argument('--preview',action='store_true')
    p.add_argument('--stride',type=int,default=2)
    a=p.parse_args()
    folder=Path(a.folder).resolve()
    meta=json.loads((folder/'metadata.json').read_text())
    z=np.load(folder/'states.npz')
    j=a.index
    it=meta['iterations'][j]
    m,d,renderer,cam,qids=build(folder,meta,j)
    renderer.close()
    m.vis.quality.offsamples=0
    m.vis.quality.shadowsize=256
    renderer=mujoco.Renderer(m,height=270,width=480)
    m.vis.headlight.ambient[:]=[.35]*3
    m.vis.headlight.diffuse[:]=[.45]*3
    m.mat_specular[:]=0.
    for gi in range(m.ngeom):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,gi) or ''
        if name.startswith('v_') and 'piper' not in name and 'orin' not in name:
            m.geom_rgba[gi,:3]=[.56,.62,.68]
    cam.distance=2.2
    cam.azimuth=130
    cam.elevation=-22
    cam.lookat[:]=[.25,0,.34]  # Identical fixed viewpoint for all policies.
    font_path=PROJECT/'unitree_rl_mjlab/reports/stability_flat_force_comparison_20261110/assets/NotoSansKR-Medium.ttf'
    font=ImageFont.truetype(str(font_path),21)
    small=ImageFont.truetype(str(font_path),17)
    output=folder.parent/('checkpoint_%d.mp4'%it)
    indices=[int(np.argmin(abs(z['time']-18)))] if a.preview else range(0,len(z['time']),a.stride)
    fps=25/a.stride
    writer=None if a.preview else imageio.get_writer(str(output),fps=fps,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-threads','1'])
    max_fk=0.
    reset_frames=[]
    zoom_frames=0
    try:
        for i in indices:
            root=z['root'][i,j]
            d.qpos[:3]=root[:3]
            d.qpos[3:7]=root[[6,3,4,5]]
            d.qpos[qids]=z['q'][i,j]
            mujoco.mj_forward(m,d)
            err=np.linalg.norm(d.body('piper_umi_nominal_tcp').xpos-z['ee'][i,j])
            reset_near=any(e['env']==meta['video_envs'][j] and abs(z['time'][i]-e['time'])<.061 for e in meta['reset_events'])
            assert err<.001 or reset_near, 'TCP replay mismatch: '+str(err)
            if err>=.001:
                reset_frames.append(i)
            else:
                max_fk=max(max_fk,float(err))
            # Preserve azimuth/elevation; expand only when a failed policy drifts out of view.
            expanded=False
            if expanded:
                lo=np.minimum(root[:2]-[.35,.25],z['target'][i,j,:2])
                hi=np.maximum(root[:2]+[.65,.25],z['target'][i,j,:2])
                cam.lookat[:]=[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,.34]
                cam.distance=max(2.2,float(np.linalg.norm(hi-lo))*1.65)
                zoom_frames+=1
            else:
                cam.lookat[:]=[float(root[0])+.25,float(root[1]),.34]
                cam.distance=2.2
            renderer.update_scene(d,camera=cam)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0
            for key,radius,color in [('target',.014,[1,.15,.05,.95]),('ee',.009,[.05,.95,.9,1.])]:
                g=renderer.scene.geoms[renderer.scene.ngeom]
                renderer.scene.ngeom+=1
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([radius]*3),z[key][i,j],np.eye(3).ravel(),np.array(color,dtype=np.float32))
                # Position trail, no future trajectory that hides the robot.
                for k in range(max(0,i-50),i,5):
                    line(renderer.scene,z[key][k,j],z[key][min(k+5,i),j],.0015,color)
            q=z['target_quat'][i,j]
            matrix=np.zeros(9)
            mujoco.mju_quat2Mat(matrix,q[[3,0,1,2]])
            for axis,col in zip(matrix.reshape(3,3).T,[[1,.45,.4,.8],[.4,1,.4,.8],[.4,.55,1,.8]]):
                line(renderer.scene,z['target'][i,j],z['target'][i,j]+axis*.065,.002,col)
            for xx in np.arange(-.5,1.01,.1):line(renderer.scene,[xx,-.5,.001],[xx,.5,.001],.0007,[.65,.69,.70,.6])
            for yy in np.arange(-.5,.51,.1):line(renderer.scene,[-.5,yy,.001],[1.,yy,.001],.0007,[.65,.69,.70,.6])
            panel=Image.new('RGB',(640,500),'#f5f7fa')
            panel.paste(Image.fromarray(renderer.render().copy()).resize((640,360),Image.Resampling.BILINEAR),(0,80))
            draw=ImageDraw.Draw(panel)
            phase=meta['knots'][int(z['phase'][i])][3]
            draw.text((9,2),'%s iteration | %s'%(format(it,','),phase),font=font,fill='#183344')
            draw.text((9,29),'EE 위치 오차 %.1fcm / 방향 오차 %.1f°'%(z['position_error'][i,j]*100,z['orientation_error_deg'][i,j]),font=small,fill='#183344')
            draw.text((9,50),'목표 높이 %.1fcm / 실제 높이 %.1fcm | t=%.2fs'%(z['target'][i,j,2]*100,z['ee'][i,j,2]*100,z['time'][i]),font=small,fill='#183344')
            cmd=z['commands'][i,j]
            draw.text((9,444),'명령 vx %.2f / vy %.2f m/s / yaw %.2f rad/s'%(cmd[0],cmd[1],cmd[2]),font=small,fill='#183344')
            draw.text((9,474),'실제 vx %.2f / vy %.2f m/s / yaw %.2f rad/s'%(z['velocity'][i,j],z['velocity_xy'][i,j,1],z['yaw'][i,j]),font=small,fill='#183344')
            reset=any(e['env']==meta['video_envs'][j] and 0<=z['time'][i]-e['time']<.8 for e in meta['reset_events'])
            if reset:
                draw.rectangle((450,80,634,112),fill='#b33434')
                draw.text((456,83),'실패 RESET',font=font,fill='white')
            elif expanded:
                draw.text((420,84),'몸통 이동: 시야 확대',font=small,fill='#883d20')
            if a.preview or abs(z['time'][i]-18.)<.041:
                panel.save(folder.parent/('preview_%d.jpg'%it))
            if writer is not None:writer.append_data(np.asarray(panel))
            if i%125==0:print(it,'render',i,'/',len(z['time']),flush=True)
    finally:
        if writer is not None:writer.close()
        renderer.close()
    if not a.preview:
        (folder/('render_%d_validation.json'%it)).write_text(json.dumps({'iteration':it,'max_nonreset_fk_error_m':max_fk,'reset_discontinuity_frames':reset_frames,'expanded_view_frames':zoom_frames,'frames':len(indices),'fps':fps,'duration_s':len(indices)/fps,'render':'MuJoCo CPU replay of PhysX states, not MuJoCo physics'},indent=2)+'\n')
        print('Saved',output,flush=True)


if __name__=='__main__':main()
