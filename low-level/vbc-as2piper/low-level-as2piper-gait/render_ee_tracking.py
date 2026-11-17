"""Render measured EE command following from saved PhysX rollouts."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent

def load(folder,env_index):
    folder=Path(folder); trace=np.load(folder/'trajectories.npz'); meta=json.loads((folder/'metadata.json').read_text())
    xml=ET.parse(ROOT/'resources/commands/plate_forward.xml').getroot();asset=xml.find('asset')
    for mesh in asset.findall('mesh'):
        mesh.set('file',str((ROOT/'resources/commands'/mesh.get('file')).resolve()))
    ET.SubElement(asset,'texture',name='ee_grid',type='2d',builtin='checker',rgb1='.24 .29 .31',rgb2='.38 .42 .44',width='512',height='512')
    ET.SubElement(asset,'material',name='ee_floor',texture='ee_grid',texrepeat='6 6',reflectance='0')
    ground=xml.find("./worldbody/geom[@name='ground']")
    rough='terrain_replay' in json.loads((folder/'policy_evaluation.json').read_text())
    if rough:
        for geom in xml.findall('.//geom'): geom.set('group','5' if geom.get('group')=='3' else '0')
        xml.find('worldbody').remove(ground)
        for panel in range(4):
            ET.SubElement(asset,'mesh',name='ee_terrain_%d'%panel,file=str((folder/('terrain_panel_%d.obj'%panel)).resolve()))
            ET.SubElement(xml.find('worldbody'),'geom',name='ee_terrain_%d'%panel,type='mesh',mesh='ee_terrain_%d'%panel,
                          rgba='.43 .47 .42 1',group=str(panel+1),contype='0',conaffinity='0')
    else: ground.set('material','ee_floor')
    target_body=ET.SubElement(xml.find('worldbody'),'body',name='ee_target_marker',mocap='true')
    ET.SubElement(target_body,'geom',name='ee_target_sphere',type='sphere',size='.018',rgba='1 .18 .06 .75',contype='0',conaffinity='0')
    actual_body=ET.SubElement(xml.find('worldbody'),'body',name='ee_actual_marker',mocap='true')
    ET.SubElement(actual_body,'geom',name='ee_actual_sphere',type='sphere',size='.010',rgba='.12 1 .16 1',contype='0',conaffinity='0')
    model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode')); data=mujoco.MjData(model)
    model.vis.headlight.ambient[:]=[.65,.65,.65];model.vis.headlight.diffuse[:]=[.8,.8,.8]
    renderer=mujoco.Renderer(model,height=288,width=512); camera=mujoco.MjvCamera();camera.distance=2.0;camera.elevation=-18;camera.azimuth=120
    ids=[int(model.joint(name).qposadr) for name in meta['dof_names']]
    return dict(folder=folder,trace=trace,meta=meta,model=model,data=data,renderer=renderer,camera=camera,ids=ids,rough=rough,
                target_mocap=int(model.body('ee_target_marker').mocapid),actual_mocap=int(model.body('ee_actual_marker').mocapid),env=env_index)

def render(state,index,label,cmd_index,total_frames,elapsed):
    t=state['trace'];e=state['env'];m=state['model'];d=state['data']
    root=t['root'][index,e];d.qpos[:3]=root[:3];d.qpos[3:7]=root[[6,3,4,5]];d.qpos[state['ids']]=t['q'][index,e]
    target=t['target'][index,e];actual=t['ee'][index,e];error=float(t['position_error'][index,e])
    d.mocap_pos[state['target_mocap']]=target;d.mocap_pos[state['actual_mocap']]=actual
    mujoco.mj_forward(m,d);R=d.xmat[1].reshape(3,3)
    cam=state['camera'];cam.azimuth=120;cam.lookat[:]=root[:3]+[.08,0,.18]
    option=mujoco.MjvOption()
    if state['rough']:
        option.geomgroup[:]=0;option.geomgroup[0]=1;option.geomgroup[e+1]=1
    state['renderer'].update_scene(d,camera=cam,scene_option=option)
    image=Image.fromarray(state['renderer'].render().copy());draw=ImageDraw.Draw(image)
    command=t['commands'][index,e]
    draw.rectangle((0,0,512,39),fill=(248,248,248))
    draw.text((7,4),label+' | command EE red, measured EE green',fill=(15,15,15))
    draw.text((7,21),'t %.1fs | vx %.2f yaw %.2f | EE position error %.3f m'%(elapsed,command[0],command[2],error),fill=(15,15,15))
    # Error history over the most recent five seconds.
    gx,gy,gw,gh=350,48,152,72;draw.rectangle((gx,gy,gx+gw,gy+gh),fill=(245,245,245),outline=(60,60,60))
    draw.text((gx+4,gy+2),'EE position error (m)',fill=(25,25,25))
    first=max(0,index-125);hist=t['position_error'][first:index+1,e]
    if len(hist)>1:
        ymax=max(.06,float(np.quantile(hist,.98))*1.2)
        pts=[]
        for j,val in enumerate(hist):
            x=gx+4+(gw-8)*j/(len(hist)-1);y=gy+gh-5-(gh-23)*min(float(val),ymax)/ymax;pts.append((x,y))
        draw.line(pts,fill=(20,95,185),width=2)
        cy=gy+gh-5-(gh-23)*min(error,ymax)/ymax;draw.ellipse((gx+gw-8,cy-3,gx+gw-2,cy+3),fill=(220,50,35))
    return np.asarray(image)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plane',required=True);p.add_argument('--rough',required=True)
    p.add_argument('--command',choices=['stand','forward','turn','forward_turn'],default='forward_turn')
    p.add_argument('--seconds',type=float,default=12);p.add_argument('--start-seconds',type=float,default=0)
    p.add_argument('--stride',type=int,default=2);p.add_argument('--output',required=True)
    a=p.parse_args();env={'stand':0,'forward':1,'turn':2,'forward_turn':3}[a.command]
    states=[load(a.plane,env),load(a.rough,env)]
    n=min(len(s['trace']['step']) for s in states);first=max(0,int(a.start_seconds*25));last=min(n,first+int(a.seconds*25))
    if first>=last:raise ValueError('Requested clip has no frames')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with imageio.get_writer(str(out),fps=25/a.stride,codec='libx264',quality=8,macro_block_size=1) as writer:
        for i in range(first,last,a.stride):
            elapsed=float(states[0]['trace']['step'][i])*.02
            frames=[render(s,i,'PLANE' if j==0 else 'ROUGH',env,len(states),elapsed) for j,s in enumerate(states)]
            joined=np.concatenate(frames,axis=1);writer.append_data(joined)
            if i==first+((last-first)//(2*a.stride))*a.stride:Image.fromarray(joined).save(str(out.with_suffix('.jpg')),quality=94)
            if (i-first)%(a.stride*50)==0:print('rendered %.1fs'%(elapsed-a.start_seconds),flush=True)
    for s in states:s['renderer'].close()
    print('Rendered recorded PhysX policy states; no scripted motion or physics integration.')
if __name__=='__main__':main()
