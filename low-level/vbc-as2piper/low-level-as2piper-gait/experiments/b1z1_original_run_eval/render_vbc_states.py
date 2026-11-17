"""MuJoCo replay renderer for PhysX states of the original B1Z1 VBC run."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3]
XML=ROOT/'mujoco_mount_compare/models/mount_workspace/b1z1.xml'

def build(folder,width,height):
 folder=Path(folder)
 xml=ET.parse(str(XML)).getroot();asset=xml.find('asset')
 for mesh in asset.findall('mesh'):
  mesh.set('file',str((XML.parent/mesh.get('file')).resolve()))
 rough_obj=next(iter(folder.glob('*_terrain.obj')),None)
 if rough_obj:
  xml.find('worldbody').remove(xml.find("./worldbody/geom[@name='ground']"))
  ET.SubElement(asset,'mesh',name='eval_terrain',file=str(rough_obj.resolve()))
  ET.SubElement(xml.find('worldbody'),'geom',name='eval_terrain',type='mesh',mesh='eval_terrain',
                rgba='.38 .43 .39 1',contype='0',conaffinity='0',group='0')
 target=ET.SubElement(xml.find('worldbody'),'body',name='ee_target_marker',mocap='true')
 ET.SubElement(target,'geom',type='sphere',size='.025',rgba='1 .12 .04 .85',contype='0',conaffinity='0')
 actual=ET.SubElement(xml.find('worldbody'),'body',name='ee_actual_marker',mocap='true')
 ET.SubElement(actual,'geom',type='sphere',size='.016',rgba='.08 1 .15 1',contype='0',conaffinity='0')
 m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));d=mujoco.MjData(m)
 m.vis.headlight.ambient[:]=[.6,.6,.6];m.vis.headlight.diffuse[:]=[.8,.8,.8]
 renderer=mujoco.Renderer(m,height=height,width=width);camera=mujoco.MjvCamera();camera.distance=2.6;camera.azimuth=120;camera.elevation=-18
 ids=[int(m.joint(n).qposadr) for n in json.loads((folder/'dof_names.json').read_text())]
 return m,d,renderer,camera,ids,int(m.body('ee_target_marker').mocapid),int(m.body('ee_actual_marker').mocapid)

def main():
 p=argparse.ArgumentParser();p.add_argument('--states',required=True);p.add_argument('--metrics',required=True);p.add_argument('--output',required=True);p.add_argument('--width',type=int,default=800);p.add_argument('--height',type=int,default=450);p.add_argument('--stride',type=int,default=1);a=p.parse_args()
 z=np.load(a.states);meta=json.load(open(a.metrics));folder=Path(a.states).parent
 (folder/'dof_names.json').write_text(json.dumps(meta['dof_names']))
 m,d,r,cam,jids,target_id,actual_id=build(folder,a.width,a.height)
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);phases=meta['phases'];errors=z['ee_position_error'];plot_max=max(.08,float(np.quantile(errors,.98))*1.15)
 with imageio.get_writer(str(out),fps=25/a.stride,codec='libx264',quality=8,macro_block_size=1) as w:
  for i in range(0,len(z['time_s']),a.stride):
   root=z['root_xyzw'][i];d.qpos[:3]=root[:3];d.qpos[3:7]=root[[6,3,4,5]];d.qpos[jids]=z['dof_pos'][i]
   d.mocap_pos[target_id]=z['ee_target_world'][i];d.mocap_pos[actual_id]=z['ee_actual_world'][i]
   mujoco.mj_forward(m,d);cam.lookat[:]=root[:3]+[.05,0,.22]
   r.update_scene(d,camera=cam);frame=Image.fromarray(r.render().copy());draw=ImageDraw.Draw(frame)
   pid=int(z['phase_id'][i]);phase=phases[pid];cmd=z['command_vxy_yaw'][i];vel=z['base_lin_vel'][i];yaw=z['base_ang_vel'][i,2];err=float(errors[i])
   draw.rectangle((0,0,a.width,43),fill=(248,248,248))
   terrain='ROUGH' if list(folder.glob('*_terrain.obj')) else 'FLAT'
   draw.text((8,5),'B1+Z1 ORIGINAL VBC | 45,000 updates | '+terrain+' | '+phase['name'],fill=(12,12,12))
   draw.text((8,23),'t %.1fs | cmd vx %.2f yaw %.2f | measured vx %.2f yaw %.2f | EE error %.3fm'%(float(z['time_s'][i]),cmd[0],cmd[2],vel[0],yaw,err),fill=(12,12,12))
   gx,gy,gw,gh=a.width-178,52,168,82;draw.rectangle((gx,gy,gx+gw,gy+gh),fill=(246,246,246),outline=(50,50,50));draw.text((gx+5,gy+3),'EE position error (m)',fill=(20,20,20))
   start=max(0,i-124);hist=errors[start:i+1]
   if len(hist)>1:
    pts=[]
    for j,val in enumerate(hist):
     x=gx+5+(gw-10)*j/(len(hist)-1);y=gy+gh-6-(gh-24)*min(float(val),plot_max)/plot_max;pts.append((x,y))
    draw.line(pts,fill=(30,95,185),width=2)
   w.append_data(np.asarray(frame))
   if i%(a.stride*125)==0:print('rendered %.1fs'%(float(z['time_s'][i])),flush=True)
 r.close();print('saved',out,'frames',len(range(0,len(z['time_s']),a.stride)))
if __name__=='__main__':main()
