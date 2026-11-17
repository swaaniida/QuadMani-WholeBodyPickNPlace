"""Render measured CPU PhysX rollout; never re-simulate dynamics in MuJoCo."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent;root=O.parents[1];meta=json.loads((O/'metrics.json').read_text());z=np.load(O/'states.npz');eid=1
source=root/'mujoco_mount_compare/models/mount_workspace/b1z1.xml';xml=ET.parse(source).getroot()
for mesh in xml.findall('./asset/mesh'):mesh.set('file',str((source.parent/mesh.get('file')).resolve()))
m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));d=mujoco.MjData(m);m.vis.global_.offwidth=640;m.vis.global_.offheight=420
m.vis.headlight.ambient[:]=[.65,.65,.65];m.vis.headlight.diffuse[:]=[.75,.75,.75]
qids=[int(m.joint(name).qposadr[0]) for name in meta['dof_names']]
fontpath=root/'reports/ee_ground_compare_20261114/viewer/vendor/fonts/NotoSansKR-Medium.ttf'
def text(im,xy,s,size=20,color='#19334b'):ImageDraw.Draw(im).text(xy,s,font=ImageFont.truetype(str(fontpath),size),fill=color)
renderer=mujoco.Renderer(m,height=420,width=640)
def ball(p,color,r):
 gg=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(gg,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([r,0,0]),p,np.eye(3).ravel(),np.array(color,dtype=np.float32));renderer.scene.ngeom+=1
writer=imageio.get_writer(str(O/'b1z1_stand_ee_20s.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=1)
shots=[]
try:
 for i in range(0,len(z['time_s']),2):
  rootpose=z['root_xyzw'][i,eid];d.qpos[:3]=rootpose[:3];d.qpos[3:7]=rootpose[[6,3,4,5]];d.qpos[qids]=z['dof_pos'][i,eid];mujoco.mj_forward(m,d)
  frame=Image.new('RGB',(1280,670),'white')
  text(frame,(16,8),'B1+Z1 원본 45k | 정지 명령 + 다양한 EE 목표 | 측정된 PhysX 상태 재생',28)
  phase=int(z['phase_id'][i]);name=meta['waypoints'][phase][0] if phase>=0 else 'initial pose';err=z['ee_position_error'][i,eid]
  text(frame,(16,48),'t %.2fs  |  vx=vy=yaw 명령 0  |  EE 목표: %s  |  위치 오차 %.1fcm'%(z['time_s'][i],name,err*100),22)
  for k,az in enumerate([115,180]):
   cam=mujoco.MjvCamera();cam.lookat[:]=rootpose[:3]+[.10,0,.20];cam.distance=2.8;cam.azimuth=az;cam.elevation=-19
   renderer.update_scene(d,cam)
   ball(z['ee_target_world'][i,eid],[1,.12,.05,1],.025);ball(z['ee_actual_world'][i,eid],[.05,.85,.25,1],.016)
   for j in range(4):
    bottom=z['foot_positions'][i,eid,j].copy();bottom[2]-=meta['collision_radius_m'][j]
    contact=z['foot_forces'][i,eid,j,2]>2;ball(bottom,[.05,.8,.25,1] if contact else [1,.5,.02,1],.012)
   frame.paste(Image.fromarray(renderer.render().copy()),(k*640,90))
  text(frame,(15,511),'빨강: 목표 EE  /  초록: 실제 EE     발 표식: 초록 Fz>2N / 주황 접촉 부족',21)
  for j,label in enumerate(['FL 왼앞','FR 오른앞','RL 왼뒤','RR 오른뒤']):
   x=15+j*318;fz=z['foot_forces'][i,eid,j,2];gap=z['foot_clearance'][i,eid,j]*1000
   contact=fz>2;col='#168137' if contact else '#bc6500'
   text(frame,(x,548),label,23,col);text(frame,(x,580),'Fz %.1fN / 간격 %.1fmm'%(fz,gap),19,col)
  text(frame,(15,615),'발 간격은 collision 반경40mm 기준. visual 반경30mm로 정상 접촉 중에도 약1cm 떠 보일 수 있음.',19)
  text(frame,(15,643),'4개 초기조건 중 발 들림이 가장 컸던 반복1 표시. DR·외란 없음 / history encoder 사용 / 학습 없음.',18)
  writer.append_data(np.asarray(frame))
  if i in [0,250,500,750,998]:
   frame.save(O/('frame_%04d.png'%i));shots.append(frame)
  if i%250==0:print('render',i,'/',len(z['time_s']),flush=True)
finally:writer.close();renderer.close()
meta['video_env']=eid;meta['video_note']='Selected replica1 has the largest transient foot clearance among four; two fixed views of same measured rollout.';(O/'metrics.json').write_text(json.dumps(meta,indent=2))
sheet=Image.new('RGB',(1280*3,670*2),'white')
for k,frame in enumerate(shots):sheet.paste(frame,((k%3)*1280,(k//3)*670))
sheet.save(O/'contact_sheet.jpg',quality=90)
print('saved',O/'b1z1_stand_ee_20s.mp4',flush=True)
