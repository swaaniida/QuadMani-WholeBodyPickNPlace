"""Geometric preview only; no IK filtering or training configuration changes."""
from pathlib import Path
import json,math,hashlib
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
import geometry_tools as g
O=Path(__file__).resolve().parent;m,d=g.m,g.d
cfg=g.ROOT/'low-level-as2piper-head-orin-deploy/legged_gym/envs/manip_loco/as2_head_orin_config.py'
before=hashlib.sha256(cfg.read_bytes()).hexdigest()
fontpath=g.ROOT/'reports/ee_ground_compare_20261114/viewer/vendor/fonts/NotoSansKR-Medium.ttf'
def text(im,xy,s,size=28,color='#17324a'):
 ImageDraw.Draw(im).text(xy,s,font=ImageFont.truetype(str(fontpath),size),fill=color)
center=np.array([.22435,0,.4563069024246581]);blue=[.06,.34,.91,.8];orange=[.96,.43,.03,.8]
profiles=[dict(name='current',rmax=.65,pmin=-math.asin((center[2]-.02)/.65),pmax=math.asin((1-center[2])/.65),ymax=.3,color=blue),dict(name='candidate',rmax=.80,pmin=-math.asin((center[2]-.02)/.8),pmax=math.asin((1-center[2])/.8),ymax=1.2,color=orange)]
def xyz(r,p,y):return center+r*np.array([np.cos(p)*np.cos(y),np.cos(p)*np.sin(y),np.sin(p)])
m.vis.global_.offwidth=1280;m.vis.global_.offheight=800
m.vis.headlight.ambient[:]=[.55,.55,.55];m.vis.headlight.diffuse[:]=[.65,.65,.65]
assert g.leg_ik(.3244069024246581,0)
d.qpos[g.aq]=g.neutral;mujoco.mj_forward(m,d)
renderer=mujoco.Renderer(m,height=800,width=1280)
def ball(p,col,r=.008):
 gg=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(gg,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([r,0,0]),p,np.eye(3).ravel(),np.array(col,dtype=np.float32));renderer.scene.ngeom+=1

def line(a,b,col,width=.0016):
 gg=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(gg,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(col,dtype=np.float32));mujoco.mjv_connector(gg,mujoco.mjtGeom.mjGEOM_CAPSULE,width,a,b);renderer.scene.ngeom+=1

def poly(points,col):
 for a,b in zip(points[:-1],points[1:]):line(a,b,col)
records=json.loads((g.ROOT/'low-level-as2piper-head-orin-deploy/validation/standing_armbase_reach/verified_poses.json').read_text())
frames=[]
for az,el,dist,look in [(115,-17,2.85,(.35,0,.49)),(90,-80,2.85,(.35,0,.40))]:
 cam=mujoco.MjvCamera();cam.lookat[:]=look;cam.distance=dist;cam.azimuth=az;cam.elevation=el
 renderer.update_scene(d,cam)
 for profile in profiles:
  pmin,pmax,ymax,col=[profile[k] for k in ['pmin','pmax','ymax','color']]
  for r in [.35,profile['rmax']]:
   for p in np.linspace(pmin,pmax,7):poly([xyz(r,p,y) for y in np.linspace(-ymax,ymax,32)],col)
   for y in np.linspace(-ymax,ymax,7):poly([xyz(r,p,y) for p in np.linspace(pmin,pmax,24)],col)
  for p in [pmin,pmax]:
   for y in np.linspace(-ymax,ymax,7):line(xyz(.35,p,y),xyz(profile['rmax'],p,y),col)
  for p in [pmin,0,pmax]:
   ball(xyz(profile['rmax'],p,0),col,.012)
 for rec in records:
  if not rec['passed']:ball(np.array(rec['target']),[.95,.02,.05,1],.017)
 ball(center,[.65,.1,.8,1],.018)
 for axis,col in zip(np.eye(3),[[.85,.1,.1,1],[.1,.7,.1,1],[.1,.2,.9,1]]):line(center,center+.085*axis,col,.003)
 frame=Image.fromarray(renderer.render().copy())
 cc=renderer.scene.camera
 eye=(np.array(cc[0].pos)+np.array(cc[1].pos))/2
 forward=np.array(cc[0].forward);up=np.array(cc[0].up);right=np.cross(forward,up)
 delta=center-eye;depth=np.dot(delta,forward);focal=800/(2*np.tan(np.deg2rad(m.vis.global_.fovy)/2))
 px=640+focal*np.dot(delta,right)/depth;py=400-focal*np.dot(delta,up)/depth
 ImageDraw.Draw(frame).ellipse((px-9,py-9,px+9,py+9),outline='#b323cf',width=4)
 text(frame,(int(px)+14,int(py)-33),'Arm Base',24,'#9d19b7')
 frames.append(frame)
renderer.close()
im=Image.new('RGB',(2560,1120),'white')
text(im,(30,12),'Standing Arm Base 중심 — 확대 후보 도달성 검사',46)
text(im,(30,79),'파랑: 현재 r 0.35~0.65m / 좌우 ±17.2°     주황: 후보 r 0.35~0.80m / VBC 좌우 ±68.8°',31)
text(im,(30,128),'보라색 중심: 실제 standing Arm Base (x=0.22435m, z=0.45631m)  |  목표 높이 0.02~1.00m',29)
for i,frame in enumerate(frames):im.paste(frame,(1280*i,190))
text(im,(35,167),'사선 측면',24);text(im,(1310,167),'위쪽에서',24)
text(im,(30,996),'후보 고도각 −33.1~42.8°  |  267개 중 261개 고정발 통과 + 2개 발 배치 조정 후 통과',30)
text(im,(30,1040),'빨강 4개: 지정 그리퍼 방향까지 만족하는 해 미발견. 방향 제약을 풀면 위치는 모두 도달.',29)
text(im,(30,1080),'평지 정적 IK·관절한계·충돌·CoM 검사. 연속 전영역/동적 추종 보증 아님. 확대 범위는 미적용.',25)
im.save(O/'workspace_reachability.png')
assert hashlib.sha256(cfg.read_bytes()).hexdigest()==before
meta=dict(center=center.tolist(),radius_min=.35,current_radius_max=.65,candidate_radius_max=.8,current_yaw=.3,candidate_yaw=1.2,
 candidate_elevation=[profiles[1]['pmin'],profiles[1]['pmax']],height=[.02,1.],
 radius_only_height_if_elevation_unchanged=[center[2]+.8*math.sin(profiles[0]['pmin']),center[2]+.8*math.sin(profiles[0]['pmax'])],
 candidate_ground_forward_x=center[0]+math.sqrt(.8**2-(center[2]-.02)**2),candidate_upper_forward_x=center[0]+math.sqrt(.8**2-(1-center[2])**2),
 config_hash_before_after=before,IK_checks=0,training_started=False)
(O/'preview.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
