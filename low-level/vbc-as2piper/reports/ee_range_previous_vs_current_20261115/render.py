"""Actual previous45k vs current position ranges, nominal standing latest URDF."""
from pathlib import Path
import json,math,hashlib,ast
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
import geometry_tools as g
O=Path(__file__).resolve().parent;m,d=g.m,g.d;center=np.array([.22435,0,.4563069024246581])
fontpath=g.ROOT/'reports/ee_ground_compare_20261114/viewer/vendor/fonts/NotoSansKR-Medium.ttf'
paths=[g.URDF,g.ROOT/'low-level-as2piper-head-orin-deploy/legged_gym/envs/manip_loco/as2_head_orin_config.py',g.ROOT/'low-level-as2piper-head-orin-deploy/legged_gym/envs/manip_loco/manip_loco.py']
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def text(im,xy,s,size=28,color='#17324a'):ImageDraw.Draw(im).text(xy,s,font=ImageFont.truetype(str(fontpath),size),fill=color)
profiles=[dict(name='previous45k',rmin=.4,rmax=.48,pmin=.1,pmax=.5,ymax=.25,color=[.05,.32,.96,.95]),dict(name='current',rmin=.35,rmax=.8,pmin=-math.pi/2.5,pmax=math.pi/3,ymax=1.2,color=[.98,.43,.025,.90])]
def xyz(r,p,y):return center+r*np.array([np.cos(p)*np.cos(y),np.cos(p)*np.sin(y),np.sin(p)])
m.vis.global_.offwidth=1280;m.vis.global_.offheight=800
m.vis.headlight.ambient[:]=[.55,.55,.55];m.vis.headlight.diffuse[:]=[.65,.65,.65]
assert g.leg_ik(.3244069024246581,0);d.qpos[g.aq]=g.neutral;mujoco.mj_forward(m,d)
renderer=mujoco.Renderer(m,height=800,width=1280)
def ball(p,col,r=.009):
 gg=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(gg,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([r,0,0]),p,np.eye(3).ravel(),np.array(col,dtype=np.float32));renderer.scene.ngeom+=1

def line(a,b,col,width=.0017):
 a,b=np.array(a,dtype=float),np.array(b,dtype=float)
 if max(a[2],b[2])<.02:return
 if a[2]<.02:a=a+(.02-a[2])/(b[2]-a[2])*(b-a)
 if b[2]<.02:b=b+(.02-b[2])/(a[2]-b[2])*(a-b)
 gg=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(gg,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(col,dtype=np.float32));mujoco.mjv_connector(gg,mujoco.mjtGeom.mjGEOM_CAPSULE,width,a,b);renderer.scene.ngeom+=1

def poly(points,col):
 for a,b in zip(points[:-1],points[1:]):line(a,b,col)
frames=[];zmax=center[2]+.8*np.sin(np.pi/3)
for az,el in [(115,-16),(90,-78)]:
 cam=mujoco.MjvCamera();cam.lookat[:]=[.35,0,.53];cam.distance=2.90;cam.azimuth=az;cam.elevation=el
 renderer.update_scene(d,cam)
 for profile in profiles:
  rmin,rmax,pmin,pmax,ymax,col=[profile[k] for k in ['rmin','rmax','pmin','pmax','ymax','color']]
  for r in [rmin,rmax]:
   for p in np.linspace(pmin,pmax,9):poly([xyz(r,p,y) for y in np.linspace(-ymax,ymax,40)],col)
   for y in np.linspace(-ymax,ymax,9):poly([xyz(r,p,y) for p in np.linspace(pmin,pmax,40)],col)
  for p in [pmin,pmax]:
   for y in np.linspace(-ymax,ymax,7):line(xyz(rmin,p,y),xyz(rmax,p,y),col)
  if profile['name']=='current':
   # Boundary of the original sphere sector cut by the unchanged 2cm clearance.
   threshold=-np.arcsin((center[2]-.02)/rmax)
   for y in np.linspace(-ymax,ymax,9):
    poly([xyz((center[2]-.02)/(-np.sin(p)),p,y) for p in np.linspace(pmin,threshold,35)],col)
   for p in np.linspace(pmin,threshold,6):
    rr=(center[2]-.02)/(-np.sin(p));poly([xyz(rr,p,y) for y in np.linspace(-ymax,ymax,40)],col)
 ball(center,[.65,.1,.8,1],.015);ball(xyz(.8,np.pi/3,0),[.9,.15,.1,1],.015)
 # A marker overlay keeps the true arm-base origin visible through the mesh.
 frame=Image.fromarray(renderer.render().copy());cc=renderer.scene.camera;eye=(np.array(cc[0].pos)+np.array(cc[1].pos))/2
 forward=np.array(cc[0].forward);up=np.array(cc[0].up);right=np.cross(forward,up);focal=800/(2*np.tan(np.deg2rad(m.vis.global_.fovy)/2))
 def project(p):
  v=p-eye;dep=np.dot(v,forward);return 640+focal*np.dot(v,right)/dep,400-focal*np.dot(v,up)/dep
 px,py=project(center);ImageDraw.Draw(frame).ellipse((px-8,py-8,px+8,py+8),outline='#b323cf',width=4);text(frame,(int(px)+12,int(py)-30),'Arm Base',23,'#9d19b7')
 px,py=project(xyz(.8,np.pi/3,0));text(frame,(int(px)-160,int(py)-40),'최대 높이 1.149m',26,'#ba311b')
 frames.append(frame)
renderer.close();im=Image.new('RGB',(2560,1170),'white')
text(im,(30,12),'이전 45k 학습 범위 ↔ 현재 적용 범위',46)
text(im,(30,82),'파랑: 이전 r 0.40~0.48m / 고도각 5.7~28.6° / 좌우 ±14.3°',30)
text(im,(30,125),'주황: 현재 r 0.35~0.80m / VBC 고도각 −72~60° / VBC 좌우 ±68.8°',30)
for i,frame in enumerate(frames):im.paste(frame,(1280*i,205))
text(im,(35,175),'사선 측면',24);text(im,(1310,175),'위쪽에서',24)
text(im,(30,1015),'Standing Arm Base 중심 (x=0.22435, z=0.45631m)  |  최대 높이 = 0.45631 + 0.80 × sin60° = 1.14913m',28)
text(im,(30,1057),'바닥용 pitch 혼합 제거: 이전과 동일한 pitch = 기준각 − 고도각 + 무작위 편차',29)
text(im,(30,1100),'지면 위 2cm 제한을 반영한 위치 영역. 몸통 충돌 필터 전이며, 전체 IK 도달 가능 영역을 뜻하지 않음.',27)
text(im,(30,1138),'이전 범위도 standing 기준으로 표시. 중심은 현재 몸통 높이·roll·pitch에 따라 내려가지 않음. 기존 학습·checkpoint 보존 / 새 학습 없음.',24)
im.save(O/'previous_vs_current.png')
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
meta=dict(center=center.tolist(),previous=profiles[0],current=profiles[1],current_max_height_m=float(zmax),current_raw_min_height_m=float(center[2]+.8*np.sin(-np.pi/2.5)),clearance_min_m=.02,body_box_not_subtracted=True,IK_checked=False,source_hashes=before)
(O/'render_metadata.json').write_text(json.dumps(meta,indent=2));print(O/'previous_vs_current.png')
