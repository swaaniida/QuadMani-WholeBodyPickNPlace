"""Static FK illustrations of goal-frame invariance; no physics or training."""
from pathlib import Path
import json,ast,types
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).parent
font=FontProperties(fname=str(ROOT/'reports/ee_ground_compare_20261114/viewer/vendor/fonts/NotoSansKR-Medium.ttf'))
plt.rcParams['font.family']=font.get_name()
from matplotlib import font_manager
font_manager.fontManager.addfont(font.get_file())
xml=ROOT/'reports/deployment_workspace_review_20261114/analysis_model.xml'
m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m)
m.vis.global_.offwidth=640;m.vis.global_.offheight=480
r=mujoco.Renderer(m,height=480,width=640)
poses=json.loads((xml.parent/'workspace_scan.json').read_text())['postures']
chosen=[next(p for p in poses if abs(p['base_z_m']-h)<1e-8 and p['tilt_deg']==a) for h,a in [(.3244069024246581,0),(.27,0),(.27,30)]]
center=np.array([.22435,0,.4563069024246581]);target=np.array([.75,0,.02])
colors=[np.array([.94,.16,.12,1]),np.array([.12,.7,.25,1]),np.array([.15,.4,.95,1])]
records=[]
fig,axes=plt.subplots(1,3,figsize=(18,6.6),facecolor='white')
fig.subplots_adjust(left=.015,right=.985,top=.82,bottom=.18,wspace=.025)
fig.suptitle('몸통이 낮아져도 EE 가상 목표 중심은 같은 높이를 유지',fontsize=21,y=.98)
fig.text(.5,.905,'보라색: 가상 중심  |  주황색: 실제 Arm Base  |  하늘색: 동일 바닥 목표  |  축: X 빨강 · Y 초록 · Z 파랑',ha='center',fontsize=12)
for ax,p,title in zip(axes,chosen,['Standing · 직립','Crouching · 수평으로 낮춤','Front Crouching · 앞다리를 굽혀 30° 숙임']):
 d.qpos[:]=0;d.qpos[:3]=[0,0,p['base_z_m']]
 a=np.deg2rad(p['tilt_deg']);d.qpos[3:7]=[np.cos(a/2),0,np.sin(a/2),0]
 for leg,angles in zip(['FL','FR','RL','RR'],np.array(p['leg_q']).reshape(4,3)):
  for joint,value in zip(['hip','thigh','calf'],angles):d.qpos[m.joint(leg+'_'+joint+'_joint').qposadr]=value
 for i,value in enumerate([0,1.1,-.55,0,0,0],1):d.qpos[m.joint('piper_joint'+str(i)).qposadr]=value
 for name,value in [('piper_gripper_joint1',.049),('piper_gripper_joint2',-.049)]:d.qpos[m.joint(name).qposadr]=value
 mujoco.mj_forward(m,d)
 bid=m.body('piper_base_link').id;actual=d.xpos[bid].copy();rot=d.xmat[bid].reshape(3,3).copy()
 cam=mujoco.MjvCamera();cam.lookat[:]=[.32,0,.39];cam.distance=1.72;cam.azimuth=100;cam.elevation=-12
 r.update_scene(d,cam)
 def sphere(pos,color,size):
  g=r.scene.geoms[r.scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([size,0,0]),pos,np.eye(3).flatten(),np.array(color,dtype=np.float32));r.scene.ngeom+=1
 def arrow(start,end,color,width=.006):
  g=r.scene.geoms[r.scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).flatten(),np.array(color,dtype=np.float32));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,width,start,end);r.scene.ngeom+=1
 # Render axis origins just toward the camera to avoid the mesh occluding them.
 display_shift=np.array([0,-.20,0])
 for origin,rotation,length,color in [(actual,rot,.105,[1,.48,.05,1]),(center,np.eye(3),.16,[.68,.12,.88,1])]:
  o=origin+display_shift
  for k in range(3):arrow(o,o+rotation[:,k]*length,colors[k])
  sphere(o,color,.014)
 sphere(target,[0,.85,.95,1],.017)
 # Horizontal dashed reference at virtual-center height.
 for x in np.arange(-.25,.95,.045):
  arrow(np.array([x,-.215,center[2]]),np.array([x+.022,-.215,center[2]]),[.68,.12,.88,.8],.003)
 image=r.render().copy();ax.imshow(image);ax.axis('off');ax.set_title(title,fontsize=13,pad=10)
 ax.text(.5,-.035,f"실제 Arm Base: x={actual[0]:.3f}, z={actual[2]:.3f}m\n가상 중심: x={center[0]:.3f}, z={center[2]:.3f}m · 유지",transform=ax.transAxes,ha='center',va='top',fontsize=12,linespacing=1.6)
 records.append(dict(posture=title,base_height_m=p['base_z_m'],pitch_deg=p['tilt_deg'],actual_arm_base=actual.tolist(),virtual_center=center.tolist(),floor_target=target.tolist()))
fig.text(.5,.035,'현재 카메라 포함 모델 · 같은 시점 · 발 위치 유지 Leg IK · 정적 FK 비교\n축은 가림 방지를 위해 화면 앞쪽(y −0.20m)에 표시. 바닥 점은 좌표 유지 확인용이며, 도달 가능성·물리 안정성 검증은 별도.',ha='center',fontsize=10,color='#555555')
fig.savefig(OUT/'standing_crouching_frames.png',dpi=145)
(OUT/'render_frames.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
r.close();print(OUT/'standing_crouching_frames.png')
