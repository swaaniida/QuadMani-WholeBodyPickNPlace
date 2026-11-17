"""Visualize EXACT legacy EE point-rejection box; no config/physics edits."""
from pathlib import Path
import ast,json,itertools,hashlib
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import geometry_tools as g
O=Path(__file__).resolve().parent;R=g.ROOT;m,d=g.m,g.d
font_manager.fontManager.addfont(str(R/'reports/ee_ground_compare_20261114/viewer/vendor/fonts/NotoSansKR-Medium.ttf'))
plt.rcParams['font.family']='Noto Sans KR';plt.rcParams['axes.unicode_minus']=False
cfg=R/'low-level-as2piper-head-orin-deploy/legged_gym/envs/manip_loco/as2_head_orin_config.py'
root=ast.parse(cfg.read_text());klass=next(n for n in root.body if isinstance(n,ast.ClassDef) and n.name=='AS2HeadOrinCfg')
vals={}
for cls in klass.body:
 if isinstance(cls,ast.ClassDef) and cls.name in ('asset','goal_ee'):
  for n in cls.body:
   if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('arm_base_offset','collision_lower_limits','collision_upper_limits'):
    vals[n.targets[0].id]=ast.literal_eval(n.value)
lo=np.array(vals['collision_lower_limits']);hi=np.array(vals['collision_upper_limits']);mount=np.array(vals['arm_base_offset'])
corners=np.array(list(itertools.product(*zip(lo,hi))));edges=[(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(corners[i]!=corners[j])==1]
m.vis.global_.offwidth=760;m.vis.global_.offheight=540
renderer=mujoco.Renderer(m,height=540,width=760)
fig,axes=plt.subplots(2,3,figsize=(17,9.7));fig.subplots_adjust(left=.01,right=.99,bottom=.105,top=.875,wspace=.015,hspace=.13)
fig.suptitle('현재 EE 목표 충돌 박스 · 최신 카메라 URDF',fontsize=23,y=.985)
fig.text(.5,.94,'주황색: 목표 경로 배제 박스   |   로봇의 실제 충돌 형상과는 별도   |   박스는 몸통 회전·높이를 따라 이동',ha='center',fontsize=13)
records=[]
for row,(height,angle,label) in enumerate([(.3244069024246581,0,'Standing'),(.27,30,'전방 Crouching · 30°')]):
 assert g.leg_ik(height,np.deg2rad(angle));d.qpos[g.aq]=g.neutral;g.fk();mujoco.mj_forward(m,d)
 bid=m.body('base_link').id;rot=d.xmat[bid].reshape(3,3);base=d.xpos[bid].copy()
 world=base+(corners+mount)@rot.T
 for col,(az,el,name) in enumerate([(90,-9,'측면'),(0,-8,'정면'),(90,-88,'위에서')]):
  cam=mujoco.MjvCamera();cam.lookat[:]=[.12,0,.32 if row==0 else .27];cam.distance=1.5;cam.azimuth=az;cam.elevation=el
  renderer.update_scene(d,cam)
  def add(kind,size,pos,mat,color):
   geom=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(geom,kind,np.array(size),np.array(pos),np.array(mat).flatten(),np.array(color,dtype=np.float32));renderer.scene.ngeom+=1;return geom
  add(mujoco.mjtGeom.mjGEOM_BOX,(hi-lo)/2,base+rot@(mount+(hi+lo)/2),rot,[1,.36,.04,.14])
  for i,j in edges:
   geom=add(mujoco.mjtGeom.mjGEOM_CAPSULE,[0,0,0],[0,0,0],np.eye(3),[1,.30,.015,1])
   mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.0025,world[i],world[j])
  ax=axes[row,col];ax.imshow(renderer.render().copy());ax.axis('off');ax.set_title(f'{label} | {name}',fontsize=13,pad=4)
 # Orin collision box corners in base coordinates, compared with rejection box.
 orin=[]
 for gi in range(m.ngeom):
  if m.geom(gi).name.startswith('c_jetson_agx_orin64_link'):
   half=m.geom_size[gi];pts=np.array(list(itertools.product(*zip(-half,half))))
   pts=d.geom_xpos[gi]+pts@d.geom_xmat[gi].reshape(3,3).T
   local=(pts-base)@rot
   orin.append(dict(base_min=local.min(0).tolist(),base_max=local.max(0).tolist(),inside_vertices=int(np.all((local>=mount+lo)&(local<=mount+hi),axis=1).sum())))
 records.append(dict(height_m=height,pitch_deg=angle,world_box_corners=world.tolist(),orin=orin))
fig.text(.5,.05,f'Arm Base 상대 범위(m): X [{lo[0]:.5f}, {hi[0]:.5f}] · Y [{lo[1]:.3f}, {hi[1]:.3f}] · Z [{lo[2]:.4f}, {hi[2]:.4f}]\n몸통 좌표 범위(m): X [{(mount+lo)[0]:.3f}, {(mount+hi)[0]:.3f}] · Y ±{hi[1]:.3f} · Z [{(mount+lo)[2]:.3f}, {(mount+hi)[2]:.4f}]',ha='center',fontsize=12)
fig.savefig(O/'collision_box_standing_crouching.png',dpi=145)
renderer.close()
(O/'measurements.json').write_text(json.dumps(dict(urdf_sha256=hashlib.sha256(g.URDF.read_bytes()).hexdigest(),configured=vals,box_base_min=(mount+lo).tolist(),box_base_max=(mount+hi).tolist(),poses=records,note='EE point path rejection only. Current deploy continuous sampler uses this legacy EE point-path box. Static FK, no training.'),indent=2,ensure_ascii=False))
print(O/'collision_box_standing_crouching.png')
