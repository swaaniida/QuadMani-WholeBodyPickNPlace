"""Original B1Z1 VBC sphere range; static visualization, no collision checks."""
from pathlib import Path
import sys,ast,json,hashlib,xml.etree.ElementTree as ET
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent
URDF=Path('/data/team/users/suhwan/vbc-legacy/low-level/resources/robots/b1z1/urdf/b1z1.urdf')
CFG=Path('/data/team/users/suhwan/vbc-legacy/low-level/legged_gym/envs/manip_loco/b1z1_config.py')
XML=R/'mujoco_mount_compare/models/mount_workspace/b1z1.xml'
raw_sha=hashlib.sha256(URDF.read_bytes()).hexdigest();cfgsha=hashlib.sha256(CFG.read_bytes()).hexdigest()
# The cached MJCF came from the exact same URDF. Verify joint-frame origins.
assert raw_sha==hashlib.sha256((R/'low-level/resources/robots/b1z1/urdf/b1z1.urdf').read_bytes()).hexdigest()
urdf=ET.fromstring(URDF.read_text().replace('0.09>>','0.09'));xml=ET.parse(XML).getroot()
checked=0
for joint in urdf.findall('joint'):
 child=joint.find('child').get('link');body=xml.find(".//body[@name='%s']"%child)
 if body is None:raise ValueError('Cached model missing link '+child)
 origin=joint.find('origin');pos=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
 rpy=np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
 assert np.allclose(pos,np.fromstring(body.get('pos','0 0 0'),sep=' '),atol=1e-8)
 q=np.fromstring(body.get('quat','1 0 0 0'),sep=' ')
 assert np.allclose(Rotation.from_euler('xyz',rpy).as_matrix(),Rotation.from_quat(q[[1,2,3,0]]).as_matrix(),atol=1e-7)
 checked+=1
for mesh in xml.findall('./asset/mesh'):mesh.set('file',str((XML.parent/mesh.get('file')).resolve()))
ET.SubElement(xml.find('asset'),'texture',name='sky',type='skybox',builtin='gradient',rgb1='.94 .96 .98',rgb2='.94 .96 .98',width='256',height='256')
xml.find('visual/global').set('offwidth','1280');xml.find('visual/global').set('offheight','800')
ground=xml.find("./worldbody/geom[@name='ground']");ground.set('rgba','.4 .45 .45 0')
(O/'b1z1_visualization.xml').write_text(ET.tostring(xml,encoding='unicode'))
m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));d=mujoco.MjData(m)
tree=ast.parse(CFG.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));init=next(n for n in cls.body if isinstance(n,ast.ClassDef) and n.name=='init_state');angles=ast.literal_eval(next(n.value for n in init.body if isinstance(n,ast.Assign) and n.targets[0].id=='default_joint_angles'))
goal=next(n for n in cls.body if isinstance(n,ast.ClassDef) and n.name=='goal_ee');ranges=next(n for n in goal.body if isinstance(n,ast.ClassDef) and n.name=='ranges');center_class=next(n for n in goal.body if isinstance(n,ast.ClassDef) and n.name=='sphere_center')
def values(body):return {n.targets[0].id:eval(compile(ast.Expression(n.value),'<config>','eval'),{'np':np,'__builtins__':{}}) for n in body if isinstance(n,ast.Assign)}
ranges=values(ranges.body);center_values=values(center_class.body);center=np.array([center_values['x_offset'],center_values['y_offset'],center_values['z_invariant_offset']])
d.qpos[:7]=[0,0,.5,1,0,0,0]
for name,value in angles.items():d.qpos[m.joint(name).qposadr]=value
mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
feet=[m.body(n+'_foot').id for n in ['FL','FR','RL','RR']]
# Ground-seat the default joint pose using the original URDF foot radii.
lowest=[]
for body in feet:
 ids=np.where(m.geom_bodyid==body)[0]
 for gi in ids:
  if m.geom_type[gi]==mujoco.mjtGeom.mjGEOM_SPHERE:lowest.append(d.geom_xpos[gi,2]-m.geom_size[gi,0])
if lowest:d.qpos[2]-=min(lowest)
mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
m.vis.headlight.ambient[:]=[.5,.5,.5];m.vis.headlight.diffuse[:]=[.6,.6,.6]
renderer=mujoco.Renderer(m,height=800,width=1280)
def add(kind,size,pos,rgba):
 geom=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(geom,kind,np.array(size),np.array(pos),np.eye(3).ravel(),np.array(rgba,dtype=np.float32));renderer.scene.ngeom+=1;return geom
def line(a,b,color,width=.002):
 geom=add(mujoco.mjtGeom.mjGEOM_CAPSULE,[0,0,0],[0,0,0],color);mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,width,a,b)
def ball(p,color,size=.007):add(mujoco.mjtGeom.mjGEOM_SPHERE,[size,0,0],p,color)
def cart(r,p,y):return center+r*np.array([np.cos(p)*np.cos(y),np.cos(p)*np.sin(y),np.sin(p)])
def polyline(points,color,width=.0018):
 for a,b in zip(points[:-1],points[1:]):line(a,b,color,width)
rlo,rhi=ranges['pos_l'];plo,phi=ranges['pos_p'];ylo,yhi=ranges['pos_y'];rng=np.random.default_rng(123)
points=[cart(r,p,y) for r,p,y in zip(rng.uniform(rlo,rhi,180),rng.uniform(plo,phi,180),rng.uniform(ylo,yhi,180))]
frames=[]
for az,el in [(112,-14),(110,-67)]:
 cam=mujoco.MjvCamera();cam.lookat[:]=[.40,0,.65];cam.distance=3.75;cam.azimuth=az;cam.elevation=el;renderer.update_scene(d,cam)
 for x in np.arange(-.8,1.81,.2):line(np.array([x,-1.1,0]),np.array([x,1.1,0]),[.55,.6,.65,.45],.001)
 for y in np.arange(-1.,1.01,.2):line(np.array([-.8,y,0]),np.array([1.8,y,0]),[.55,.6,.65,.45],.001)
 for r in [rlo,rhi]:
  for p in np.linspace(plo,phi,9):polyline([cart(r,p,y) for y in np.linspace(ylo,yhi,40)],[.03,.6,.82,.7])
  for y in np.linspace(ylo,yhi,11):polyline([cart(r,p,y) for p in np.linspace(plo,phi,40)],[.03,.6,.82,.7])
 for p in [plo,phi]:
  for y in np.linspace(ylo,yhi,7):line(cart(rlo,p,y),cart(rhi,p,y),[.03,.6,.82,.8])
 for y in [ylo,yhi]:
  for p in np.linspace(plo,phi,7):line(cart(rlo,p,y),cart(rhi,p,y),[.03,.6,.82,.8])
 for p in points:ball(p,[.3,.4,.5,.5],.004)
 ball(center,[.65,.1,.8,1],.022)
 for k,color in enumerate([[.95,.1,.1,1],[.1,.75,.2,1],[.1,.3,.95,1]]):line(center,center+np.eye(3)[k]*.18,color,.003)
 ball(cart(*ranges['init_pos_end']),[1.,.5,.05,1],.018)
 frames.append(Image.fromarray(renderer.render().copy()))
font=R/'reports/ee_ground_compare_20261114/viewer/vendor/fonts/NotoSansKR-Medium.ttf'
im=Image.new('RGB',(2560,1100),'white');draw=ImageDraw.Draw(im)
def text(x,y,s,size=28):draw.text((x,y),s,font=ImageFont.truetype(str(font),size),fill='#18334a')
text(30,12,'원본 B1+Z1 VBC · 기본 구면 목표 생성 범위',46)
text(30,82,'청록: 구면 영역 경계  |  보라: 가상 구면 중심  |  회색: 무작위 생성 예시  |  주황 점: 초기 목표  |  격자: 지면 z=0',28)
text(35,122,'사선 측면',26);text(1315,122,'위쪽에서',26)
for i,frame in enumerate(frames):im.paste(frame,(i*1280,160))
text(30,980,'반경 0.40~0.95m · 위치 고도각 −72°~+60° · 위치 좌우각 ±68.8° · 세 변수를 각각 독립 추출',29)
text(30,1020,'가상 중심: 몸통 XY 기준 전방 0.30m / 지면 높이 0.70m · 이론상 목표 높이 −0.20~1.52m',28)
text(30,1060,'충돌·지면·IK 유효성 검사 없이 원본 생성 영역 전체를 표시. 아래쪽 지면 관통 영역도 그대로 포함.',27)
im.save(O/'b1z1_spherical_range_3d.png');renderer.close()
assert raw_sha==hashlib.sha256(URDF.read_bytes()).hexdigest() and cfgsha==hashlib.sha256(CFG.read_bytes()).hexdigest()
(O/'metadata.json').write_text(json.dumps(dict(urdf=str(URDF),urdf_sha256=raw_sha,config_sha256=cfgsha,model_source=str(XML),joint_frame_origins_verified=checked,center=center.tolist(),ranges=ranges,display_root_height_m=float(d.qpos[2]),range_height_min_m=float(center[2]+rhi*np.sin(plo)),range_height_max_m=float(center[2]+rhi*np.sin(phi)),collision_checks=False,IK_checks=False,physics_steps=0,training=False),indent=2))
print(O/'b1z1_spherical_range_3d.png',flush=True)
