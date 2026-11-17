"""Latest-URDF static IK and MuJoCo visualization only. No training changes."""
from pathlib import Path
import json,hashlib,itertools,math
import numpy as np,mujoco
from scipy.optimize import least_squares,lsq_linear
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
import geometry_tools as g
O=Path(__file__).resolve().parent;R=g.ROOT;m,d=g.m,g.d
FONT=R/'reports/ee_ground_compare_20261114/viewer/vendor/fonts/NotoSansKR-Medium.ttf'
def font(n):return ImageFont.truetype(str(FONT),n)
def text(im,xy,s,size=28,fill='#18334a'):ImageDraw.Draw(im).text(xy,s,font=font(size),fill=fill)
center=np.array([.22435,0,.4563069024246581]);baseid=m.body('base_link').id
data=json.loads((O/'poses.json').read_text());records=data['poses']
assert data['urdf_sha256']==hashlib.sha256(g.URDF.read_bytes()).hexdigest()
m.vis.global_.offwidth=1280;m.vis.global_.offheight=800
m.vis.headlight.ambient[:]=[.55,.55,.55];m.vis.headlight.diffuse[:]=[.65,.65,.65]
renderer=mujoco.Renderer(m,height=640,width=960)
def scene(az=110,el=-15,distance=1.95,look=(.36,0,.46)):
 cam=mujoco.MjvCamera();cam.lookat[:]=look;cam.distance=distance;cam.azimuth=az;cam.elevation=el
 renderer.update_scene(d,cam)
def add(kind,size,pos,mat,color):
 geom=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(geom,kind,np.array(size),np.array(pos),np.array(mat).ravel(),np.array(color,dtype=np.float32));renderer.scene.ngeom+=1;return geom
def ball(p,color,r=.008):add(mujoco.mjtGeom.mjGEOM_SPHERE,[r,0,0],p,np.eye(3),color)
def line(a,b,color,width=.002):
 geom=add(mujoco.mjtGeom.mjGEOM_CAPSULE,[0,0,0],[0,0,0],np.eye(3),color);mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,width,np.asarray(a),np.asarray(b))
def triad(p,rot,length=.085):
 for k,col in enumerate([[.9,.15,.1,1],[.1,.7,.2,1],[.1,.4,.95,1]]):
  end=p+rot[:,k]*length
  geom=add(mujoco.mjtGeom.mjGEOM_ARROW,[0,0,0],[0,0,0],np.eye(3),col);mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.003,p,end)

tiles=[]
for index,rec in enumerate(records):
 d.qpos[:]=rec['qpos'];mujoco.mj_forward(m,d);scene()
 target=np.array(rec['target_xyz']);rot=Rotation.from_euler('ZYX',[rec['tool_rpy_rad'][2],rec['tool_rpy_rad'][1],0]).as_matrix()
 # Small cross at exact target and a triad showing the desired gripper frame.
 for k in range(3):
  v=np.eye(3)[k]*.018;line(target-v,target+v,[.92,.1,.65,1],.002)
 triad(target,rot);ball(d.xpos[g.tcp],[.1,.85,.75,1],.008)
 if not rec['reached']:line(target,d.xpos[g.tcp],[1.,.2,.05,1],.003)
 for foot in g.feet:ball([*d.xpos[foot,:2],.004],[.15,.75,.3,1],.008)
 frame=Image.fromarray(renderer.render().copy());frame.save(O/f'pose_{index+1}_render.png')
 tile=Image.new('RGB',(960,820),'white');tile.paste(frame,(0,58));text(tile,(20,8),rec['title'],33)
 p=rec['target_xyz'];text(tile,(20,707),f'목표 TCP  ({p[0]:.3f}, {p[1]:+.3f}, {p[2]:.3f}) m  |  그리퍼 pitch {math.degrees(rec["tool_rpy_rad"][1]):.1f}°',25)
 text(tile,(20,745),f'몸통 높이 {rec["body_z_m"]:.3f} m · pitch {rec["body_pitch_deg"]}°  |  위치 오차 {rec["position_error_m"]*1000:.2f} mm',25)
 text(tile,(20,783),('도달 확인 · ' if rec['reached'] else '이번 탐색에서 미도달 · ')+f'방향 오차 {rec["orientation_error_deg"]:.2f}° · CoM 여유 {rec["com_margin_m"]*1000:.0f} mm',24, '#176b4b' if rec['reached'] else '#bd2f24')
 tile.save(O/f'pose_{index+1}.png');tiles.append(tile)
canvas=Image.new('RGB',(2880,1840),'white')
text(canvas,(35,12),'높이 상한 1.0m · 바닥/상단 한계 목표 6개',48)
text(canvas,(35,80),'정적 Leg IK + Arm 6D IK  |  분홍 십자: 목표 TCP · 청록 점: 실제 TCP · RGB: 그리퍼 목표 축 · 초록 점: 네 발 기준',29)
for i,tile in enumerate(tiles):canvas.paste(tile,((i%3)*960,135+(i//3)*820))
text(canvas,(35,1790),'실제 생성기 범위·명목 방향식 기준. 정적 IK 자세이며 학습 정책의 동작 아님. 미도달은 전역 불가능 판정이 아님.',29)
canvas.save(O/'extreme_poses_2x3.png')
renderer.close()


renderer=mujoco.Renderer(m,height=800,width=1280)
assert g.leg_ik(.3244069024246581,0);d.qpos[g.aq]=g.neutral;mujoco.mj_forward(m,d)
def cart(r,phi,yaw):return center+r*np.array([np.cos(phi)*np.cos(yaw),np.cos(phi)*np.sin(yaw),np.sin(phi)])
def expanded(x,z,yaw):return np.array([center[0]+(x-center[0])*np.cos(yaw),(x-center[0])*np.sin(yaw),z])
def polyline(points,color,width=.0018):
 for a,b in zip(points[:-1],points[1:]):line(a,b,color,width)
rng=np.random.default_rng(123);phi=rng.uniform(-1.0934979298811465,1.177398919915009,220);yaw=rng.uniform(-.3,.3,220)
low=.22565/np.cos(phi);upper=np.minimum(.75065/np.cos(phi),.926864395071399)
vertical=np.where(phi>=0,(1.-center[2])/np.maximum(np.sin(phi),1e-8),(center[2]-.02)/np.maximum(-np.sin(phi),1e-8));upper=np.minimum(upper,vertical);radius=low+rng.random(len(phi))*np.maximum(upper-low,0)
points=[cart(r,p,y) for r,p,y in zip(radius,phi,yaw)];frames=[]
for az,el in [(112,-14),(105,-65)]:
 scene(az,el,2.50,(.40,0,.50));blue=[.08,.38,.98,.85];amber=[.95,.52,.04,.85];purple=[.65,.17,.85,.85]
 for r in [.40,.48]:
  for phi0 in np.linspace(.10,.50,6):polyline([cart(r,phi0,y) for y in np.linspace(-.25,.25,20)],blue)
  for y in np.linspace(-.25,.25,6):polyline([cart(r,p,y) for p in np.linspace(.10,.50,20)],blue)
 for p in [.10,.50]:
  for y in [-.25,.25]:line(cart(.40,p,y),cart(.48,p,y),blue)
 for z in [.02,1.00]:
  for x in [.45,.975]:polyline([expanded(x,z,y) for y in np.linspace(-.3,.3,35)],amber)
  for y in [-.3,.3]:polyline([expanded(x,z,y) for x in np.linspace(.45,.975,20)],amber)
 for x in [.45,.975]:
  for y in [-.3,.3]:line(expanded(x,.02,y),expanded(x,1.,y),amber)
 for point in points:ball(point,[.32,.42,.50,.6],.0035)
 ball(center,[.65,.15,.8,1],.011)
 for rec in records:ball(rec['target_xyz'],[.05,.75,.35,1] if rec['reached'] else [.95,.12,.12,1],.012)
 frames.append(Image.fromarray(renderer.render().copy()))
comparison=Image.new('RGB',(2560,1100),'white')
text(comparison,(30,12),'이전 학습 범위 ↔ 현재 생성 범위',46)
text(comparison,(30,82),'파랑: 이전 학습 범위  |  주황: 현재 생성 범위 (지면 위 0.02~1.00m)  |  회색 점: 연속 생성 예시',28)
for i,frame in enumerate(frames):comparison.paste(frame,(i*1280,160))
text(comparison,(35,122),'사선 측면',26);text(comparison,(1315,122),'위쪽에서',26)
text(comparison,(30,980),'이전: 반경 0.40~0.48m · 고도각 0.10~0.50rad · 좌우각 ±0.25rad  →  현재: 전방 x 0.45~0.975m · 높이 0.02~1.00m · 좌우각 ±0.30rad',28)
text(comparison,(30,1020),'강조한 목표 6개: 초록은 도달 자세 확인, 빨강은 이번 유한 자세 탐색에서 미도달. 전역 도달 불가 판정 아님.',28)
text(comparison,(30,1060),'범위 내부를 IK로 제한하지 않음. 학습 시 연속 생성 + 단순 충돌 박스/지면·높이 검사, bank 없음.',26)
comparison.save(O/'workspace_comparison_3d.png');renderer.close()
print('SAVED',O/'extreme_poses_2x3.png',O/'workspace_comparison_3d.png',flush=True)
