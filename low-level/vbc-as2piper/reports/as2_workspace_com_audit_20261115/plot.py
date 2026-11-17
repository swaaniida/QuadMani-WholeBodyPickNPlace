from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.spatial import ConvexHull
O=Path(__file__).resolve().parent;root=O.parents[1]
font_manager.fontManager.addfont(str(root/'reports/ee_ground_compare_20261114/viewer/vendor/fonts/NotoSansKR-Medium.ttf'));plt.rcParams.update({'font.family':'Noto Sans KR','axes.unicode_minus':False,'font.size':12})
a=json.loads((O/'scan.json').read_text());s=json.loads((O/'summary.json').read_text());rec=a['records'];sols=[q for x in rec for q in x['solutions']];force=a['weight_N']
fig=plt.figure(figsize=(17,11),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1]);ax=fig.add_subplot(gs[0,0]);bx=fig.add_subplot(gs[0,1]);cx=fig.add_subplot(gs[1,:])
good=[x for x in rec if x['solutions']];bad=[x for x in rec if not x['solutions']];pos=np.array([x['target'] for x in good]);color=np.array([min(q['rear_total_min_N'] for q in x['solutions'])/force*100 for x in good])
im=ax.scatter(pos[:,0],pos[:,2],c=color,cmap='viridis',s=35,vmin=15,vmax=50)
if bad:
 p=np.array([x['target'] for x in bad]);ax.scatter(p[:,0],p[:,2],marker='x',c='#8b8b8b',s=45,label='IK/충돌 검사 미통과 — CoM 판정 제외');ax.legend(fontsize=10)
ax.set(xlabel='목표 전방 x (m)',ylabel='목표 지면 높이 z (m)',title='현재 생성 목표별 후방 하중: 찾은 자세 중 최솟값');fig.colorbar(im,ax=ax,label='정적 뒷다리 합산 하중 하한 / 총중량 (%)');ax.grid(alpha=.2)
feet=np.array(a['baseline']['foot_positions'])[:,:2];h=ConvexHull(feet);closed=np.r_[h.vertices,h.vertices[0]];bx.fill(feet[closed,0],feet[closed,1],color='#e6f0f7');bx.plot(feet[closed,0],feet[closed,1],color='#3b6d8c')
com=np.array([q['com_xyz'] for q in sols]);bx.scatter(com[:,0],com[:,1],s=10,c='#2c9475',alpha=.35,label='전신 IK 자세의 CoM 투영')
for name,p in zip(['FL','FR','RL','RR'],feet):bx.plot(*p,'ko');bx.annotate(name,p,xytext=(5,5),textcoords='offset points')
w=np.array(s['worst_pose']['com_xyz']);bx.plot(w[0],w[1],'r*',ms=15,label='전방 여유 최소 자세')
bx.set(xlabel='전방 x (m)',ylabel='좌우 y (m)',title='네 발 지지 다각형과 CoM 투영');bx.set_aspect('equal');bx.grid(alpha=.2);bx.legend(fontsize=10)
old=np.load(O/'recorded_com.npz');source=root/'reports/deployment_workspace_review_20261114/contact_capture';physics=np.load(source/'foot_physics.npz')['values'][1::2];mask=(old['time']>=12)&(old['time']<=60);t=old['time'][mask];j=3
cx.plot(t,old['front_margin'][mask,j]*100,color='#1762a2',lw=1.8,label='CoM → 전방 지지선 여유 (cm)')
cx.plot(t,old['clearance'][mask,j,3]*100,color='#e18023',lw=1.8,label='오른뒷발 지면 간격 (cm)')
dx=cx.twinx();fz=physics[mask,24,12:24].reshape(-1,4,3)[:,:,2];dx.plot(t,fz[:,2:].sum(-1),color='#2a9859',alpha=.75,lw=1.2,label='뒷다리 합산 실제 Fz (N)');dx.plot(t,fz[:,3],color='#ba4242',lw=1,label='오른뒷발 실제 Fz (N)')
cx.set(xlabel='평가 시간 (s)',ylabel='거리 (cm)',title='별도 근거: 기존 AS2 45k 로그 — RR은 들리지만 CoM은 전방 지지선 뒤에 있음',ylim=(-1,18));dx.set(ylabel='수직 접촉력 (N)',ylim=(-5,115));cx.grid(alpha=.2)
h1,l1=cx.get_legend_handles_labels();h2,l2=dx.get_legend_handles_labels();cx.legend(h1+h2,l1+l2,loc='upper right',ncol=2,fontsize=10)
fig.suptitle('AS2+PiPER EE 작업범위: 전방 CoM 쏠림·뒷발 하중 검사\n현재 URDF 정적 검사와 기존 학습 정책의 실제 접촉 로그를 구분',fontsize=19)
fig.savefig(O/'com_and_rear_support.png',dpi=150);fig.savefig(O/'com_and_rear_support.pdf');plt.close(fig)
print('saved',O/'com_and_rear_support.png')
