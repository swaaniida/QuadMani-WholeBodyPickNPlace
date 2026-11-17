from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
O=Path(__file__).parent;ROOT=O.resolve().parents[2]
font=ROOT/'reports/ee_ground_compare_20261114/viewer/vendor/fonts/NotoSansKR-Medium.ttf'
font_manager.fontManager.addfont(str(font));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False})
a=json.loads((O/'candidate_inspection.json').read_text());b=json.loads((O/'old_orientation_inspection.json').read_text())
valid=[r for r in a['samples'] if r['path_pass']]
by_height=[]
for z in sorted({r['z'] for r in a['samples']}):
 group=[r for r in valid if r['z']==z];old=[r for r in b['samples'] if r['z']==z and r['path_pass']]
 by_height.append(dict(z=z,candidate=len(group),old_orientation=len(old),candidate_min_nominal_x=min((r['nominal_x'] for r in group),default=None),candidate_max_nominal_x=max((r['nominal_x'] for r in group),default=None)))
proposal=dict(status='inspection_only_pending_user_review',sampler_modified=False,full_training=False,
 sample_bank='Offline position+orientation+whole-body witness bank; never sample rectangular r/phi/yaw bounding box blindly.',
 orientation='Keep initial FK-matching high-target law. Blend into downward tool pitch 175 to 150 degrees below 0.25m, based on forward reach. This is candidate only.',
 transition='Offline adjacency screening of actual sphere interpolation for every allowed target pair. Home-star paths tested here; arbitrary target-to-target transitions NOT tested.',
 tolerance=dict(position_m=.003,orientation_deg=2,joint_reserve_deg=5,com_margin_m=.02,penetration_reject_m=.001),
 unverified=['PiPER continuous/peak torque and thermal limits','path dynamics/inertial torque','payload/grasp/object collision','PhysX learned policy feasibility','rough local target ground and terrain-relative collision','orientation perturbations and roll diversity beyond tested zero-roll poses'],
 geometry_bounds_do_not_define_uniform_sampling={k:[min(r[k] for r in valid),max(r[k] for r in valid)] for k in ['r_m','phi_rad','yaw_rad']},
 counts=dict(tested=90,old_orientation=60,candidate=69),by_height=by_height,
 max_abs_arm_gravity_tau_Nm=np.max([r['witness']['path']['max_arm_gravity_tau_Nm'] for r in valid],axis=0).tolist(),
 source_sha256=a['urdf_sha256'],nominal_mass_kg=a['mass_kg'])
(O/'proposal.json').write_text(json.dumps(proposal,indent=2,ensure_ascii=False))
fig=plt.figure(figsize=(13,5.5),layout='constrained')
ax=fig.add_subplot(121,projection='3d')
for passed,color,label in [(True,'#258d72','해·경로 찾음 (69)'),(False,'#b9bec6','이번 검사에서 해 못 찾음 (21)')]:
 pts=np.array([r['target'] for r in a['samples'] if r['path_pass']==passed]);ax.scatter(*pts.T,c=color,s=28,label=label)
ax.scatter(*np.array(a['center']),c='#a745cc',marker='*',s=120,label='가상 중심')
ax.set(xlabel='전방 X (m)',ylabel='좌우 Y (m)',zlabel='지면 기준 Z (m)',title='카메라 포함 모델 · 전신 IK + 충돌·관절·정적 지지 검사')
ax.view_init(elev=19,azim=-65);ax.legend(loc='upper left',fontsize=8)
ax2=fig.add_subplot(122)
for data,style,color,label in [(b,'--','#8b8d95','기존 방향식'),(a,'-','#238c72','위치와 연동한 방향 후보')]:
 rows=[r for r in data['samples'] if r['yaw_rad']==0 and r['path_pass']]
 ax2.scatter([r['nominal_x'] for r in rows],[r['z'] for r in rows],s=140 if data is b else 45,facecolors='none' if data is b else color,edgecolors=color,label=label)
ax2.set(xlabel='전방 X (m)',ylabel='목표 TCP 높이 (m)',title='같은 목표 90개에서 방향식만 비교\n바닥 0.02m: 전방 0.75 → 0.975m 표본까지 해를 찾음')
ax2.legend(fontsize=9);ax2.grid(alpha=.2)
fig.suptitle('2단계 사전 검사 — 학습 샘플러 미변경\n토크는 중력 필요량·URDF 한계 비교이며, 공식 PiPER 구동 한계 검증값이 아님',fontsize=13)
fig.savefig(O/'workspace_inspection.png',dpi=160)
print(json.dumps(proposal,indent=2,ensure_ascii=False))
