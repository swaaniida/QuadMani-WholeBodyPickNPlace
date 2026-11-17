"""Summarize computed workspace/stance diagnosis; no production edits."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
clone=ROOT/'low-level-as2piper-head-orin-nocontact'
font=ROOT/'reports/ee_ground_compare_20261114/viewer/vendor/fonts/NotoSansKR-Medium.ttf'
font_manager.fontManager.addfont(str(font));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False})
contacts=json.loads((OUT/'contact_analysis.json').read_text());workspace=json.loads((OUT/'workspace_scan.json').read_text());envelope=json.loads((OUT/'virtual_envelope.json').read_text());approach=json.loads((OUT/'approach_angle_scan.json').read_text());refined=json.loads((OUT/'refined_tilt.json').read_text())
anchor=np.array([.22435,0,.4563069024246581]);ground=.02
math_rows=[]
for x in [.45,.60,.75,.85,.90,.925,.95,1.0]:
    v=np.array([x,0,ground])-anchor;r=np.linalg.norm(v);phi=np.arctan2(v[2],v[0]);beta=2.455216137744035-phi
    math_rows.append({'target_x_from_base_m':x,'target_z_m':ground,'virtual_r_m':float(r),'virtual_pitch_rad':float(phi),'virtual_pitch_deg':float(np.rad2deg(phi)),'old_orientation_law_beta_deg':float(np.rad2deg(beta)),'delta_to_tool160_rad':float(np.deg2rad(160)-beta),'delta_to_tool175_rad':float(np.deg2rad(175)-beta)})
b1_far=.3+np.sqrt(.95**2-(.7-ground)**2)
old_arm_z=.3244069024246581+.1319
tilt=np.deg2rad(25);new_arm_z=.3244069024246581+.1319*np.cos(tilt)-.22435*np.sin(tilt)
example={'standing_arm_z_m':old_arm_z,'tilt25_same_baseheight_arm_z_m':float(new_arm_z),'goal_z_if_center_follows_tilt_after_sampling_floor2cm':float(.02+new_arm_z-old_arm_z)}
calculations={'virtual_anchor_world_relative_root_xy':anchor.tolist(),'ground_targets':math_rows,'b1_original_sampled_far_x_z2cm_m':float(b1_far),'as2_r085_sampled_far_x_z2cm_m':float(.22435+np.sqrt(.85**2-(anchor[2]-.02)**2)),'as2_r090_sampled_far_x_z2cm_m':float(.22435+np.sqrt(.9**2-(anchor[2]-.02)**2)),'moving_center_failure_example':example,'note':'B1 values are original command envelope, not verified realized reach; AS2 static IK/collision results separately recorded.'}
(OUT/'geometry_calculations.json').write_text(json.dumps(calculations,indent=2)+'\n')

asset='resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf';manifest=json.loads((clone/'experiments/contact_force_ablation/runs/as2piper_headpiper_orin_200N_0.18m/manifest.json').read_text())
sources={k:hashlib.sha256((clone/k).read_bytes()).hexdigest() for k in manifest['source_sha256']};assert all(v==sources[k] for k,v in manifest['source_sha256'].items() if k!=asset)
new_tree=ET.parse(clone/asset).getroot();old_tree=ET.parse(OUT/'trained_asset/urdf/as2piper_head_orin_training.urdf').getroot()
def canonical(e):return [e.tag,sorted(e.attrib.items()),(e.text or '').strip(),[canonical(c) for c in e]]
for tag in ['link','joint']:
    newer={e.get('name'):e for e in new_tree.findall(tag)}
    for e in old_tree.findall(tag):assert canonical(e)==canonical(newer[e.get('name')]),e.get('name')
proof={'no_production_code_edits':True,'production_sources_except_preexisting_camera_asset_match_training_manifest':True,'camera_asset_preexisting_new_hash':sources[asset],'training_asset_frozen_hash':hashlib.sha256((OUT/'trained_asset/urdf/as2piper_head_orin_training.urdf').read_bytes()).hexdigest(),'camera_append_preserves_all_original_links_and_joints':True,'training_nominal_mass_kg':23.328064081329995,'current_camera_nominal_mass_kg':workspace['mass_kg'],'camera_added_nominal_kg':workspace['mass_kg']-23.328064081329995,'optimizer_updates':0}
assert proof['training_asset_frozen_hash']==manifest['source_sha256'][asset]
(OUT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')

z=np.load(OUT/'contact_capture/foot_physics.npz');a=z['values'];t=(np.arange(len(a))+1)*float(z['dt']);xyz=a[:,:,:12].reshape(len(a),32,4,3);F=a[:,:,12:24].reshape(len(a),32,4,3)
fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
for col,(it,j) in enumerate([(28000,1),(45000,3)]):
    h=(xyz[:,j*8:(j+1)*8,3,2]-.028)*1000
    axs[0,col].plot(t,h.mean(1),color='#ba4a35',label='오른 뒷발 간격 평균')
    axs[0,col].fill_between(t,h.min(1),h.max(1),color='#ba4a35',alpha=.16,label='8개 환경 범위')
    axs[0,col].axhline(5,color='#777',ls='--',label='5mm 진단선');axs[0,col].set_title(f'{it:,} | 정지 EE 궤적 중 오른 뒷발');axs[0,col].set_ylabel('발바닥–지면 간격 (mm)');axs[0,col].legend(fontsize=8)
    for k,name in enumerate(['FL 앞왼발','FR 앞오른발','RL 뒷왼발','RR 뒷오른발']):axs[1,col].plot(t,F[:,j*8:(j+1)*8,k,2].mean(1),label=name,lw=1.1)
    axs[1,col].set_ylabel('발별 평균 수직력 (N)');axs[1,col].set_xlabel('시간 (s)');axs[1,col].legend(fontsize=8,ncol=2)
    for ax in axs[:,col]:ax.axvspan(12,60,color='#dddddd',alpha=.17);ax.grid(alpha=.15)
fig.suptitle('세 발 지지 현상 확인 | base vx·vy·yaw 명령 모두0\n실제 PhysX 힘 + 발 충돌 구 간격 · 학습 당시 URDF · 8개 환경 · 회색: 바닥 EE 구간',fontsize=14)
fig.savefig(OUT/'rr_contact_diagnosis.png',dpi=170);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
for h,color in [(.3244069024246581,'#566b89'),(.30,'#25856b'),(.27,'#b87535')]:
    for mode,style in [('down_160deg','-'),('down_175deg','--')]:
        rows=[r for r in workspace['summary'] if r['base_z_m']==h and r['orientation_mode']==mode]
        xs=[r['tilt_deg'] for r in rows];ys=[r['max_x_geometry_margin20mm'] if r['max_x_geometry_margin20mm'] is not None else np.nan for r in rows]
        axs[0].plot(xs,ys,style,marker='o',color=color,label=f'몸통 높이{h:.3f}m / 그리퍼{160 if mode=="down_160deg" else 175}°')
axs[0].axhline(b1_far,color='#a23b35',ls=':',label='B1 원본 명령 범위 최대 (실제 도달 검증값 아님)')
axs[0].set_xlabel('전방으로 숙인 몸통 pitch (°)');axs[0].set_ylabel('바닥 TCP 도달 x / 몸통 원점 기준 (m)');axs[0].set_title('네 발 접지 자세 + 팔 6D IK + 충돌·CoM 표본 검사');axs[0].legend(fontsize=8);axs[0].grid(alpha=.15)
for kind,color,label in [('valid','#27856c','검사 자세 중 해를 찾음'),('invalid','#ba5145','검사 자세에서 해 못 찾음'),('ground','#999999','목표가 지면 아래')]:
    sample=[r for r in envelope['samples'] if r['yaw']==0 and ('ground' if r['below_ground'] else 'valid' if r['valid'] else 'invalid')==kind]
    axs[1].scatter([r['r'] for r in sample],np.rad2deg([r['phi'] for r in sample]),c=color,s=35,label=label)
axs[1].set_xlabel('지면 기준 가상 구면 반경 r (m)');axs[1].set_ylabel('위치 고도각 φ (°)');axs[1].set_title('단일 r·pitch 사각 범위 전체가 유효하지 않음 (yaw0)');axs[1].legend(fontsize=8);axs[1].grid(alpha=.15)
fig.suptitle('AS2 머리 PiPER + Orin + 현재 D435i | 바닥 목표 TCP z=2cm\n유한 표본 기구학·충돌·정적 CoM 검사이며 동적 안정성·실기 배포 인증이 아님',fontsize=14)
fig.savefig(OUT/'ground_workspace.png',dpi=170);plt.close(fig)

vx=np.linspace(-.1,.9,300);old=np.minimum(vx,.4)/(.4+1e-5)*2;symmetric=2*np.exp(-np.abs(vx-.4)/.2)
fig,ax=plt.subplots(figsize=(8,4.3),constrained_layout=True);ax.plot(vx,old,label='현재 tracking_lin_vel_max ×2');ax.plot(vx,symmetric,label='기존 x_exp 함수 ×2 (수식 비교, 미적용)');ax.axvline(.4,color='#777',ls='--');ax.set_xlabel('실제 vx (m/s), 명령0.4m/s');ax.set_ylabel('Weighted reward /100 이전');ax.set_title('현재 보상은 과속 구간에서 포화');ax.legend();ax.grid(alpha=.15);fig.savefig(OUT/'velocity_reward_comparison.png',dpi=170);plt.close(fig)
print('Calculated B1 command far x:',b1_far,'current AS2 camera mass:',workspace['mass_kg']);print('All original parameters preserved; only reports/evaluation artifacts added.')
