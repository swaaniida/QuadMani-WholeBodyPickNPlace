"""Analyze saved evaluation records only: no training/model changes."""
from pathlib import Path
import json,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
EV=ROOT/'low-level-as2piper-head-orin-deploy/experiments/contact_force_ablation/evaluations'
previous={i:json.loads((ROOT/f'reports/deploy70_iter{i}_20261116/summary.json').read_text()) for i in [10000,15000]}
summary={};rows=[];rewardrows=[]
for terrain in ['flat','rough']:
 folder=EV/f'B70_iter20000_{terrain}_20261117';r=json.loads((folder/'result.json').read_text());a=np.load(folder/'samples.npz')['physics'];assert np.isfinite(a).all();groups={}
 for j,(name,m) in enumerate(r['metrics'].items()):
  physics=a[:,j*8:(j+1)*8];contact=physics[:,:,9:13]>2
  x={'metrics':m,'contact_norm_gt2_fraction':dict(zip(['FL','FR','RL','RR'],contact.mean((0,1)).tolist())),'all_four_contact_fraction':float(contact.all(-1).mean()),'roll_abs_mean_deg':float(m['RewardRaw/leg/roll']*180/np.pi),'per_env_vx_mean_m_s':physics[:,:,3].mean(0).tolist(),'per_env_ee_mean_m':physics[:,:,6].mean(0).tolist()};groups[name]=x
  for iteration,data in [(10000,previous[10000][terrain]['groups'][name]),(15000,previous[15000][terrain]['groups'][name]),(20000,x)]:
   v=data['metrics'];rows.append({'iteration':iteration,'terrain':terrain,'condition':name,'vx_mean_m_s':v['Tracking/vx_mean_m_s'],'vx_rmse_m_s':v['Tracking/vx_rmse_m_s'],'yaw_mean_rad_s':v['Tracking/yaw_mean_rad_s'],'yaw_rmse_rad_s':v['Tracking/yaw_rmse_rad_s'],'ee_mean_cm':v['EE/position_mean_m']*100,'ee_p95_cm':v['EE/position_p95_m']*100,'failures':v['Reset/failures'],'timeouts':v['Reset/timeouts'],'force_mean_N':v['Force/norm_mean_N'],'force_p95_N':v['Force/norm_p95_N'],'force_p99_N':v['Force/norm_p99_N'],'RR_contact_pct':data['contact_norm_gt2_fraction']['RR']*100,'roll_abs_deg':data['roll_abs_mean_deg'],'base_height_abs_error_cm':v['RewardRaw/leg/base_height']*100})
   for k,value in v.items():
    if k.startswith('RewardWeighted/'):
     term=k.replace('RewardWeighted/','');rewardrows.append({'iteration':iteration,'terrain':terrain,'condition':name,'term':term,'weighted_mean_before_div100':value,'abs_share':v['RewardAbsShare/'+term],'signed_share':v['RewardSignedShare/'+term]})
 summary[terrain]={'checkpoint_sha256':r['checkpoint_sha256'],'optimizer_steps':r['optimizer_steps'],'finite_physics':True,'groups':groups,'protocol':r['protocol']}
OUT.joinpath('summary.json').write_text(json.dumps(summary,indent=2));OUT.joinpath('comparison_10k_15k_20k.json').write_text(json.dumps(rows,indent=2))
for name,table in [('comparison_10k_15k_20k.csv',rows),('reward_contributions.csv',rewardrows)]:
 with OUT.joinpath(name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
current=[r for r in rows if r['iteration']==20000]
print(json.dumps(current,indent=2))
opt=json.loads((OUT/'optimizer_summary.json').read_text());t=json.loads((OUT/'training_window.json').read_text());i=json.loads((OUT/'checkpoint_integrity.json').read_text())
lines=['# 70N・20000 iteration 평가 및 10k/15k 비교','','## 조건','',
'- 동일70N run의20k. 지형별32환경/4명령×8환경/40초,초기2초제외. 전진0.4m/s·yaw0.6rad/s·vy0. DR/Push OFF,10초timeout 유지. seed1, deterministic history actor(배포 경로),optimizer0.',
'- 10k/15k와 동일한 중앙EE궤적,actual current training source hash와manifest일치확인. workspace전체/바닥최대도달/EEorientation오차/실기검증은아님.',
'- GPU여유18.91GiB 확인 후flat/rough순차평가. 기존학습중단·재시작·설정수정없음.',
'', '## 지표: 10k → 15k → 20k','', '|지형|조건|평균vx(m/s)|vx RMSE|평균yaw(rad/s)|yaw RMSE|EE평균(cm)|실패reset|','|---|---|---:|---:|---:|---:|---:|---:|']
for terrain in ['flat','rough']:
 for name in ['stand','forward','turn','forward_turn']:
  group=[v for v in rows if v['terrain']==terrain and v['condition']==name]
  cells=[' → '.join(f'{v[k]:.3f}' for v in group) for k in ['vx_mean_m_s','vx_rmse_m_s','yaw_mean_rad_s','yaw_rmse_rad_s','ee_mean_cm']]
  lines.append('|'+terrain+'|'+name+'|'+'|'.join(cells)+'|'+' → '.join(str(v['failures']) for v in group)+'|')
lines+=['','## 접촉·자세: 10k → 15k → 20k','','|지형|조건|forceP95(N)|forceP99(N)|RR contact(%)|roll abs(°)|base height abs error(cm)|','|---|---|---:|---:|---:|---:|---:|']
for terrain in ['flat','rough']:
 for name in ['stand','forward','turn','forward_turn']:
  group=[v for v in rows if v['terrain']==terrain and v['condition']==name]
  cells=[' → '.join(f'{v[k]:.3f}' for v in group) for k in ['force_p95_N','force_p99_N','RR_contact_pct','roll_abs_deg','base_height_abs_error_cm']]
  lines.append('|'+terrain+'|'+name+'|'+'|'.join(cells)+'|')
lines+=['','## 학습 로그',' ',f"- checkpoint {i['iteration']} / SHA256 `{i['sha256']}`;model/optimizer tensor{i['tensors']}개 finite={i['finite']}.",f"- 학습19500~20000 구간{t['n']}회 평균: {t['means']}"]
for k in ['Loss/value_function','Loss/surrogate','Loss/hist_latent_loss','Loss/priv_reg_loss','Train/mean_episode_length','Train/dones','Perf/total_fps']:
 v=opt[k];lines.append(f"- {k}:mean={v['mean']:.6f},range=[{v['min']:.6f},{v['max']:.6f}],finite={v['finite']}.")
lines+=['','## 해석 범위','',
'- Contact는순간net force norm>2N. Fz 지지율 또는발높이측정이아님. Rough에서는triangle경계/접촉집계영향으로실제체공시간·하중지지율과다를수있음. 발위치/Fz미저장으로높이편차방향/발들림mm를단정하지않음.',
'- 70N은페널티시작점이며hard force limit이아님. 원본tracking_lin_vel_max는양의명령에서목표초과속도에포화하고과속을직접감점하지않음.',
'- Roll은절댓값평균으로좌우방향을의미하지않음. Timeout은failure와분리. 단일seed/GPU물리변동이있어미세차이의통계적유의성을확정하지않음.',
'- 학습로그는DR/Push/넓은EE범위를포함하므로중앙EE/명목물리평가와다른분포. Loss감소만으로배포적합판정하지않음.',
'- reward_contributions.csv는3체크포인트의raw가아닌실제weighted mean/absolute share/signed share를정리. raw 및원시physics는각평가result.json/samples.npz에보존.',
'', '## 산출물','',
'- summary.json,comparison_10k_15k_20k.{json,csv},reward_contributions.csv,checkpoint_integrity.json,training_window.json,optimizer_summary.json.',
'- 원시평가: low-level-as2piper-head-orin-deploy/experiments/contact_force_ablation/evaluations/B70_iter20000_{flat,rough}_20261117/.',
'- 실행: run_force_ablation.sh --mode evaluate --condition B --seed 1 --device cuda:0 --checkpoint .../model_20000.pt --terrain flat 또는rough --seconds 40 --name B70_iter20000_{flat,rough}_20261117.']
OUT.joinpath('REPORT_KO.md').write_text('\n'.join(lines)+'\n')
