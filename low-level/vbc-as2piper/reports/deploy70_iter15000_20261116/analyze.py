"""Offline evaluation comparison: never updates production code or model."""
from pathlib import Path
import json,csv
import numpy as np
ROOT=Path('/data/team/users/suhwan/vbc-as2piper');OUT=Path(__file__).resolve().parent
EV=ROOT/'low-level-as2piper-head-orin-deploy/experiments/contact_force_ablation/evaluations'
old=json.loads((ROOT/'reports/deploy70_iter10000_20261116/summary.json').read_text())
summary={};rewardrows=[];comparison=[]
for terrain in ['flat','rough']:
 folder=EV/f'B70_iter15000_{terrain}_20261116';r=json.loads((folder/'result.json').read_text());npz=np.load(folder/'samples.npz');physics=npz['physics'];assert np.isfinite(physics).all()
 groups={}
 for j,(name,m) in enumerate(r['metrics'].items()):
  a=physics[:,j*8:(j+1)*8];c=a[:,:,9:13]>2
  x={'metrics':m,'contact_norm_gt2_fraction':dict(zip(['FL','FR','RL','RR'],c.mean((0,1)).tolist())),'all_four_contact_fraction':float(c.all(-1).mean()),'roll_abs_mean_deg':float(m['RewardRaw/leg/roll']*180/np.pi),'per_env_vx_mean_m_s':a[:,:,3].mean(0).tolist(),'per_env_ee_mean_m':a[:,:,6].mean(0).tolist()}
  groups[name]=x;b=old[terrain]['groups'][name];bm=b['metrics']
  comparison.append({'terrain':terrain,'condition':name,'old_vx':bm['Tracking/vx_mean_m_s'],'new_vx':m['Tracking/vx_mean_m_s'],'old_vx_rmse':bm['Tracking/vx_rmse_m_s'],'new_vx_rmse':m['Tracking/vx_rmse_m_s'],'old_yaw':bm['Tracking/yaw_mean_rad_s'],'new_yaw':m['Tracking/yaw_mean_rad_s'],'old_yaw_rmse':bm['Tracking/yaw_rmse_rad_s'],'new_yaw_rmse':m['Tracking/yaw_rmse_rad_s'],'old_ee_cm':bm['EE/position_mean_m']*100,'new_ee_cm':m['EE/position_mean_m']*100,'old_failures':bm['Reset/failures'],'new_failures':m['Reset/failures'],'old_force_p95':bm['Force/norm_p95_N'],'new_force_p95':m['Force/norm_p95_N'],'old_force_p99':bm['Force/norm_p99_N'],'new_force_p99':m['Force/norm_p99_N'],'old_rr_contact_pct':b['contact_norm_gt2_fraction']['RR']*100,'new_rr_contact_pct':x['contact_norm_gt2_fraction']['RR']*100,'old_roll_abs_deg':b['roll_abs_mean_deg'],'new_roll_abs_deg':x['roll_abs_mean_deg']})
  for k,v in m.items():
   if k.startswith('RewardWeighted/'):
    term=k.replace('RewardWeighted/','');rewardrows.append({'terrain':terrain,'condition':name,'term':term,'weighted_mean_before_div100':v,'abs_share':m['RewardAbsShare/'+term],'signed_share':m['RewardSignedShare/'+term]})
 summary[terrain]={'checkpoint_sha256':r['checkpoint_sha256'],'optimizer_steps':r['optimizer_steps'],'finite_physics':True,'groups':groups,'protocol':r['protocol']}
OUT.joinpath('summary.json').write_text(json.dumps(summary,indent=2))
OUT.joinpath('comparison_10k_15k.json').write_text(json.dumps(comparison,indent=2))
for name,rows in [('reward_contributions.csv',rewardrows),('comparison_10k_15k.csv',comparison)]:
 with OUT.joinpath(name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
for x in comparison:print(x)
opt=json.loads((OUT/'optimizer_summary.json').read_text());t=json.loads((OUT/'training_window.json').read_text());integ=json.loads((OUT/'checkpoint_integrity.json').read_text())
lines=['# 70N・15000 iteration 평가 및 10000 비교','','## 평가 조건','',
'- 동일 run의 model_10000/15000 비교. 70N/scale−0.001,다른 Reward/PPO/DR 변경 없음. 기존 학습 프로세스 중단/재시작 없음.',
'- 지형별32환경(정지/전진/회전/복합 각8),40초 중 초기2초 제외. seed1, DR/Push off, 원래10초timeout 유지, deterministic history actor, optimizer update0.',
'- 전진명령0.4m/s, 회전명령0.6rad/s, vy=0. 중앙 EE궤적은10k와 동일. 확장 workspace/바닥 목표 전체 또는 EE orientation 오차 검증은 아님.',
'- GPU 여유18.95GiB 확인 후 flat/rough를 순차 평가. 본학습과 별도 프로세스.',
'', '## 주요 지표: 10k → 15k','',
'|지형|조건|평균vx(m/s)|vx RMSE|평균yaw(rad/s)|yaw RMSE|EE평균(cm)|실패reset|',
'|---|---|---:|---:|---:|---:|---:|---:|']
for x in comparison:
 lines.append(f"|{x['terrain']}|{x['condition']}|{x['old_vx']:.4f}→{x['new_vx']:.4f}|{x['old_vx_rmse']:.4f}→{x['new_vx_rmse']:.4f}|{x['old_yaw']:.4f}→{x['new_yaw']:.4f}|{x['old_yaw_rmse']:.4f}→{x['new_yaw_rmse']:.4f}|{x['old_ee_cm']:.2f}→{x['new_ee_cm']:.2f}|{x['old_failures']}→{x['new_failures']}|")
lines+=['','## 접촉 및 자세: 10k → 15k','','|지형|조건|forceP95(N)|forceP99(N)|RR contact(%)|roll abs mean(°)|','|---|---|---:|---:|---:|---:|']
for x in comparison:lines.append(f"|{x['terrain']}|{x['condition']}|{x['old_force_p95']:.1f}→{x['new_force_p95']:.1f}|{x['old_force_p99']:.1f}→{x['new_force_p99']:.1f}|{x['old_rr_contact_pct']:.3f}→{x['new_rr_contact_pct']:.3f}|{x['old_roll_abs_deg']:.2f}→{x['new_roll_abs_deg']:.2f}|")
lines+=['','## 학습 로그/무결성','',f"- checkpoint SHA256 `{integ['sha256']}`; saved iteration{integ['iteration']}, model/optimizer tensor{integ['tensors']}개 finite={integ['finite']}.",f"- training 14500~15000,{t['n']}iterations 평균: {t['means']}"]
for k in ['Loss/value_function','Loss/surrogate','Loss/hist_latent_loss','Loss/priv_reg_loss','Train/mean_episode_length','Train/dones','Perf/total_fps']:
 a=opt[k];lines.append(f"- {k}:mean{a['mean']:.6f},range[{a['min']:.6f},{a['max']:.6f}],finite={a['finite']}.")
lines+=['','## 해석 한계','',
'- Contact는 net-force norm>2N의 순간 비율이며,양의Fz 지지율이나 발 높이 측정값이 아님. rough에서는 특히 triangle 경계/접촉 집계 영향이 있으므로 낮은 비율을 실제 체공시간으로 해석하지 않음.',
'- 70N은 force hard limit이 아닌 페널티 threshold. 원래 tracking_lin_vel_max는 양의명령에서 목표 이상 속도에 포화해 과속을 직접 감점하지 않음.',
'- timeouts는정상종료로 failure reset과 구분. 평균roll은절댓값이며좌우기울기방향을 뜻하지 않음.',
'- stochastic GPU physics/초기조건에 의한 실행간 변동이 있어 단일 seed 결과로 개선의 통계적 유의성을 확정하지 않음.',
'- full raw/weighted/share는reward_contributions.csv 및각result.json,원시physics는samples.npz. 본평가 source hash는training manifest와일치하는지evaluator에서검사.',
'- 검증 명령: run_force_ablation.sh --mode evaluate --condition B --seed 1 --device cuda:0 --checkpoint .../model_15000.pt --terrain flat 또는rough --seconds 40 --name B70_iter15000_{flat,rough}_20261116. 옵션과값 사이 공백은일반CLI표기대로사용.']
OUT.joinpath('REPORT_KO.md').write_text('\n'.join(lines)+'\n')
