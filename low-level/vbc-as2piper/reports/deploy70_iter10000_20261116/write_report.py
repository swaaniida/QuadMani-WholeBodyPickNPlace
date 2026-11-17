from pathlib import Path
import json,math
p=Path(__file__).resolve().parent
s=json.loads((p/'summary.json').read_text());opt=json.loads((p/'optimizer_summary.json').read_text());train=json.loads((p/'training_window.json').read_text());integrity=json.loads((p/'checkpoint_integrity.json').read_text())
lines=['# Deploy70N・10000 iteration 검증','',
'## 판정','',
'- 수치 발산/지속적인 정지 정책 증거 없음. 학습 프로세스 유지; 설정/Reward/PPO 및 checkpoint 변경 없음.',
'- 정지 RR 무접촉과 이동 과속이 남아 있어 배포 적합 판정은 불가. 70N 감소만으로 네 발 접지/속도 정확성이 확보되지 않음.',
'', '## 조건','',
'- B70N model_10000.pt, strict load, history encoder 기반 deterministic inference(실제 배포용 actor 경로), optimizer update0.',
'- GPU 메모리 여유19.98GiB 확인 후 기존 학습과 별도 GPU PhysX 평가. 평지/rough를 순차 실행하여 추가 부하 제한.',
'- 지형별32환경,4명령×8환경,40초,초기2초 제외. v_y=0; 정지/전진0.4m/s/회전0.6rad/s/전진+회전. DR/Push 비활성화,timeout 및 일반 reset 유지.',
'- 동일한 작은 중앙 EE 구면 궤적: r=0.44+0.02sin(t),고도각=0.3+0.1sin(0.7t),좌우각=0.15sin(0.5t). 확장 workspace 전체 및 바닥 목표 시험이 아니며 EE 수치는 위치 오차다. 목표 orientation 추가delta0; orientation 오차는 이번 evaluator가 저장하지 않음.',
'- Foot contact 비율은 순간 net-force norm>2N 기준. 양의Fz 지지율/발바닥 간격으로 해석하지 않음. 발 위치 미저장으로 RR 들림 높이(mm)는 이번 결과에서 산출하지 않음.',
'', '## 실제 rollout','',
'|지형|명령|평균 vx(m/s)|vx RMSE|평균 yaw(rad/s)|yaw RMSE|EE 평균/P95(cm)|실패/timeout|roll 절댓값 평균(°)|',
'|---|---|---:|---:|---:|---:|---:|---:|---:|']
labels={'stand':'정지','forward':'전진','turn':'회전','forward_turn':'전진+회전'}
for terrain,data in s.items():
 for name,x in data['groups'].items():
  m=x['metrics'];lines.append(f"|{terrain}|{labels[name]}|{m['Tracking/vx_mean_m_s']:.4f}|{m['Tracking/vx_rmse_m_s']:.4f}|{m['Tracking/yaw_mean_rad_s']:.4f}|{m['Tracking/yaw_rmse_rad_s']:.4f}|{m['EE/position_mean_m']*100:.2f}/{m['EE/position_p95_m']*100:.2f}|{m['Reset/failures']}/{m['Reset/timeouts']}|{x['roll_abs_mean_deg']:.2f}|")
lines+=['','## 접촉/Reward','', '|지형|명령|힘 평균/P95/P99(N)|FL/FR/RL/RR 접촉율(%)|4발 동시접촉(%)|Contact Force weighted|', '|---|---|---:|---:|---:|---:|']
for terrain,data in s.items():
 for name,x in data['groups'].items():
  m=x['metrics'];c='/'.join(f'{v*100:.3f}' for v in x['contact_norm_gt2_fraction'].values());lines.append(f"|{terrain}|{labels[name]}|{m['Force/norm_mean_N']:.1f}/{m['Force/norm_p95_N']:.1f}/{m['Force/norm_p99_N']:.1f}|{c}|{x['all_four_contact_fraction']*100:.3f}|{m['RewardWeighted/leg/feet_contact_forces']:.5f}|")
lines+=['','- 70N은 hard force limit이 아니라 발별 norm 초과분 페널티 시작점. weighted=−0.001×Σmax(norm(F)-70,0),reset후2초제외. 학습 reward는 합산 후/100.',
'- 평지 전진에서 tracking_lin_vel_max weighted가 약+1.98로 거의 포화. 양의명령에서 min(vx,command)/command이므로 명령 초과 속도를 직접 구분/감점하지 않는다. 이번 검사에서는 변경하지 않음.',
'- 정지 contact shaping force/vel0 확인. moving shaping 활성. 전체 reward raw/weighted/share는 result.json 및 reward_contributions.csv.',
'', '## 이전200N・10k 참고 비교','', '|지형/전진|이전200N 평균 vx|현재70N 평균 vx|이전 forceP99|현재 forceP99|','|---|---:|---:|---:|---:|']
for terrain,data in s.items():
 x=data['groups']['forward'];a=x['previous200N_10k_reference'];b=x['metrics'];lines.append(f"|{terrain}|{a['Tracking/vx_mean_m_s']:.4f}|{b['Tracking/vx_mean_m_s']:.4f}|{a['Force/norm_p99_N']:.1f}|{b['Force/norm_p99_N']:.1f}|")
lines+=['','- 이전 nocontact와 현재 deploy는 EE 가상중심/생성범위,카메라 포함 URDF,팔 명령 필터 등이 다르다. 200→70N만 바꾼 통제 ablation이 아니므로 성능차의 원인을 threshold로 단정하지 않는다.',
'', '## 학습 로그와 무결성','',f"- checkpoint SHA256: `{integrity['sha256']}`;저장iteration10000;모델/optimizer Tensor{integrity['tensor_count']}개 모두finite.",
f"- 학습구간9500~10000 평균: EE오차{train['means']['EE/position_mean_m']*100:.2f}cm,실패 reset/1000envsteps={train['means']['Reset/failures_per_1000_env_steps']:.6f},forceP95/P99={train['means']['Force/norm_p95_N']:.1f}/{train['means']['Force/norm_p99_N']:.1f}N. DR/Push 포함 학습구간이므로 별도평가와 직접동일조건이 아님."]
for k in ['Loss/value_function','Loss/surrogate','Loss/hist_latent_loss','Loss/priv_reg_loss','Perf/total_fps','Train/mean_episode_length','Train/dones']:
 a=opt[k];lines.append(f"- {k}: 9000~10000 평균{a['mean']:.6f},범위[{a['min']:.6f},{a['max']:.6f}],finite={a['finite']}.")
lines+=['','- 평균episode약10.03초로 제한시간 근처. Train/dones는timeout 포함이므로 전도율과 다름.',
'- 기존 W&B 로컬 .wandb 기록의 history만 읽어서 Loss/Perf/Train을 추출. 원격 run/인증 파일 수정 없음.',
'- 독립 평가에서 actor hash 불변/optimizer0/source hash training manifest 일치 검사. 본학습 종료/재시작/설정변경 없음.',
'', '## 산출물','', '- summary.json,checkpoint_integrity.json,training_window.json,optimizer_summary.json,reward_contributions.csv.',
'- 평지/rough 원시 samples.npz와 result.json: low-level-as2piper-head-orin-deploy/experiments/contact_force_ablation/evaluations/B70_iter10000_{flat,rough}_20261116/.',
'- 기존 B200 10k 평가는 nocontact experiments/contact_force_ablation/evaluations/B_iter10000_{flat,rough}_20261112/result.json 참조.']
(p/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print('Saved',p/'REPORT_KO.md')
