# Deploy70N・10000 iteration 검증

## 판정

- 수치 발산/지속적인 정지 정책 증거 없음. 학습 프로세스 유지; 설정/Reward/PPO 및 checkpoint 변경 없음.
- 정지 RR 무접촉과 이동 과속이 남아 있어 배포 적합 판정은 불가. 70N 감소만으로 네 발 접지/속도 정확성이 확보되지 않음.

## 조건

- B70N model_10000.pt, strict load, history encoder 기반 deterministic inference(실제 배포용 actor 경로), optimizer update0.
- GPU 메모리 여유19.98GiB 확인 후 기존 학습과 별도 GPU PhysX 평가. 평지/rough를 순차 실행하여 추가 부하 제한.
- 지형별32환경,4명령×8환경,40초,초기2초 제외. v_y=0; 정지/전진0.4m/s/회전0.6rad/s/전진+회전. DR/Push 비활성화,timeout 및 일반 reset 유지.
- 동일한 작은 중앙 EE 구면 궤적: r=0.44+0.02sin(t),고도각=0.3+0.1sin(0.7t),좌우각=0.15sin(0.5t). 확장 workspace 전체 및 바닥 목표 시험이 아니며 EE 수치는 위치 오차다. 목표 orientation 추가delta0; orientation 오차는 이번 evaluator가 저장하지 않음.
- Foot contact 비율은 순간 net-force norm>2N 기준. 양의Fz 지지율/발바닥 간격으로 해석하지 않음. 발 위치 미저장으로 RR 들림 높이(mm)는 이번 결과에서 산출하지 않음.

## 실제 rollout

|지형|명령|평균 vx(m/s)|vx RMSE|평균 yaw(rad/s)|yaw RMSE|EE 평균/P95(cm)|실패/timeout|roll 절댓값 평균(°)|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|flat|정지|-0.0046|0.0198|0.0052|0.0892|1.65/1.96|0/24|0.45|
|flat|전진|0.6677|0.2913|0.0129|0.2516|1.91/3.35|0/24|0.48|
|flat|회전|0.0183|0.0475|0.5479|0.1679|1.67/2.23|0/24|0.34|
|flat|전진+회전|0.6647|0.2874|0.5967|0.2551|1.89/3.43|0/24|0.51|
|rough|정지|-0.0060|0.0346|0.0087|0.2747|1.73/2.39|0/24|0.48|
|rough|전진|0.6781|0.3101|-0.0029|0.2894|2.47/5.56|0/24|0.63|
|rough|회전|0.0221|0.0657|0.5420|0.2350|1.86/3.68|0/24|0.44|
|rough|전진+회전|0.6609|0.2883|0.5688|0.2727|2.30/5.04|0/24|0.55|

## 접촉/Reward

|지형|명령|힘 평균/P95/P99(N)|FL/FR/RL/RR 접촉율(%)|4발 동시접촉(%)|Contact Force weighted|
|---|---|---:|---:|---:|---:|
|flat|정지|57.5/104.8/148.3|97.796/99.171/99.033/0.046|0.000|-0.03825|
|flat|전진|61.1/214.2/294.7|44.257/49.428/46.993/45.789|0.000|-0.10155|
|flat|회전|58.3/151.8/199.2|48.434/57.895/57.493/47.579|4.599|-0.07738|
|flat|전진+회전|59.2/213.2/289.2|41.743/49.599/46.697/46.901|0.000|-0.09573|
|rough|정지|20.8/101.7/172.1|40.822/37.257/42.566/15.875|1.217|-0.01723|
|rough|전진|18.2/125.0/223.1|20.342/22.105/21.447/20.303|0.013|-0.02536|
|rough|회전|21.1/124.3/190.5|22.717/26.388/26.204/24.697|0.605|-0.02526|
|rough|전진+회전|20.4/133.0/234.6|20.105/23.553/22.434/21.908|0.007|-0.02781|

- 70N은 hard force limit이 아니라 발별 norm 초과분 페널티 시작점. weighted=−0.001×Σmax(norm(F)-70,0),reset후2초제외. 학습 reward는 합산 후/100.
- 평지 전진에서 tracking_lin_vel_max weighted가 약+1.98로 거의 포화. 양의명령에서 min(vx,command)/command이므로 명령 초과 속도를 직접 구분/감점하지 않는다. 이번 검사에서는 변경하지 않음.
- 정지 contact shaping force/vel0 확인. moving shaping 활성. 전체 reward raw/weighted/share는 result.json 및 reward_contributions.csv.

## 이전200N・10k 참고 비교

|지형/전진|이전200N 평균 vx|현재70N 평균 vx|이전 forceP99|현재 forceP99|
|---|---:|---:|---:|---:|
|flat|0.6727|0.6677|228.7|294.7|
|rough|0.7036|0.6781|208.7|223.1|

- 이전 nocontact와 현재 deploy는 EE 가상중심/생성범위,카메라 포함 URDF,팔 명령 필터 등이 다르다. 200→70N만 바꾼 통제 ablation이 아니므로 성능차의 원인을 threshold로 단정하지 않는다.

## 학습 로그와 무결성

- checkpoint SHA256: `c54f03ab258f410f589656459bc1d46dd86b5d0a073140a3766302a8c9760821`;저장iteration10000;모델/optimizer Tensor164개 모두finite.
- 학습구간9500~10000 평균: EE오차4.32cm,실패 reset/1000envsteps=0.004467,forceP95/P99=165.3/290.7N. DR/Push 포함 학습구간이므로 별도평가와 직접동일조건이 아님.
- Loss/value_function: 9000~10000 평균0.004874,범위[0.003543,0.010317],finite=True.
- Loss/surrogate: 9000~10000 평균-0.002913,범위[-0.004165,-0.000394],finite=True.
- Loss/hist_latent_loss: 9000~10000 평균0.017837,범위[0.015362,0.020601],finite=True.
- Loss/priv_reg_loss: 9000~10000 평균0.017231,범위[0.014686,0.020682],finite=True.
- Perf/total_fps: 9000~10000 평균37446.840160,범위[34310.000000,39183.000000],finite=True.
- Train/mean_episode_length: 9000~10000 평균501.589211,범위[493.060000,502.000000],finite=True.
- Train/dones: 9000~10000 평균0.001993,범위[0.001880,0.002118],finite=True.

- 평균episode약10.03초로 제한시간 근처. Train/dones는timeout 포함이므로 전도율과 다름.
- 기존 W&B 로컬 .wandb 기록의 history만 읽어서 Loss/Perf/Train을 추출. 원격 run/인증 파일 수정 없음.
- 독립 평가에서 actor hash 불변/optimizer0/source hash training manifest 일치 검사. 본학습 종료/재시작/설정변경 없음.

## 산출물

- summary.json,checkpoint_integrity.json,training_window.json,optimizer_summary.json,reward_contributions.csv.
- 평지/rough 원시 samples.npz와 result.json: low-level-as2piper-head-orin-deploy/experiments/contact_force_ablation/evaluations/B70_iter10000_{flat,rough}_20261116/.
- 기존 B200 10k 평가는 nocontact experiments/contact_force_ablation/evaluations/B_iter10000_{flat,rough}_20261112/result.json 참조.

## Rough 접촉 판정의 한계

- Rough net contact force는 삼각형 경계와 접촉 집계의 영향을 받을 수 있다. 낮은 접촉 비율을 실제 체공 시간이나 하중 지지율로 해석하지 않는다.
- 이번 평가에는 Fz·발 위치·독립적인 지형 거리를 저장하지 않았으므로 접촉력 P95/P99는 PhysX Tensor 측정값으로 한정한다. 평지 RR의 거의 무접촉인 결과와 Rough 전체 발 접촉 비율 저하의 원인을 혼동하지 않는다.
- Rough에서 평균 발별 force norm은18~21N으로 평지57~61N보다 낮다. 낮은 힘 통계가 실제 충격 감소나 더 좋은 지지를 입증하지 않는다.
- 평가 global_steps=10000×24로 원본 지연 스케줄의1 policy step(20ms) 다리 action delay가 활성화된 구간을 사용했다.
