# 70N・20000 iteration 평가 및 10k/15k 비교

## 최종 판단

- 수치 발산/실패 Reset/정지 정책 증거 없음. 모든 지형/명령에서failure0이며원래timeout은각그룹24회.
- 15k보다20k가전체적으로좋아졌다고보기어려움. 평지전진평균0.5605→0.5944m/s,rough0.5582→0.5956m/s로0.4m/s대비과속이각48.6/48.9%로커짐.
- 평지정지RR contact5.217→0.329%로후퇴. 20k의8환경모두1.37%이하이며미세하게부하되는발을제외하고정상4발지지는확보되지않음. 실제들림높이(mm)는이번데이터에없음.
- EE평균위치오차는평지1.68~1.75cm/rough1.71~2.35cm로15k와대체로유사. 중앙EE만시험했으므로바닥목표도달을입증하지않음.
- 평지전진body-height오차2.035→1.140cm개선. 정지높이오차는1.494→1.427cm로소폭개선. 높이보상개선이RR접지개선과동일하지않음.
- ForceP99는평지전진171.4→169.6N으로거의같지만복합169.2→216.8N/회전162.7→198.9N으로증가. Rough의힘수치는PhysX기록한계를포함.
- 속도추종·충격측면에서는현재15k가더균형적.20k는일부높이/회전평균추종이더좋아명확한전항목우위는없음. 어느쪽도배포완료판정은불가.
- 기존속도reward가목표초과에포화하는구조여서과속증가가높은보상과공존가능.원인분리실험을진행한것은아니며기존reward·PPO·DR등을임의변경하지않음.
- 기존학습프로세스/Checkpoint보존.이번평가의optimizer업데이트0.

## 조건

- 동일70N run의20k. 지형별32환경/4명령×8환경/40초,초기2초제외. 전진0.4m/s·yaw0.6rad/s·vy0. DR/Push OFF,10초timeout 유지. seed1, deterministic history actor(배포 경로),optimizer0.
- 10k/15k와 동일한 중앙EE궤적,actual current training source hash와manifest일치확인. workspace전체/바닥최대도달/EEorientation오차/실기검증은아님.
- GPU여유18.91GiB 확인 후flat/rough순차평가. 기존학습중단·재시작·설정수정없음.

## 지표: 10k → 15k → 20k

|지형|조건|평균vx(m/s)|vx RMSE|평균yaw(rad/s)|yaw RMSE|EE평균(cm)|실패reset|
|---|---|---:|---:|---:|---:|---:|---:|
|flat|stand|-0.005 → -0.003 → -0.003|0.020 → 0.016 → 0.015|0.005 → 0.007 → 0.009|0.089 → 0.025 → 0.024|1.655 → 1.659 → 1.685|0 → 0 → 0|
|flat|forward|0.668 → 0.560 → 0.594|0.291 → 0.174 → 0.205|0.013 → -0.006 → 0.026|0.252 → 0.045 → 0.064|1.910 → 1.695 → 1.707|0 → 0 → 0|
|flat|turn|0.018 → -0.020 → -0.039|0.047 → 0.027 → 0.042|0.548 → 0.571 → 0.606|0.168 → 0.097 → 0.109|1.668 → 1.666 → 1.684|0 → 0 → 0|
|flat|forward_turn|0.665 → 0.550 → 0.572|0.287 → 0.163 → 0.183|0.597 → 0.586 → 0.614|0.255 → 0.065 → 0.072|1.894 → 1.729 → 1.753|0 → 0 → 0|
|rough|stand|-0.006 → -0.003 → -0.003|0.035 → 0.018 → 0.018|0.009 → 0.008 → 0.009|0.275 → 0.030 → 0.029|1.727 → 1.685 → 1.710|0 → 0 → 0|
|rough|forward|0.678 → 0.558 → 0.596|0.310 → 0.186 → 0.215|-0.003 → 0.001 → 0.019|0.289 → 0.136 → 0.133|2.471 → 2.366 → 2.348|0 → 0 → 0|
|rough|turn|0.022 → -0.009 → -0.023|0.066 → 0.043 → 0.047|0.542 → 0.541 → 0.581|0.235 → 0.151 → 0.145|1.860 → 1.797 → 1.825|0 → 0 → 0|
|rough|forward_turn|0.661 → 0.556 → 0.576|0.288 → 0.177 → 0.194|0.569 → 0.574 → 0.597|0.273 → 0.123 → 0.121|2.302 → 2.178 → 2.174|0 → 0 → 0|

## 접촉·자세: 10k → 15k → 20k

|지형|조건|forceP95(N)|forceP99(N)|RR contact(%)|roll abs(°)|base height abs error(cm)|
|---|---|---:|---:|---:|---:|---:|
|flat|stand|104.781 → 106.580 → 113.296|148.306 → 109.500 → 115.382|0.046 → 5.217 → 0.329|0.452 → 0.339 → 0.111|1.086 → 1.494 → 1.427|
|flat|forward|214.200 → 151.648 → 154.696|294.736 → 171.371 → 169.634|45.789 → 47.967 → 47.908|0.480 → 0.219 → 0.361|0.729 → 2.035 → 1.140|
|flat|turn|151.833 → 132.405 → 143.122|199.177 → 162.695 → 198.939|47.579 → 51.651 → 57.138|0.341 → 0.267 → 0.292|1.916 → 2.467 → 1.787|
|flat|forward_turn|213.218 → 150.845 → 163.115|289.167 → 169.239 → 216.751|46.901 → 47.947 → 47.954|0.506 → 0.214 → 0.366|0.717 → 1.954 → 1.265|
|rough|stand|101.732 → 103.455 → 110.426|172.098 → 148.456 → 158.381|15.875 → 21.743 → 15.875|0.484 → 0.365 → 0.237|1.498 → 1.588 → 1.569|
|rough|forward|125.001 → 123.149 → 128.990|223.105 → 172.802 → 180.901|20.303 → 22.816 → 24.211|0.634 → 0.345 → 0.411|2.609 → 1.889 → 1.825|
|rough|turn|124.254 → 122.538 → 128.714|190.463 → 163.521 → 189.718|24.697 → 28.493 → 29.151|0.443 → 0.393 → 0.396|1.952 → 2.136 → 1.698|
|rough|forward_turn|133.003 → 134.143 → 138.614|234.552 → 172.476 → 192.577|21.908 → 26.066 → 24.796|0.551 → 0.264 → 0.409|2.209 → 1.720 → 1.604|

## 학습 로그
 
- checkpoint 20000 / SHA256 `262859ad99b63fc7ccf74583e6af2d11a6dd256f75ce22b061416b17bf22c544`;model/optimizer tensor164개 finite=True.
- 학습19500~20000 구간501회 평균: {'Tracking/vx_rmse_m_s': 0.23123263816038767, 'Tracking/yaw_rmse_rad_s': 0.3431901482645861, 'EE/position_mean_m': 0.04472531666031141, 'Reset/failures_per_1000_env_steps': 0.004494050094255913, 'Force/norm_p95_N': 166.049905677232, 'Force/norm_p99_N': 290.7096541627441, 'RewardWeighted/leg/feet_contact_forces': -0.0430205185598361, 'RewardRaw/leg/roll': 0.017576567787758842}
- Loss/value_function:mean=0.004733,range=[0.003696,0.010410],finite=True.
- Loss/surrogate:mean=-0.003717,range=[-0.004715,-0.000955],finite=True.
- Loss/hist_latent_loss:mean=0.012719,range=[0.011923,0.014027],finite=True.
- Loss/priv_reg_loss:mean=0.012525,range=[0.011387,0.014344],finite=True.
- Train/mean_episode_length:mean=501.631299,range=[494.280000,502.000000],finite=True.
- Train/dones:mean=0.001993,range=[0.001746,0.002451],finite=True.
- Perf/total_fps:mean=31721.528472,range=[29166.000000,33485.000000],finite=True.

## 해석 범위

- Contact는순간net force norm>2N. Fz 지지율 또는발높이측정이아님. Rough에서는triangle경계/접촉집계영향으로실제체공시간·하중지지율과다를수있음. 발위치/Fz미저장으로높이편차방향/발들림mm를단정하지않음.
- 70N은페널티시작점이며hard force limit이아님. 원본tracking_lin_vel_max는양의명령에서목표초과속도에포화하고과속을직접감점하지않음.
- Roll은절댓값평균으로좌우방향을의미하지않음. Timeout은failure와분리. 단일seed/GPU물리변동이있어미세차이의통계적유의성을확정하지않음.
- 학습로그는DR/Push/넓은EE범위를포함하므로중앙EE/명목물리평가와다른분포. Loss감소만으로배포적합판정하지않음.
- reward_contributions.csv는3체크포인트의raw가아닌실제weighted mean/absolute share/signed share를정리. raw 및원시physics는각평가result.json/samples.npz에보존.

## 산출물

- summary.json,comparison_10k_15k_20k.{json,csv},reward_contributions.csv,checkpoint_integrity.json,training_window.json,optimizer_summary.json.
- 원시평가: low-level-as2piper-head-orin-deploy/experiments/contact_force_ablation/evaluations/B70_iter20000_{flat,rough}_20261117/.
- 실행: run_force_ablation.sh --mode evaluate --condition B --seed 1 --device cuda:0 --checkpoint .../model_20000.pt --terrain flat 또는rough --seconds 40 --name B70_iter20000_{flat,rough}_20261117.

## 대표 Weighted Reward: 평지, /100 이전

|조건|항목|15k|20k|
|---|---|---:|---:|
|stand|stand_still|0.96333|0.96802|
|stand|walking_dof|0.00000|0.00000|
|stand|tracking_lin_vel_max|1.99066|1.99149|
|stand|base_height|-0.07468|-0.07133|
|stand|roll|-0.01185|-0.00386|
|stand|hip_pos|-0.00622|-0.00188|
|stand|torques|-0.01602|-0.01624|
|stand|feet_drag|-0.01270|-0.01270|
|stand|feet_contact_forces|-0.04303|-0.04799|
|stand|tracking_contacts_shaped_force|0.00000|0.00000|
|stand|tracking_contacts_shaped_vel|0.00000|0.00000|
|forward|stand_still|0.00000|0.00000|
|forward|walking_dof|1.35069|1.34252|
|forward|tracking_lin_vel_max|1.97632|1.98142|
|forward|base_height|-0.10173|-0.05699|
|forward|roll|-0.00763|-0.01259|
|forward|hip_pos|-0.00748|-0.00930|
|forward|torques|-0.03754|-0.03615|
|forward|feet_drag|-0.03374|-0.03253|
|forward|feet_contact_forces|-0.08817|-0.08875|
|forward|tracking_contacts_shaped_force|-0.11289|-0.09610|
|forward|tracking_contacts_shaped_vel|-0.11454|-0.13131|
