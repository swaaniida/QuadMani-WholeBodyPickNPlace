# 70N・15000 iteration 평가 및 10000 비교

## 평가 조건

- 동일 run의 model_10000/15000 비교. 70N/scale−0.001,다른 Reward/PPO/DR 변경 없음. 기존 학습 프로세스 중단/재시작 없음.
- 지형별32환경(정지/전진/회전/복합 각8),40초 중 초기2초 제외. seed1, DR/Push off, 원래10초timeout 유지, deterministic history actor, optimizer update0.
- 전진명령0.4m/s, 회전명령0.6rad/s, vy=0. 중앙 EE궤적은10k와 동일. 확장 workspace/바닥 목표 전체 또는 EE orientation 오차 검증은 아님.
- GPU 여유18.95GiB 확인 후 flat/rough를 순차 평가. 본학습과 별도 프로세스.

## 주요 지표: 10k → 15k

|지형|조건|평균vx(m/s)|vx RMSE|평균yaw(rad/s)|yaw RMSE|EE평균(cm)|실패reset|
|---|---|---:|---:|---:|---:|---:|---:|
|flat|stand|-0.0046→-0.0031|0.0198→0.0156|0.0052→0.0065|0.0892→0.0249|1.65→1.66|0→0|
|flat|forward|0.6677→0.5605|0.2913→0.1740|0.0129→-0.0059|0.2516→0.0446|1.91→1.70|0→0|
|flat|turn|0.0183→-0.0199|0.0475→0.0268|0.5479→0.5709|0.1679→0.0967|1.67→1.67|0→0|
|flat|forward_turn|0.6647→0.5499|0.2874→0.1630|0.5967→0.5860|0.2551→0.0648|1.89→1.73|0→0|
|rough|stand|-0.0060→-0.0034|0.0346→0.0177|0.0087→0.0075|0.2747→0.0296|1.73→1.68|0→0|
|rough|forward|0.6781→0.5582|0.3101→0.1860|-0.0029→0.0008|0.2894→0.1358|2.47→2.37|0→0|
|rough|turn|0.0221→-0.0088|0.0657→0.0429|0.5420→0.5408|0.2350→0.1508|1.86→1.80|0→0|
|rough|forward_turn|0.6609→0.5565|0.2883→0.1772|0.5688→0.5743|0.2727→0.1230|2.30→2.18|0→0|

## 접촉 및 자세: 10k → 15k

|지형|조건|forceP95(N)|forceP99(N)|RR contact(%)|roll abs mean(°)|
|---|---|---:|---:|---:|---:|
|flat|stand|104.8→106.6|148.3→109.5|0.046→5.217|0.45→0.34|
|flat|forward|214.2→151.6|294.7→171.4|45.789→47.967|0.48→0.22|
|flat|turn|151.8→132.4|199.2→162.7|47.579→51.651|0.34→0.27|
|flat|forward_turn|213.2→150.8|289.2→169.2|46.901→47.947|0.51→0.21|
|rough|stand|101.7→103.5|172.1→148.5|15.875→21.743|0.48→0.36|
|rough|forward|125.0→123.1|223.1→172.8|20.303→22.816|0.63→0.34|
|rough|turn|124.3→122.5|190.5→163.5|24.697→28.493|0.44→0.39|
|rough|forward_turn|133.0→134.1|234.6→172.5|21.908→26.066|0.55→0.26|

## 학습 로그/무결성

- checkpoint SHA256 `484d9f3da7682de1fbebefd41a18c7db9badccbf9d6f1996fbb3e7f5f2e14a12`; saved iteration15000, model/optimizer tensor164개 finite=True.
- training 14500~15000,501iterations 평균: {'Tracking/vx_rmse_m_s': 0.23913692395963118, 'Tracking/yaw_rmse_rad_s': 0.3625358131950249, 'EE/position_mean_m': 0.045233647088090814, 'Reset/failures_per_1000_env_steps': 0.006443276641162086, 'Force/norm_p95_N': 166.929775175601, 'Force/norm_p99_N': 291.1291703708064, 'RewardWeighted/leg/feet_contact_forces': -0.042854872887244, 'RewardRaw/leg/roll': 0.01923390657096328}
- Loss/value_function:mean0.006046,range[0.004738,0.012386],finite=True.
- Loss/surrogate:mean-0.003348,range[-0.004657,-0.001454],finite=True.
- Loss/hist_latent_loss:mean0.013529,range[0.012476,0.014466],finite=True.
- Loss/priv_reg_loss:mean0.013101,range[0.011821,0.014849],finite=True.
- Train/mean_episode_length:mean501.448052,range[493.910000,502.000000],finite=True.
- Train/dones:mean0.001994,range[0.001667,0.002495],finite=True.
- Perf/total_fps:mean34271.776224,range[31663.000000,35942.000000],finite=True.

## 해석 한계

- Contact는 net-force norm>2N의 순간 비율이며,양의Fz 지지율이나 발 높이 측정값이 아님. rough에서는 특히 triangle 경계/접촉 집계 영향이 있으므로 낮은 비율을 실제 체공시간으로 해석하지 않음.
- 70N은 force hard limit이 아닌 페널티 threshold. 원래 tracking_lin_vel_max는 양의명령에서 목표 이상 속도에 포화해 과속을 직접 감점하지 않음.
- timeouts는정상종료로 failure reset과 구분. 평균roll은절댓값이며좌우기울기방향을 뜻하지 않음.
- stochastic GPU physics/초기조건에 의한 실행간 변동이 있어 단일 seed 결과로 개선의 통계적 유의성을 확정하지 않음.
- full raw/weighted/share는reward_contributions.csv 및각result.json,원시physics는samples.npz. 본평가 source hash는training manifest와일치하는지evaluator에서검사.
- 검증 명령: run_force_ablation.sh --mode evaluate --condition B --seed 1 --device cuda:0 --checkpoint .../model_15000.pt --terrain flat 또는rough --seconds 40 --name B70_iter15000_{flat,rough}_20261116. 옵션과값 사이 공백은일반CLI표기대로사용.

## 최종 판단

- 동일 중앙 EE 및 속도명령 평가에서는15k가10k보다 전반적으로 개선됐다. 모든 지형/명령의 실패 reset0; 속도 RMSE,회전 RMSE,동적 forceP99가 대부분 감소.
- 평지 전진0.4m/s 명령에0.5605m/s(+40.1%),rough0.5582m/s(+39.5%)로 과속은 남아 있음. Composite도0.5499/0.5565m/s로 과속.
- 정지 RR 접촉norm>2N 비율은평지0.046→5.217%로 상승했으나,다른3발98.7~99.8%에 크게 못 미침. 정지 네 발 지지가 확보됐다고 볼 수 없음.
- 평균 roll 절댓값은0.21~0.39°로10k보다 감소. 중앙 EE 위치오차는평지1.66~1.73cm,rough1.68~2.37cm.
- 몸통높이 목표 오차(평지전진)는0.73→2.03cm로 증가. 높이 metric은절댓값으로측정했으므로높아졌는지낮아졌는지이번로그로단정하지 않음.
- 발산/지속적 정지 정책 징후는 없으므로검증을이유로학습중단하지않음. 다만 과속/RR미접촉이남아배포완료판정은불가. 임의로Reward/scale/설정수정하지않음.

## 대표 Reward 비교: 평지전진, /100 이전

|항목|10k|15k|
|---|---:|---:|
|walking_dof|1.35540|1.35069|
|tracking_lin_vel_max|1.97750|1.97632|
|tracking_ang_vel|0.38565|0.49556|
|base_height|-0.03645|-0.10173|
|roll|-0.01677|-0.00763|
|torques|-0.03737|-0.03754|
|feet_drag|-0.05364|-0.03374|
|feet_contact_forces|-0.10155|-0.08817|
|tracking_contacts_shaped_force|-0.11687|-0.11289|
|tracking_contacts_shaped_vel|-0.20066|-0.11454|

- 속도 Reward가1.98근처로포화해과속을직접구분하지않는구조는유지됨. gait/drag/force 보상개선과높이목표오차증가가함께발생한결과이며개별항목의인과효과를분리한실험은아님.
- Rough접촉력통계는PhysX tensor 측정값. 실제지지율과일치하는지독립검증하지않았으므로rough P99 감소만으로실제충격감소를확정하지않음.
