# B 조건 최종 45,000 iteration 체크포인트 검증

작성 UTC: `2026-11-14T05:32:10.351551+00:00`

## 최종 판단

**45,000회 학습과 W&B 동기화는 정상 완료됐다. 최종 모델의 정지·보행·회전·중앙 EE 추종은 동작하지만, 전진 정밀 속도 추종은 미흡하며 20,000회보다 악화됐다.** 최종 iteration을 모든 용도의 최고 모델로 선택할 근거는 없다.

- 평지/rough 모두 정지·전진·회전·복합 평가의 실패 reset0건. 시간종료는 각 지면96건으로 정상10초 episode 종료.
- 정지 yaw RMS 평지0.0308/rough0.0293rad/s, EE 평균 약1.5~1.6cm 유지.
- 전진0.4m/s 명령: 평지0.620/rough0.615m/s로 약54~55% 과속.20k0.584/0.581,30k0.603/0.596보다 전진 정확도가 낮다.
- 순수 회전0.6rad/s 명령: 평지0.653/rough0.637rad/s로 약6~9% 초과. 복합 명령의 yaw 평균은0.606/0.596rad/s.
- 발산·지속 정지 정책은 확인되지 않음. 추가 학습·설정 변경·영상 생성 없이 완료된 모델을 읽기 전용으로 평가.
- 전진 정밀 추종은20k, 순수 회전 drift/yaw RMSE는 대체로30k가 유리하다. 별도 seed·명령·외란 평가 전에 실물 배포용 최선 모델을 확정하지 않는다.

## 최종 파일·완료 근거

- checkpoint: `/data/team/users/suhwan/vbc-as2piper/low-level-as2piper-head-orin-nocontact/experiments/contact_force_ablation/runs/as2piper_headpiper_orin_200N_0.18m/model_45000.pt`
- SHA256: `3b979389d0624e794a5176a2431db381b487894578662e224ed1f9207e1b3d80`
- 저장 iteration45000, 모델 전체 tensor와 저장 main optimizer tensor 모두 finite. Optimizer state41개, Adam step예시855,000.
- 콘솔은44999/45000까지 완료, runner의 최종model_45000.pt 저장 후 W&B sync 완료 메시지 확인. 현재 해당 학습 프로세스 없음.
- W&B project `as2piper-low`, 표시명 `as2piper_headpiper_orin_200N_0.18m`, run ID `gdr8lou4`.
- W&B 동기화 로그의 artifact 수는0이다. 모델 checkpoint는 위 로컬 경로에서 보존되며 W&B artifact 업로드가 확인된 것은 아니다.

## 평가 조건

- 20k·30k와 source hash·seed1·평가 코드가 동일. GPU PhysX, history encoder 경로의 deterministic 추론, strict state load 성공. 평가 전후 model hash 일치, optimizer update0회.
- 지면별32환경, 명령별8환경,40초. 초기2초 제외하여 조건별15,200 env-steps. 평가에서만 DR/push 비활성.
- 명령(vx,vy,yaw): (0,0,0), (0.4,0,0), (0,0,0.6), (0.4,0,0.6).20k/30k와 같은 중앙 EE 구면 궤적·orientation delta0.
- 시작 GPU memory8,703/40,192MiB로 여유 확인 후 평지와rough 순차 실행.
- 실패 reset은 활성 roll/pitch/지형상대높이 threshold에서 timeout을 제외한 수치다. 몸통 접촉 종료는 비활성이므로 실패0만으로 모든 비정상 접촉이 없다고 판단하지 않는다.
- Ground picking, 강한 외란, 여러 seed, 실제 지연·토크/gain을 반영한 실물 배포 시험은 이번 평가 범위가 아니다.

## 최종 제어 성능

|지면|명령|vx 평균 / RMSE (m/s)|yaw 평균 / RMSE (rad/s)|EE 평균/P95 (cm)|실패/timeout|
|---|---|---:|---:|---:|---:|
|flat|정지|-0.002/0.019|0.003/0.031|1.54/1.85|0/24|
|flat|전진|0.620/0.230|0.022/0.064|1.52/1.88|0/24|
|flat|회전|0.011/0.030|0.653/0.101|1.53/1.95|0/24|
|flat|전진+회전|0.607/0.215|0.606/0.072|1.54/2.06|0/24|
|rough|정지|-0.002/0.016|0.001/0.029|1.55/1.85|0/24|
|rough|전진|0.615/0.230|0.019/0.120|1.56/2.31|0/24|
|rough|회전|0.016/0.046|0.637/0.125|1.56/2.16|0/24|
|rough|전진+회전|0.608/0.219|0.596/0.119|1.57/2.32|0/24|

## 20k·30k·45k 전진 속도 비교

|지면|명령|20k 평균/RMSE|30k 평균/RMSE|45k 평균/RMSE|
|---|---|---:|---:|---:|
|flat|전진|0.584/0.194|0.603/0.212|0.620/0.230|
|flat|전진+회전|0.571/0.180|0.600/0.208|0.607/0.215|
|rough|전진|0.581/0.198|0.596/0.210|0.615/0.230|
|rough|전진+회전|0.571/0.184|0.601/0.212|0.608/0.219|

원본 전진 reward는 목표 이상의 속도에서 포화한다. 실제 과속에도 높은 reward가 가능하므로 episode reward 상승을 정밀 추종 향상으로 해석하지 않는다. 이번 검증으로 과속의 단독 원인을 확정하지 않으며 수식·scale을 변경하지 않았다.

## 접촉력·Reward

힘은 발4개·전체 시간의 simulator net contact vector norm이며 비접촉0 포함. 물리 센서 측정값이 아니다.

|지면|명령|평균/P95/P99 (N)|force weighted reward|force 절대 기여 (%)|
|---|---|---:|---:|---:|
|flat|정지|57.2/115.1/122.8|0.00000|0.000|
|flat|전진|63.1/161.2/287.1|-0.00536|0.098|
|flat|회전|59.1/159.6/296.0|-0.00606|0.111|
|flat|전진+회전|60.9/160.7/228.4|-0.00381|0.069|
|rough|정지|17.7/112.2/141.8|-0.00048|0.011|
|rough|전진|16.0/117.8/174.5|-0.00196|0.035|
|rough|회전|19.9/123.2/178.3|-0.00179|0.033|
|rough|전진+회전|19.8/130.7/179.5|-0.00181|0.033|

- Contact Shaping force/vel은 정지에서0·이동에서음수로 정상. 실제 발 센서 observation 제거 상태 유지.
- 평지 순수 회전 force P99는30k210→45k296N으로 증가. 평지 복합은267→228N으로 감소. 모든 접촉 지표가 같은 방향으로 개선되지는 않았다.
- Rough net contact force 과소집계 가능성 때문에 평지/rough 힘을 실제 하중·안전성의 직접 비교에 쓰지 않는다. Threshold200N은 초과분 페널티 기준이며 물리적 힘 상한이 아니다.
- Weighted 값은 실제 reward 합산의 `/100` 이전. 전체 raw/weighted/절대·signed 기여도는 `comparison.csv`와 `summary.json`에 보존.
- EE는 DLS IK/position drive를 포함한 결과이며 arm actor 단독 성능이 아니다.

## 학습·ROA 로그

- 마지막44,000~44,999회: vx RMSE 평균0.2228m/s, yaw RMSE 평균0.3122rad/s, EE 평균1.929cm. 실패 reset 약2.618건/100만 env-steps. 접촉력 P99의iteration 평균288.47N.
- 위 값은 무작위 명령·DR·push가 혼합된 학습 지표다. 고정 명령 평가와 동일 조건이 아니며 분위수 평균도 전체 pooled 분위수가 아니다.
- 최종 로그 Value loss0.0029, surrogate-0.0054, history latent loss0.0093, privileged regularization0.0086. Mean reward16.18, mean episode length501.10step.
- 마지막44999는 PPO 회차다. History loss는 직전 DAgger 갱신의 유지 표시값이다. Loss 자체를 보행 성능 점수로 해석하지 않는다.
- History/privileged encoder의30k→45k 가중치 L2 변화량1.986/3.135. Adaptation module 학습 지속 확인.
- 저장 main optimizer와 모델은 정상이나 별도history optimizer·ROA counter·RNG·환경 history는 기존 포맷에 저장되지 않는다. 모델 추론과 완전히 동일한 학습 재개를 구분한다.
- 기존 코드·reward·PPO·DR·모든 checkpoint·B1 원본 보존. 추가 학습 없음.
