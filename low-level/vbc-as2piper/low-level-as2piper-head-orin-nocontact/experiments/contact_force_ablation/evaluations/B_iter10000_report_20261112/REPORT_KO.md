# B 조건 10,000 iteration 체크포인트 검증

## 결론

- 보행·회전 및 중앙 EE 궤적 추종은 동작한다. 발산·지속적인 정지 정책은 확인되지 않았다.
- 정밀 속도 추종·정지 안정성은 미흡하다. 0.4m/s 명령 대비 전진 0.673~0.704m/s, 정지 yaw 속도 RMS 0.515~0.538rad/s.
- 명확한 발산이나 지속적인 정지 정책이라는 중단 조건에는 해당하지 않아 기존 학습을 유지한다. 보상·PPO·DR·학습 코드는 수정하지 않았다. 정지 흔들림과 과속은 후속 체크포인트에서 재검증해야 한다.

## 조건과 정합성

- checkpoint: `/data/team/users/suhwan/vbc-as2piper/low-level-as2piper-head-orin-nocontact/experiments/contact_force_ablation/runs/as2piper_headpiper_orin_200N_0.18m/model_10000.pt`
- SHA256: `8cbb7d46080ce71677dcdf0a9ccdb87d39fbbf92f2a4a5511421a5b9fdbd942c`
- B: 200N, feet height 0.18m, gait clock 활성, contact observation 제거, 정지 push 2.5배인 학습 manifest 확인.
- 평지/rough 각각 32환경, 명령별 8환경, 40초 실행. 초기 2초 제외하여 명령별 15,200 env-steps 집계.
- 명령 (vx,vy,yaw): 정지 (0,0,0), 전진 (0.4,0,0), 회전 (0,0,0.6), 복합 (0.4,0,0.6).
- 평가에서만 DR/push 비활성. 원본 rough-flat 생성기의 4×4 terrain 사용, 평지는 높이 [0,0]. 다른 seed·외란·음의 방향은 본 평가 범위에 포함하지 않음.
- EE 구면 명령 r=0.44+0.02sin(t), pitch=0.3+0.1sin(0.7t), yaw=0.15sin(0.5t), orientation delta=0. 중앙 작업공간 평가이며 ground picking 검증은 아님.
- 학습 manifest와 평가 source hash 일치, 모델 strict load 성공, history 입력으로 deterministic 추론. 평가 optimizer 업데이트 0회, 평가 전후 모델 hash 일치.
- 체크포인트 모델 전체 유한값. 저장된 optimizer state 41개, 업데이트 step 예시 190,000. Value loss 0.0041, surrogate -0.0024, history loss 0.0174, privileged regularizer 0.0157.
- EE는 기존 DLS IK/position-drive를 포함한 전체 시스템 추종 결과이다. 학습된 arm action 단독 성능으로 해석하지 않는다.

## 추종·Reset

|지면|명령|실제 vx (m/s)|vx RMSE|실제 yaw (rad/s)|yaw RMSE|EE 평균/P95 (cm)|실패/시간종료 reset|
|---|---|---:|---:|---:|---:|---:|---:|
|flat|정지|-0.065|0.099|0.010|0.538|1.56/2.01|0/24|
|flat|전진|0.673|0.286|0.059|0.167|1.56/2.22|0/24|
|flat|회전|0.004|0.047|0.607|0.121|1.52/2.10|0/24|
|flat|전진+회전|0.663|0.278|0.638|0.225|1.58/2.41|1/24|
|rough|정지|-0.036|0.075|0.004|0.515|1.59/2.06|1/24|
|rough|전진|0.704|0.332|0.062|0.266|1.66/2.85|0/24|
|rough|회전|0.019|0.066|0.600|0.180|1.56/2.35|0/24|
|rough|전진+회전|0.684|0.306|0.620|0.214|1.61/2.66|0/24|

실패 reset은 timeout을 제외한 termination이다. 시간종료 reset은 정상적인 10초 episode 종료이며, 실패와 합산하지 않는다. 정지 yaw 평균이 0에 가까워도 RMS가 커 양·음 흔들림을 가린다.

## 시뮬레이션 발 접촉력

발 4개·전체 시간의 net contact force 벡터 norm을 풀링한다. 비접촉 샘플 0도 포함한다. 실제 센서 측정값이 아니다.

|지면|명령|평균 (N)|P95 (N)|P99 (N)|contact force reward|절대 reward 기여 (%)|
|---|---|---:|---:|---:|---:|---:|
|flat|정지|57.0|165.8|226.3|-0.00192|0.043|
|flat|전진|59.4|179.4|228.7|-0.00225|0.040|
|flat|회전|57.6|158.9|182.7|-0.00044|0.008|
|flat|전진+회전|59.8|198.4|255.3|-0.00605|0.106|
|rough|정지|17.9|112.8|183.9|-0.00118|0.026|
|rough|전진|18.5|128.7|208.7|-0.00226|0.038|
|rough|회전|21.1|126.2|182.4|-0.00136|0.025|
|rough|전진+회전|20.5|136.1|216.5|-0.00240|0.041|

Rough에서 평균 합력은 평지보다 현저히 작다. 접촉력은 Isaac Gym trimesh net-contact 추정값으로, 이 차이를 실제 하중 감소나 안정성 개선으로 해석할 수 없다. 실하중·강도 판정은 불가하다. 200N은 초과 페널티가 시작되는 threshold이며 힘을 200N 이하로 강제하는 물리 상한이 아니다.

## Reward 확인

- Contact Shaping force/vel: 정지 0, 이동 음수 페널티 확인.
- 전진 속도 보상은 원본 `min(vx_actual,vx_command)/vx_command` 형태로 목표 초과에서 포화한다. 실제 과속에도 전진 weighted reward 약 +1.98이므로 높은 reward를 정확한 속도 추종으로 해석하면 안 된다. 과속을 구분하지 못하는 구조적 한계이며, 관측된 과속의 단독 원인이라고 확정하지 않는다.
- `stand_still`은 관절 기본 자세와의 거리 보상이므로 정지 yaw 흔들림의 부재를 보장하지 않는다.
- weighted 값은 현재 실제 reward 합산 경로의 `/100` 이전 값. 절대 기여도는 각 leg reward 절댓값 합 대비 비율이며 arm branch와 분리한다.
- 모든 reward의 raw·weighted·절대 기여도는 `reward_contributions.csv`, signed 기여도와 전체 지표는 `summary.json`에 보존.

## 학습 중 10,000회 로그와 구분

- 학습 iteration 10000: vx RMSE 0.260m/s, yaw RMSE 0.505rad/s, EE 평균/P95 2.11/4.15cm, 실패 0·timeout 288, 발 힘 P95/P99 169.0/327.6N.
- 이 수치는 6,144환경·무작위 명령·DR·push가 섞인 1 iteration 집계이다. 고정 명령 평가와 직접적인 동일 조건 비교가 아니며, 1회만으로 장기 reset율을 판단하지 않는다.
