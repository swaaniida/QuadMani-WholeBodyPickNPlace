# VBC 대비 변경 감사 및 AS2 하드웨어 포팅 복구

## 1. 현재 판단

- 기존 `walk_wbc_vbc` run의 약24~25시간 ETA는 실제 약2초/iteration에 기반한다. 업데이트 수를 줄인 결과가 아니다.
- 추가 후보 IK 제거와 동시 실행 부하 감소는 속도 개선 원인이다. 평지·외란 축소는 학습 과제를 단순화한 변경이므로 일반화 성능 개선으로 해석할 수 없다.
- 앞선 gait500 실험은 reward·탐색·단계 분리 등을 함께 변경했다. 개별 변경이 보행 성공에 필수였다는 원인별 ablation 근거는 없다.
- 불필요한 제약과 임의 reward 변경을 복구한 별도 task를 추가했다: `as2piper_plate_vbc_port`.
- 원본 B1, 이전 AS2/forward 실험은 보존했다. 이 복제본의 이전 task도 비교용으로 남겼다.

비교 기준은 서버의 VBC 소스와 기본 설정이다. 과거 B1 W&B run의 미기록 override 전체를 재구성한 것은 아니다.

## 2. 약25시간으로 줄어든 원인

현재 run은6,144환경·rollout24step·50Hz 제어·총45,000iteration이다. 전체 transition 수는6,635,520,000으로 기존 설정과 같다. Physics timestep0.005s, decimation4, PPO epoch5/minibatch4도 유지한다.

현재 run의 초반 안정 구간은 평균 약2.03초/iteration이었다. 이전 약4초대 속도 검사는 기존 느린 학습과 진단 학습을 함께 실행한 조건이었다. 현재는 해당 학습 한 개만 실행하므로 두 측정의 GPU 부하가 같지 않다.

이전 plane 대조 실험에서는 추가 후보 검사를 켰을 때26.11초 중22.37초가 목표 생성,20.36초가 반복 IK였다. 해당 검사만 원본 VBC 방식으로 되돌린 대조군은3.79초였다. 이는 후보 검사 비용을 분리한 결과이며, 좁은 명령 범위를 정당화하는 근거가 아니다.

이번 감사에서는 실제 Runner처럼 `env.reset()`을 먼저 호출하고, Python/NumPy/Torch seed와 PhysX 설정 전체를 일치시킨 진단 경로를 추가했다. 상세 결과는 아래 JSON 파일을 기준으로 한다.

- `profile_current_seed1.json`: 기존 좁은 범위·plane task.
- `profile_port_plane_full_seed1.json`: 복구 task, plane/full DR.
- `profile_port_rough_full_seed1.json`: 복구 task, original rough/full DR.
- `profile_b1_original_seed1.json`: 원본 B1 소스를 읽기 전용으로 실행한 비교.

모두 별도 AS2 학습이 실행 중인 조건이다. 단독 장시간 소요 시간으로 그대로 외삽하지 않는다. Rough terrain 복구 후에도 기존25시간 ETA가 유지된다고 보장하지 않는다.

| 동일 GPU/6,144환경/24step/seed1 진단 | 수집 시간 |
|---|---:|
| 기존 제한 AS2 task |2.628s|
| 복구 AS2 task, plane/full DR |2.958s|
| 복구 AS2 task, original rough/full DR |5.582s|
| 원본 B1, original rough/full DR/gait 관측 |5.636s|

복구한 AS2와 원본 B1의 수집 비용은 이 측정에서 비슷했다. 같은 복구 AS2 설정에서 plane→rough만 바꾼 비교는2.958→5.582s였다. 평지 변경을 속도 최적화와 구분해야 한다. 이는 한 seed의 짧은 zero-action 측정이며 학습 수렴 속도 비교는 아니다.

## 3. B1의 arm-base 기준과 body 회전

B1은 관측 원점과 목표 구면 중심을 서로 다르게 정의한다.

```text
B1 관측 원점:
  o_obs = p_body + R_yaw × [0.3, 0, 0.09]
  ee_obs = R_body^T × (goal_world - o_obs)

B1 목표 구면 중심:
  c_goal = [body_x, body_y, 0] + R_yaw × [0.3, 0, 0.7]
  goal_world = c_goal + R_yaw × sphere_to_cartesian(command)
```

관측에는 arm-base offset을 사용한다. 그러나 목표 중심은 실제 arm base를 그대로 따라가는 좌표계가 아니다. Yaw를 따르되 world Z는0.7m로 고정한다.

이전 AS2 변경은 다음과 같았다.

```text
c_previous = p_body + R_body × mount_offset
```

여기에는 root Z 추종과 mount offset의 roll/pitch 회전이 모두 추가된다. 구면 축 전체를 body 회전시킨 것은 아니지만, 원점의 정의를 바꾼 것이다.

### 확인한 영향

팔 관절과 body 회전을 유지한 채 root 높이만±5cm 바꾸면:

| 방식 | EE 실제 높이 변화 | 목표 높이 변화 | EE 높이 오차 변화 |
|---|---:|---:|---:|
| 이전 AS2 |±5cm|±5cm|0|
| 원본 VBC 방식 |±5cm|0|±5cm|

이전 방식은 몸통 높이 이동이 위치 추적 오차에서 상쇄된다. 따라서 몸통 높이를 사용해 EE 목표에 접근하도록 학습할 유인이 사라진다. 이 변경은 단순한 좌표 보정이 아니었다.

### 복구

- 목표 중심: 원본 VBC 함수 직접 상속.
- AS2 중심 offset: `[0.005, 0, 0.40643538]m`.
- 관측 원점: body 위치 + yaw 회전된 실제 AS2 mount offset `[0.005,0,0.08202848]m`.
- 관측의 body 역회전: 원본 유지.
- 로봇 자체의 실제 mount 기구학은 URDF에 따라 body 전체 회전을 계속 따른다.

원본처럼 목표 XY는 로봇 위치/yaw를 따른다. 전체 목표가 world XYZ에 고정되는 물체 추적 인터페이스를 구현한 것은 아니다.

## 4. 불필요하거나 근거가 부족해 복구한 변경

| 영역 | 이전 변경 | 복구 내용 | 이유 |
|---|---|---|---|
| 지형 |plane, height measurement off|원본 rough-flat trimesh와 terrain 설정 전체|환경 단순화는 하드웨어 포팅 필수 조건이 아님|
| Friction |0.6~1.2|0.3~3.0|재질 실측 근거 없이 범위를 축소했음|
| Leg strength |0.9~1.1|0.7~1.3|원본의 강건성 범위 복구|
| Mass/CoM/payload/push |nominal에서 비활성|모두 활성|일반화 대상 제거를 복구|
| 추가 mass |전체 질량10% cap|B1 범위×질량비:0~5.879kg|임의 cap 제거, 동일 상대 질량 범위|
| CoM |지지영역5% cap|B1 범위×길이비:각축±0.09427m|임의 지지영역 cap 제거|
| 보행 명령 |전진0.15~0.4, yaw±0.5, 고정 혼합비|원본 continuous sampler와5000회 이후 후진 전환|후진 제외·고정 명령 비율 제거|
| 속도 envelope |제한된 전진 범위|AS2 선속도±0.634m/s, yaw±1rad/s|선속도만 중력/길이 시간 스케일 적용|
| Gait clock |2.523Hz|원본2Hz|주파수 변경의 개별 학습 이득이 입증되지 않음|
| Leg action scale |0.2018/0.2247/0.239|원본0.4/0.45/0.45|임의20% effort 예산으로 정책 평균까지 축소한 변경 제거|
| Leg reset |±5%|원본±20% multiplicative reset|초기 상태 다양성 복구|
| Arm reset |C 자세에 고정|원본처럼 기본각±20%|초기 EE 오차를 없애기 위한 동결 제거|
| Root 초기 속도 |±0.02|원본±0.1|축소 필요성이 입증되지 않음|
| Root 초기 높이 |매 reset 발 FK로 재배치|AS2 명목 높이0.3264m+terrain origin|원본 reset 구조로 복구|
| 초기 EE end |좁은 범위 중심|[0.46,0.25,0]|원본처럼 시작점에서 초기 이동 유지|
| Checkpoint 간격 |100회|원본200회|불필요한 운영 설정 변경 제거|
| Gait 선행/transfer |500회 보행+입력층 전환|현재 실험에서 미사용|WBC를0부터 동시에 학습|

원본 terrain의 roughness 높이도 그대로0~0.1m이다. 평지를 유지하거나 terrain 난도를 낮추는 별도 curriculum을 추가하지 않았다.

## 5. Reward 복구

모든 leg/arm reward 가중치는 B1 config와 일치하도록 검사했다.

- `walking_dof`:0→원본1.5.
- `tracking_lin_vel_max`:0→원본2.
- 추가 Gaussian `tracking_lin_vel=3` 제거.
- `tracking_ang_vel`:1→원본0.5, 원본 함수/sigma0.2.
- `feet_air_time`:원본 함수·가중치2·0.5초 기준·앞발2개 기본 설정.
- `feet_height`:4발 phase 기반40mm clearance 함수 제거, 원본 함수/가중치1.
- 정지 시 네 발 desired contact를1로 덮어쓴 override 제거, 원본 접촉 위상 함수 사용.

하드웨어 단위 상수만 변환했다.

| 상수 | 현재 값 | 근거 |
|---|---:|---|
| Base height target |0.3244m|AS2 명목 기하|
| Contact force 기준 |15.678N|원본40N×질량비|
| Feet height norm 기준 |0.18855m|원본0.3m×길이비|
| Gait force sigma |0.07681|원본0.5×질량비²|
| Gait velocity sigma |0.31425|원본0.5×길이비|

이전167N force 기준과4cm 발 높이 목표는 설계자의 별도 선택이었다. 원본과 같은 목표 정의를 비교하려는 이번 task에서는 제거했다.

### 별도로 유지한 논리 수정

Contact-shaped force/velocity는 원본에서 raw≤0, scale<0이어서 위반에 양의 보상을 준다. 예를 들어 swing 중 큰 접촉력은 원본에서 raw≈−1, scale−2이므로+2다. 새 task는 raw 부호만 반전하여−2가 되도록 유지했다. 가중치는 원본과 같다.

이는 명시적인 비원본 변경이다. 부호의 물리적 의미는 수식으로 확인했지만, 보행 수렴 개선량을 독립적으로 입증한 실험은 아니다. 그 외 reward 함수 재설계는 복구했다.

## 6. 하드웨어에 필요해 남긴 변경

- AS2/PiPER/UMI asset, mount 위치, TCP, 관절/발 이름과 순서 검사.
- Finger 열림 제어 및 URDF joint-limit clamp.
- 실제 `base_link`에 mass/CoM DR 적용: B1의 body index1을 AS2에 그대로 사용하지 않음.
- AS2 발 좌표계에 맞춘 force sensor 위치.
- 다리 Kp59.467/53.404/75.327, Kd2.626/2.358/2.628.
- AS2 simulation preset의 leg armature0.026/0.026/0.038.
- 질량·길이 단위에 따른 DR/속도/물리 reward 기준값 변환.
- PPO sampled action이 환경의 arm masking으로 덮어써지지 않도록 `actions.clone()` 전달.
- Reset 직후 이전 episode의 EE 목표/방향을 한 step 실행하지 않도록 새 시작 목표 즉시 반영.
- Reset yaw cache를 현재 root yaw로 동기화.
- History optimizer/counter/RNG를 포함한 확장 checkpoint, atomic 저장, 파일 로그와 분리 실행.

Kp/Kd는 정적 지지 토크·관성 기반 초기 설계이며 최적 gain이나 실제 연속 모터 한계로 인증된 값이 아니다. URDF effort는 시뮬레이션 제한이다.

## 7. 탐색 std의 원인별 검사

AS2 Kp를 사용하면서 B1 action scale/std를 모두 그대로 사용하면 상대적인 초기 탐색 토크가 커진다. Action scale은 복구하되 std는 다음 비례식으로 정했다.

```text
std_AS2_j = std_B1_j × (80 × mass_ratio × length_ratio / Kp_AS2_j)
```

동일 action scale에서 초기 PD 목표 토크의 표준편차를 B1의 질량×길이 토크 스케일에 맞춘 것이다. 정책 평균 action 범위를 제한하는 방식이 아니다.

| 항목 | Hip | Thigh | Calf |
|---|---:|---:|---:|
| 초기 std |0.2651|0.3690|0.2616|
| 최소 std |0.0497|0.0923|0.0654|

동일 seed·128환경·4초·plane·DR off·실제 Runner reset/PhysX 설정, Gaussian action 비교:

- B1 std 그대로:reset113회, 최대 적용 torque90Nm.
- 비례식 std:reset0회, 최대 적용 torque52.93Nm.

이 검사는 학습된 정책의 성능 비교가 아니다. 해당 제어기의 초기 탐색 안정성을 분리해 확인한 결과다. 이 std를 사용한 full-DR/rough task의20회 PPO 업데이트와 저장도 통과했다.

## 8. 초기 자세와 reset 검토

10,000개 FK 샘플에서 원본 B1 초기 EE 위치 오차 평균은약8.79cm, AS2에 원본±20% arm reset을 적용하면약5.20cm였다. 초기 오차를0에 가깝게 만들기 위해 팔 reset을 동결할 근거가 없다.

원본 reset의 고정 root 높이와 독립 관절 변동에는 초기 발 관통이 존재한다. B1도 해당된다. AS2 명목 높이와 원본 reset을 복구한128환경·4초 zero-action 검사에서는 reset0회였다. 이는 reset 직후 시뮬레이션 transient 검사이며 실물에 임의 관절각을 즉시 적용해도 안전하다는 의미가 아니다.

## 9. EE 범위 과도 축소의 원인과 복구

이전 collision guard는 AS2 몸통·머리 STL 전체를 하나의 AABB로 감쌌다. 본체와 머리 사이 빈 공간도 채워진 것으로 판정한다. 보수적인 상자가 겹쳤다는 사실을 실제 관통으로 해석하면 작업 범위를 불필요하게 줄인다.

실제 MuJoCo model의 collision 형상으로 비교한 `port_inner` 초기 검토에서 AABB가 충돌로 판정한43개 pose는 실제 body contact가 모두0이었다.

| 항목 | 이전 좁은 task | 복구 task |
|---|---|---|
| Radius |0.46~0.49m|0.40~0.50m|
| Pitch |0.30~0.45rad|0.10~0.55rad|
| Yaw |±0.10rad|±0.30rad|
| 방향 delta 각 축 |±0.015rad|±0.10rad|

구면 parameter box 부피는30배, 해당 Cartesian 구면 sector 부피는약27.3배다. 원본 VBC의 독립 sampling/10점 box·floor 검사/보간을 사용하며 학습 중 추가 arm IK/OBB 검사는 없다.

### 사전 검사 결과와 한계

- 별도 seed:1,088경로×17점=18,496pose. 실제 body collision0. 위치1.2cm/방향5도 기준을 넘긴 경로4개(0.37%), 최대 위치1.20cm/방향7.01도.
- Body roll/pitch±0.1rad, root 높이0.30/0.3244/0.35m:5,184경로 검사. 실제 body collision0, 엄격한 IK 기준 초과42경로(0.81%). 일부는 고정 body에서의 미수렴/도달 한계이며 전신 정책의 실패율은 아니다.
- 고정 몸통 PhysX32환경15초:위치 평균1.61cm/p95 3.11cm, 방향 평균4.35도/p95 10.17도, arm contact>1N 관측0.

따라서 전체 pose/연속 경로에서 완벽한6D 추적을 보증하지 않는다. 위치·방향 범위를 넓힌 만큼 추적 오차 분포도 함께 공개한다.

더 낮은 pitch/ground 후보도 비교했다. 단순 직사각형 범위를 바닥까지 확대하면 nominal body에서 IK 기준 초과와 실제 body 접촉이 증가했다. 해당 결과를 전신 crouch로도 불가능하다고 단정하지는 않는다. 현재 확정 범위는 전방 작업 영역이며 바닥 파지까지 검증한 범위는 아니다.

## 10. 실행·검증 상태

- Full rough/DR,512환경×24step×20iteration:체크포인트 없이0부터 실행·완료.
- Model tensor finite, 다음 iteration20, 실제 관측799/관절좌표20/action18/arm Jacobian6×6 검사 통과.
- 원본 reward scale/arm scale/terrain/action scale 일치 검사를 통과.
- 명령 추출·목표 중심·관측 helper·충돌 검사·접촉 위상을 원본 `ManipLoco`에서 직접 상속.
- PPO 차이는 하드웨어 비례 std와 experiment name뿐이다. 업데이트 수식·네트워크·storage는 원본이다.
- 20회 진단은 보행 수렴 검증이 아니다. 진단 체크포인트를 장시간 학습에 재사용하지 않는다.

일부 수동 진단은 초기에는 PhysX buffer 기본값을 사용했다. Rough terrain에서 capacity 경고가 난 결과는 제외하고 원본 cfg 전체와 명시적 `env.reset()`을 적용해 다시 측정했다. 실제 `train_hw.py`는 원래부터 TaskRegistry의 전체 sim config와 Runner reset을 사용하므로 학습 설정을 누락한 문제는 아니다.

Finger는 현재20개 simulation joint 좌표 중2개로 남아 있다. 실제 기구의1개 개폐 자유도와 물리적 연동 구속을 완전히 구현한 모델이라는 뜻은 아니다. 이번 변경에서 이 구조나 V7 형상을 임의로 바꾸지 않았다. 팔 torque reward 누락, orientation 관측/reward0도 원본과 같은 제한이다.

## 11. 전환 명령

현재 실행 중인 좁은 task는 자동 변경되지 않는다. 현재 Codex는suhwan이고 run은vbc 소유이므로 직접 종료할 수 없다. vbc 터미널에서 다음 명령으로 현재 run을 종료하고, 비교 기록을 보존한 채 새 task를0→45000으로 시작한다.

```bash
cd /data/team/users/suhwan/vbc-as2piper
python low-level-as2piper-hw/restart_fresh.py \
  --old-run-id as2piper_hw_gaitfix_vbcsampler_scratch_20261029_134701 \
  --keep-old-run
```

프로젝트:`as2piper-low`. 새 ID:`as2piper_vbcport_scratch_<UTC시각>`.

상세 config 차이:`audit_hw/vbc_port_audit_20261029/config_contract.json`.
실행 당시 설정:`logs/as2piper-low/vbc_port_smoke20_20261029/runtime_snapshot.json`.
변경 diff·보존 검사·시간 측정 원본도 같은 audit 디렉터리에 저장한다.
