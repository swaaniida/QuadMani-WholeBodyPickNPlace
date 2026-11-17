# EE 명령 생성 병목 수정: 사전 범위 검토 + 원본 VBC 학습 경로

이 문서는 이전 제한 범위 실험 기록이다. 현재 기본 task와 복구 사항은 `VBC_PORT_AUDIT_KO.md`를 우선한다. 전체-body AABB의 보수적 거절을 실제 관통으로 일반화하지 않는다.

## 적용 대상

- 현재 권장 task: `as2piper_plate_walk_wbc_vbc`.
- 기존 HW/nominal task는 비교·복구용으로 보존했다.
- 새 장시간 실험은 체크포인트 없이 0→45000 iteration이다. Gait500/transfer520을 사용하지 않는다.
- PPO, reward, 다리 제어, nominal DR 설정은 이번 수정에서 변경하지 않았다.
- 기존 B1, 기존 AS2 및 forward 실험은 수정하지 않았다.

## 병목 측정

6,144환경, 24 control step, 같은 명목 모델·zero action·목표 재추출 일정에서 CUDA 동기화 계측을 수행했다. 기존 vbc 소유 학습과 동시에 실행했으므로 절대 속도에는 공유 GPU 부하가 포함된다. Nested timer는 서로 더하지 않는다.

| 항목 | 추가 pose guard | 원본 VBC 검사, 기존 범위 | 최종 task, 확정 범위 |
|---|---:|---:|---:|
| 전체 수집 시간 |26.108s|3.789s|3.814s|
| 목표 생성 포함 시간 |22.371s|0.0477s|0.0486s|
| 후보 반복 IK |20.363s|없음|없음|
| 팔/body OBB |0.658s|없음|없음|
| 목표 생성 외 시간 |3.737s|3.741s|3.765s|

추가 guard에서 5,888개 목표에 8,273개 후보를 검사했다(약1.405후보/목표). 거절 재추출만 폭증한 문제가 아니라 후보마다 수행한 반복 IK 자체가 주된 비용이다. 목표 생성이 수집 시간의85.7%, 반복 IK가78.0%를 차지했다.

원본 VBC 검사만 사용하되 기존 범위를 그대로 둔 대조군에서도 시간이 감소했다. 범위를 좁혀 계산을 생략한 것만으로 설명되는 개선이 아니다.

실제 PPO 진단은 새 task를 6,144환경으로 체크포인트 없이8 iteration 실행했다. Iteration2~7 전체 시간은3.91~4.45s였다. 마지막3회는3.92/4.19/3.91s였다. 약4초가 유지될 경우45000회는 약2.1일 규모다. 짧은 동시 실행 측정의 단순 외삽이며 단독 장시간 속도 보장은 아니다.

## 학습 중 제거한 추가 검사

- 목표 후보의 시작·중간·끝 pose를 반복 IK로 푸는 검사.
- 후보 arm/body OBB 검사.
- 실패 원인에 따른 추가52회 IK 반복.
- 실패 시 orientation까지 다시 추출하는 HW 전용 루프.
- HW 전용 이전 pose fallback.
- 학습 환경의 `ArmCollisionGuard` 객체 생성.

## 원본 VBC에서 유지한 검사·알고리즘

- Radius/pitch/yaw 독립 uniform proposal.
- Orientation delta 독립 추출.
- 목표 위치 최대10회 재추출.
- 구면 경로10점의 EE 금지 box/floor 검사.
- 구면 선형 보간, trajectory1~3초, hold0.5~2초.
- 실제 팔을 제어하는 매 control step의 DLS IK 및 position drive.

`_collision_check`는 `ManipLoco._collision_check`를 직접 사용하고, 초기화 이외 `_resample_ee_goal`도 `ManipLoco` 구현을 직접 호출한다. 초기화에서는 AS2 FK와 일치하는 현재 목표를 즉시 반영하는 기존 보완을 유지한다. 실제 팔 제어 IK를 제거한 것은 아니다.

## 확정 범위

Arm origin은 실제 PiPER mount를 사용하고, 구면 축은 yaw/중력 정렬을 유지한다.

| 항목 | 이전 | 확정 |
|---|---|---|
| 반경 |0.38~0.52m|0.46~0.49m|
| 구면 pitch |−0.25~0.60rad|0.30~0.45rad|
| 구면 yaw |±0.40rad|±0.10rad|
| Orientation delta 각 축 |±0.15rad|±0.015rad|
| 초기 end |[0.46,0.25,0]|[0.475,0.375,0]|

방향 delta의±0.015rad는 약±0.86도다. 기본 EE 방향은 구면 pitch/yaw에 따라 계속 변하므로 절대 방향이 고정되는 설정은 아니다.

## 사전 검토 방법과 결과

기존 범위와 compact/tight/front/extended 등 후보를 비교했다. 넓은 후보 중 몸통 간섭·IK 오차가 검출된 범위는 채택하지 않았다. 일부64회 IK 미수렴은 오프라인에서 추가 반복으로 분리 확인했다. 이 추가 반복은 학습 경로에 포함하지 않는다.

최종 별도 seed 검증:

- Seed20261030, 후보 선정 검사와 분리.
- 위치3축·방향 delta3축의64개 경계 조합과512개 무작위 목표.
- 경계 조합은 초기 FK 위치에서 출발, 무작위 목표는 독립적으로 추출한 이전 위치에서 출발.
- Body roll/pitch 각각−0.15,−0.075,0,+0.075,+0.15rad:25조건.
- 14,400경로 ×17점 =244,800pose.
- 최대256회 수준의 오프라인 IK 검토, 위치 허용1.2cm·방향 허용5도.
- 허용오차 초과0, arm/body OBB 간섭0, ground 위반0.
- 측정 최대 IK 위치 오차 약1.20cm, 방향 오차 약4.01도.

이는 샘플 검사이며 연속 공간 전체의 안전 증명이 아니다. Body tilt±0.15rad 범위 밖, self/leg/object/cable 충돌은 보장하지 않는다. 확정 범위는 현재 보행/전방 조작 시작 자세용이다. 명목 조건에서 TCP는 지면 약0.54m 이상이며 바닥 파지 범위를 검증한 것이 아니다.

## 실제 PhysX 추적·학습 확인

32환경15초, 몸통 고정·중력 적용·무작위 목표:

- 생성 요청149건.
- 무작위 구간 위치 평균1.47cm, p95 1.68cm.
- 방향 평균3.60도, p95 4.11도.
- Arm contact>1N 관측0.
- 추가 candidate IK/OBB 함수를 호출하면 AssertionError가 발생하도록 금지한 상태에서 통과.

체크포인트 없이512환경40 PPO iteration을 실행해 업데이트·저장·799관측·20DOF·18action·6×6 arm Jacobian 검사를 통과했다. 이후32환경10초 free-base 평가의 위치 평균은1.36~1.42cm, p95 1.55~2.03cm였고 넘어짐0이었다.40회 정책의 실제 전진 속도는 거의0이었다. 실행 및 추적 smoke 검사이며 보행 수렴 검증으로 해석하지 않는다. 해당40회 모델은 장시간 학습의 시작점으로 사용하지 않는다.

## 실행

기존 느린 run은 이미 로드한 이전 클래스로 계속 실행된다. 파일 수정만으로 실행 중인 프로세스가 바뀌지 않는다. 현재 Codex는suhwan, 해당 프로세스는vbc 소유여서 직접 종료할 수 없다.

인증된vbc 셸에서 현재 느린 run만 종료·W&B/로컬 기록 삭제 후 새 task를0부터 시작:

```bash
cd /data/team/users/suhwan/vbc-as2piper
python low-level-as2piper-hw/restart_fresh.py \
  --old-run-id as2piper_hw_gaitfix_ee_scratch_20261029_130112
```

새 run 프로젝트는`as2piper-low`, ID는`as2piper_hw_gaitfix_vbcsampler_scratch_<UTC시각>`이다. 다른 실험과500/520 진단 체크포인트는 보존한다.

재현용 진단:

```bash
cd /data/team/users/suhwan/vbc-as2piper/low-level-as2piper-hw
export PYTHONPATH="$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python"
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
python profile_goal_runtime.py --mode screened --num-envs 6144 --steps 24 \
  --output audit_hw/goal_runtime_20261029/profile_repeat.json
python audit_ee_range.py --candidates extended_small_orn --random-paths 512 \
  --seed 20261030 --tilt-grid 5 --path-samples 17 --extra-offline-iterations 192 \
  --output audit_hw/goal_runtime_20261029/range_repeat.json
```

측정 원본은`audit_hw/goal_runtime_20261029`에 보존한다. 실제 W&B 실행의 교체 여부는 이 문서의 진단 결과와 구분한다.
