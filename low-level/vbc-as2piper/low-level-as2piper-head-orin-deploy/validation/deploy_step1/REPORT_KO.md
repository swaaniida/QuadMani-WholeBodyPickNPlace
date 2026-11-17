# 배포 수정 1단계: EE 가상 구면 중심

## 적용 범위

- 원본: `low-level-as2piper-head-orin-nocontact` 보존.
- 복제본: `low-level-as2piper-head-orin-deploy`.
- 현재 손목 카메라·거치대 포함 URDF, resources, 학습 코드 및 로컬 의존성을 복제. 기존 experiments/checkpoint는 복제하지 않고 원위치 보존. 과거 validation 자료는 참고 이력으로 복제했으며 이번 검증을 의미하지 않음.
- 기존 task 이름은 복제본 내부에서 유지. Python 모듈 검색 경로와 `LEGGED_GYM_ROOT_DIR`는 복제본 경로를 사용. 신규 학습 run/task 이름은 학습 승인 전에 구분할 예정.
- 실제 수정: `as2_head_orin_config.py`, `manip_loco.py` 두 파일.
- 학습·optimizer update·W&B run 생성 없음.

## 원인과 수정

기존 중심:

`center = base_position + R_full_body * arm_mount + R_yaw * center_offset(0)`

따라서 몸통 높이와 pitch/roll에 따라 중심과 EE 목표가 움직임.

수정 중심:

`center = [base_x, base_y, local_ground_z] + R_yaw * [0.22435, 0, 0.4563069024246581]`

`target_world = center + R_yaw * sphere_to_cart(command)`

가상 높이는 명목 base 높이 0.3244069024246581m + mount 높이 0.1319m. 새 물리 치수나 최적 작업공간 추정치가 아니라 기존 중립 자세를 기준으로 정한 좌표 원점임. reset 시작 높이 0.3264069024246581m와 명목 안정 높이의 2mm 차이는 기존대로 유지.

평지에서 B1 원본의 XY/yaw 가상 중심 방식과 동일한 불변성을 가짐. rough에서는 기존 지형 그리드의 몸통 아래 국소 높이를 사용. 몸통 방향에 따라 이동하는 높이 probe 평균을 사용하지 않음. 이 지형값은 시뮬레이션의 지면 참조이며 실제 배포에는 지면 추정/물체 목표 높이의 동등한 입력 경로가 필요함.

실제 Arm Base 기반 observation 변환과 IK, TCP, 초기 관절각, orientation observation, action/observation 차원, 보상과 PPO/DAgger/ROA는 수정하지 않음. 초기 구면 목표와 범위도 이번에는 유지. Collision check와 디버그 렌더는 기존 공통 중심 함수를 통해 새 중심을 참조.

## 검증

- 실제 center/terrain/goal-update 메서드를 AST로 불러 CPU에서 실행. PhysX 환경 생성이나 물리 rollout은 수행하지 않음.
- 몸통 높이 0.20~0.40m, roll ±0.5rad, pitch ±0.65rad, 1,000개 자세: 중심 불변.
- 바닥 z=0.02m 및 상단 z=0.80m의 동일 가상 목표: 최대 높이 오차 1.74e-17m.
- 실제 goal update 위치·방향 출력도 몸통 높이/roll/pitch에 대해 불변.
- yaw 90도 및 XY 변위에 대한 좌표 변환 확인.
- rough 그리드 높이 +0.20m: 중심 높이도 정확히 +0.20m. 몸통 crouch와 구분.
- 실제 Arm Base 높이는 위 자세군에서 약 0.445m 변화: 가상 중심과 실제 Arm Base를 분리한 결과 확인.
- 원본 234개 소스/자산 SHA256 유지. 기존 226개 checkpoint의 크기·mtime 유지.
- 복제본의 Reward/Actor/Critic/IK/관측/PPO/자산 파일은 원본과 byte 동일. 코드 diff는 `step1.patch`.

## 경계와 다음 검토

- XY/yaw는 원본 B1처럼 몸통 이동을 따름. 절대 world 좌표 물체를 집을 때 XY까지 고정하는 별도의 object/world target 모드는 이번에 추가하지 않음.
- 거친 지형에서 XY 이동으로 국소 지면 높이가 바뀌면 중심 높이도 바뀜. 동일 XY에서 몸통 crouch/tilt만 하는 경우에는 불변.
- 바닥/상단 높이 검사는 좌표 불변성 검사이며 해당 목표의 IK·충돌·토크 타당성 증명이 아님. 샘플링 범위는 기존의 좁은 범위 유지.
- 다음 2단계에서 카메라 포함 자산의 Leg+Arm IK, collision, CoM 및 확인된 torque 제약을 먼저 검사하고, 위치·그리퍼 방향을 연동하는 workspace 적용안을 검토받을 예정. 미확인 모터 토크를 확정 제약으로 취급하지 않음.
- 3~8단계는 미착수. Reward scale·DR·gain·limit·delay 변경 없음.

## 재실행

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/usr/bin/python3 low-level-as2piper-head-orin-deploy/validation/deploy_step1/verify_frame.py
```

Git은 기존 worktree의 dubious ownership 오류로 status/diff 실행 불가. 글로벌 Git 설정을 변경하지 않고 원본 SHA256 manifest 및 unified diff로 보존·변경 범위를 검증.
