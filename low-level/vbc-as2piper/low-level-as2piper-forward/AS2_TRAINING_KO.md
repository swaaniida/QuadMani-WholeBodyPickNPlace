# AS2 + PiPER plate_only low-level 포팅

> 아래는 최초 포팅 당시의 기록이다. 2026-10-28 사용자 요청으로 환경 수·domain randomization·EE 목표 분포·PD·reward 설정을 원본 기준으로 복구했다. **현재 설정과 실행 명령은 [ORIGINAL_RUN_KO.md](ORIGINAL_RUN_KO.md)를 따른다.** 아래의 1024 env, W&B disabled, 높이 목표 0.30m 등은 현재 기본값이 아니다.

`low-level/` 전체를 이 디렉터리로 복제하고 PPO 라이브러리도 `rsl_rl_local/`로 복제했다. 원본 `low-level/`과 `/data/team/users/suhwan/vbc-legacy`의 코드를 수정하지 않는다. 기존 체크포인트는 재사용하지 않으며 로그도 이 복제본 안에 생성한다.

## 학습 실행

```bash
cd /data/team/users/suhwan/vbc-as2piper
bash low-level-as2piper/run_as2.sh --exptid plate_C_001 --num_envs 1024
```

기본값은 W&B disabled이며 로컬 checkpoint/log만 쓴다. 이후 W&B 사용을 원하면 실행 환경에서 `WANDB_MODE=online`을 지정한다. 로그인/인증 파일을 읽거나 변경하는 작업은 수행하지 않았다. 학습 기본값은 45,000 iterations이며 `--max_iterations`로 바꿀 수 있다. 새 experiment ID를 사용한다. `AS2_SMOKE=1`은 아래 검사 전용이며 본 학습에서는 지정하지 않는다.

검증 재실행:

```bash
AS2_SMOKE=1 bash low-level-as2piper/run_as2.sh \
  --exptid smoke_check_new --num_envs 8 --max_iterations 2
python low-level-as2piper/audit_port.py
```

설치 없이 현재 환경에서 실행한다. PYTHONPATH는 이 복제본, 복제한 RSL-RL, 프로젝트 안의 기존 Isaac Gym 순으로 구성한다. 지형 생성에 필요한 `pydelatin==0.2.8` 등은 이 디렉터리 `.deps/`에만 설치했다. Torch extension build cache도 이 디렉터리 안에 둔다. 시스템 Python과 기존 학습 패키지는 변경하지 않았다.

## 포팅한 연결부

- Asset: `resources/robots/as2piper/urdf/as2piper_plate_training.urdf`. 원본 mesh-seated plate_only URDF 사본에서 발 4개와 UMI nominal TCP fixed joint에 `dont_collapse=true`만 추가한다. joint origins, masses, inertias, collisions, mesh transforms는 보존한다. 매번 `prepare_asset.py`로 재생성한다.
- Isaac Gym 실제 DOF 순서: FL/FR/RL/RR 각 hip/thigh/calf, PiPER joint1..6, gripper joint1/2. 순서가 바뀌면 실행을 중단한다. 고정 발 링크는 반드시 별도 rigid body로 남겨 4 force sensors를 만든다.
- 20 DOF / 18 actions / 744 observations / 6×6 arm Jacobian. 원본 reindex는 원래와 동일한 native FL/FR/RL/RR 순서에서 정책 FR/FL/RR/RL로 바꾸므로 유지했다. 물리 토크 20차원과 정책 출력 18차원을 구분한다.
- 원본 제어 구조 보존: policy arm action 6개는 마스킹되고, 팔은 EE command에 대한 DLS IK와 PhysX position drive로 움직인다. 따라서 이 포팅은 18차원 인터페이스의 기존 알고리즘이며 팔 6개를 정책이 직접 torque-control하도록 바꾼 것이 아니다. 그리퍼 2개는 정책·관측 joint 벡터에서 제외하고 +0.049/-0.049m 열림 위치로 별도 유지한다.
- 초기 팔 자세는 후보 C `[0,1.1,-0.55,0,0,0]`, 다리는 hip=0/thigh=0.8/calf=-1.6, 초기 root z=0.33m다. env 생성 직후 명시적으로 reset해 URDF zero pose에서 학습이 시작되지 않게 했다. reset 관절각은 URDF 한계 안으로 제한하고 손가락은 열림 목표로 고정한다.
- AS2 leg PD 90/3.5, PiPER position drive 80/8, gripper 1800/35. PiPER effort 상한은 이전 모델의 보수적 simulation 값인 9Nm를 적용한다. 이 값은 하드웨어 실측 인증 값이 아니다. 다리/손가락 effort·velocity 한계는 URDF를 사용한다.
- 실제 arm mount offset `(-0.02,0,0.0820284782)m`, AS2 base body index를 이름으로 찾는 mass/CoM randomization, UMI nominal TCP body의 Jacobian과 pose로 교체했다. EE orientation은 PiPER TCP frame에 맞춰 기본 roll offset=0으로 보정했다.
- EE spherical goal center는 terrain 기준 z=0.45m, base-relative x=-0.02m, radius 0.25~0.48m, pitch -0.65~0.8rad다. 기존 B1의 큰 goal workspace와 box keep-out 영역을 AS2 크기에 맞췄다. 이는 학습 command 분포 후보이며 모든 orientation/goal의 reachability를 보장하지 않는다.
- Domain randomization의 추가 base mass는 0~3kg, CoM 변화는 x ±30mm/y,z ±20mm로 AS2 크기에 맞춰 줄였다. 나머지 알고리즘·reward 가중치는 보존했다.

## reward 보존 범위

`maniploco_rewards.py`와 `compute_reward()`의 AST, B1 원본 config, 복제한 PPO 코드가 원본과 일치함을 `audit/preservation.json`에서 확인한다. 모든 reward scale/arm scale, reward 공식, PPO loss는 유지했다.

단 하나의 하드웨어 길이 목표는 수정했다: `rewards.base_height_target=0.55 → 0.30m`. AS2에서 0.55m 몸통 높이를 계속 요구하지 않도록 한 보정이다. 다른 reward 설정(예: feet_height_target=0.3)은 요청대로 유지했으며 AS2에서의 학습 성능 튜닝은 하지 않았다. 학습은 실행 가능하지만 원본 reward의 AS2 성능 최적성을 의미하지 않는다.

## 검증 결과

실제 CUDA PhysX GPU pipeline에서 8 env, 2×2 rough terrain으로 다음을 통과했다.

- plate_only 로드: robot 26 rigid bodies, 20 DOF, 4 feet, nominal TCP body 유지.
- observation `[8,744]`, Jacobian `[8,6,6]`, finite tensors.
- 100-step zero-action rollout와 reset; 그리퍼 목표 오차 최대 약 0.641mm.
- PPO rollout 24 steps × 8 env × 2 iterations = 384 transitions, 실제 역전파/optimizer step 및 model_2.pt 저장.
- `logs/as2piper-low/smoke_final_20261028/smoke_contracts.json`, `model_2.pt`가 증거다. 장시간 수렴 학습은 실행하지 않았다. 기본 1024 env 규모의 장시간 메모리·성능은 별도 확인 대상이다.

원본 mesh-seated의 실제 structural support/bolt fastening, 카메라 질량/외형, 실제 모터 전류·토크·지연, base/EE tracking 성능과 ground reach 정책 수렴은 아직 검증되지 않았다. 이번 환경은 가짜 base servo나 soft-grasp 외력 없이 Isaac Gym의 중력·접촉·모터로 실행한다.

`audit/port.patch`에서 복제 원본 대비 코드 diff를 볼 수 있다. 개발 저장소의 해당 디렉터리가 untracked이므로 일반 `git diff --stat`에는 새 복제본이 나타나지 않는다.
