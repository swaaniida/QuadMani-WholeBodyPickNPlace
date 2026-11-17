# AS2 + PiPER plate-only 전진 배치 복제본

`low-level-as2piper`를 독립 복제했다. 기존 코드·URDF·실행 스크립트·학습과 B1 baseline `/data/team/users/suhwan/vbc-legacy`는 수정/중지하지 않았다. 로그·체크포인트·이전 audit·build cache는 복제하지 않았으며, 코드·메시·PPO·로컬 의존성은 별도 파일로 복사했다. 원본 55개 코드/URDF 파일의 보존 기록은 `audit_forward/source_before.json`과 `validation.json`에 있다.

## 실행

```bash
cd /data/team/users/suhwan/vbc-as2piper
bash low-level-as2piper-forward/run_as2.sh
```

- Task: `as2piper_plate_forward`
- W&B project: **as2piper-low**, online 기본.
- Run name: `plate_forward_armbase_<UTC 날짜_시간>_<PID>`
- Run ID: `as2piper_forward_armbase_<UTC 날짜_시간>_<PID>`
- 매 실행 새 ID, resume 기본 false. 이전 744차원 체크포인트를 이어 쓰지 않는다.
- 본 학습: **6144 env / rollout 24 step / 45000 iterations**, 기존과 동일.
- 로그/checkpoint: 이 복제본 내부 `logs/as2piper-low/<run name>/`.
- 이번 작업에서는 짧은 검증만 실행했으며 장시간 학습은 시작하지 않았다.

직접 이름을 지정하려면 새 ID를 사용한다.

```bash
WANDB_NAME=plate_forward_armbase_001 \
WANDB_RUN_ID=as2piper_forward_armbase_001 \
bash low-level-as2piper-forward/run_as2.sh
```

## 기구 모델

`prepare_asset.py`는 이 복제본의 mesh-seated plate-only 원본에서 새 `as2piper_plate_forward_training.urdf`를 만든다. 판은 **140×120×6mm**, 판 중심은 body x=-20mm로 유지한다. 프로파일·연장판을 추가하지 않는다.

PiPER만 판 위에서 +25mm 옮겨 arm base를 **AS2 body `[+0.005,0,0.0820284782] m`**에 둔다. PiPER base mesh 앞끝은 x=45mm, 판 앞끝은 x=50mm라 nominal 5mm 여유다. 기존 판/5mm 여유 가정 아래의 전진 한계이며 실제 볼트 홀·강도 한계가 확인된 값은 아니다.

판 형상·질량·관성·지지 footprint, PiPER 방향, UMI base/손가락 링크는 유지한다. 발 4개와 nominal TCP fixed joint는 Isaac Gym collapse에서 보존한다. **20 DOF / 18 policy actions / gripper 열림 ±0.049m / arm Jacobian 6×6** 유지.

접합 검사: plate–PiPER gap=0, plate–AS2 최소 gap≈0, 메시 관통=0. 외장은 완전 평면이 아니므로 샘플 최대 gap≈2.108mm, 1mm 이내 근접 비율≈98.89%. 외장 shell과 실제 구조 프레임/볼트 체결의 구분은 여전히 현물/CAD 확인이 필요하다.

## 좌표계와 EE 목표

명령 원점은 실제 **piper_base_link**다. AS2 body→arm fixed rotation은 identity이며 body의 **roll/pitch/yaw 전체**를 사용한다.

```
p_arm_world = p_body_world + R_body_world * [0.005, 0, 0.0820284782]
p_goal_world = p_arm_world + R_body_world * p_goal_arm
q_goal_world = q_body_world * q_goal_arm
```

관측도 동일한 변환의 역변환으로 구한다. yaw만 적용한 mount 위치와 full-body inverse rotation을 섞던 불일치를 제거했다. 위치 3차원은 arm-local cartesian goal, 방향 3차원은 **home 대비 spatial rotation vector(arm-base 축 표현)**다. 절대 Euler가 home 부근 ±π에서 급변하는 문제를 피하며 방향 명령을 0으로 숨기지 않는다. 기본 gait 관측도 원본 B1 실제 실행에 맞게 활성화해 proprio=71, 전체=799다. 이전 AS2 744 관측과 의미·차원이 다르므로 새로 학습한다.

큰 B1 구면 범위와 독립 orientation randomization은 활성 sampler에서 사용하지 않는다. **PiPER 관절 공간의 256개 endpoint와 home↔endpoint 경로**를 검증한 뒤 FK로 위치와 방향을 함께 만든다. home→endpoint→home 순서로 이동해 서로 다른 endpoint 사이의 미검증 경로를 피한다. 관절 공간을 cubic smoothstep으로 보간하고 FK하므로 경로상의 목표 pose는 같은 관절 해를 갖는다. 실제 제어는 기존 DLS IK+PhysX position drive를 유지한다. 정책이 팔을 직접 제어하는 구조로 바꾸지 않았다.

| 항목 | 활성 범위/값 |
|---|---|
| Endpoint arm-local x | 0.340169–0.569800m |
| Endpoint arm-local y | −0.214654–+0.214335m |
| Endpoint arm-local z | 0.060971–0.399789m |
| Sampling envelope | x=[.34,.58], y=[−.22,.22], z=[.06,.40]m |
| Home arm q | `[0,1.1,-0.55,0,0,0]`rad |
| Home TCP arm-local | 약 `[0.390415,0,0.175139]`m |
| Joint proposal q1 | [−.45,+.45]rad |
| q2 / q3 | [.55,1.65] / [−1.20,−.04]rad |
| q4 / q5 / q6 | [−.35,.35] / [−.45,.45] / [−.35,.35]rad |
| Trajectory / hold | 기존 [1,3]초 / [.5,2]초 유지 |

위 cartesian bounds는 점군의 외접 범위이며 내부를 독립 균등분포로 sampling하는 상자가 아니다. 활성 sampler는 항상 bank의 joint endpoint를 선택한다. 버퍼 호환을 위해 config에 남긴 sphere ranges는 활성 sampler에서 사용하지 않는다. 정확한 metadata는 `resources/commands/arm_commands.json`, bank는 `.npz`에 있다.

Locomotion에서는 목표가 arm base를 따라가며 몸통이 기울어도 arm-local 목표를 유지한다. 고정 물체에 접근하는 후속 전신 제어용으로 `set_world_ee_goal(env_ids,position,quaternion_xyzw)`를 제공한다. world latch 이후에는 몸통 tilt로 목표가 움직이지 않는다. 이 외부 목표는 bank 검증을 우회하므로 호출자가 reach/충돌/지형을 검증해야 한다. 이번 bank는 **locomotion과 manipulation 시작 자세용**이며 바닥 파지·dragging 정책이나 카메라 인식 학습 결과가 아니다.

## Reset과 보존한 설정

- AS2 neutral legs: hip=0/thigh=.8/calf=−1.6rad. neutral grounded body z=.3244069m.
- 기존 다리 reset noise를 유지한다. 각 다리의 실제 URDF FK와 foot sphere 반경28mm로 최소 발바닥이 terrain origin 기준 2mm 위에 오도록 root z를 계산한다. rough terrain 각 지점에 완전히 접지한다는 보장은 아니다.
- Arm의 reset 곱셈 noise만 제거하고 home에서 초기화한다. **동일한 FK 위치·방향**에서 초기 목표를 시작하며 첫 trajectory는 home을 유지해 초기 목표 점프를 없앤다.
- Reward 공식·scale·PPO는 동일. 유일한 reward parameter 변경은 기하학적 **base_height_target=.55→.3244069m**다. feet height 등 다른 reward target은 변경하지 않았다.
- Terrain, domain randomization, 보행 속도 command, 노이즈, PD gain, URDF effort 상한은 유지한다. 추가 mass0–15kg/CoM±.15m 같은 큰 randomization도 남아 있어 이번 수정만으로 수렴/안전성을 보장할 수 없다.
- 원본 contact shaping의 음수 metric×음수 weight=양수 reward도 유지한다. gait 관측 활성화로 값은 0 고정에서 벗어나지만 부호의 타당성 문제를 수정한 것은 아니다.
- 상속한 step이 action을 in-place로 mask하므로 **새 subclass 입구에서 clone**해 PPO의 sampled action/log-prob pair를 보존한다. PPO loss와 실제 arm action mask는 바꾸지 않았다.
- 실제 arm torque 읽기 누락, PiPER 실모터 한계, 카메라 질량 등 기존 미검증 항목은 이번 수정으로 해결되지 않았다.

## 검증과 재실행

기존 MuJoCo 의존성 경로를 추가해 offline 검증한다.

```bash
cd /data/team/users/suhwan/vbc-as2piper
umask 0002
export PYTHONPATH="$PWD/mujoco_mount_compare/.deps"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python low-level-as2piper-forward/validate_forward.py
```

URDF/command 범위를 바꾼 경우에만 bank를 재생성한다.

```bash
PYTHONPATH="$PWD/mujoco_mount_compare/.deps" \
python low-level-as2piper-forward/build_command_bank.py
```

실행 시 bank의 asset SHA256을 대조해 다른 URDF에 오래된 bank를 사용하는 일을 막는다.

- 256경로×129점=**33,024자세**의 독립 검사에서 self-collision proxy의 0.5mm 초과 관통 없음.
- Torch FK–MuJoCo 최대 위치 오차 **0.376µm**, quaternion dot 오차6.91e−8.
- nominal 자세에서 경로 중 최대 arm 중력 torque **6.677Nm**. screening 9Nm는 기존 demo의 가정이며 실제 모터 사양이 아니다.
- 실제 GPU PhysX **8env,350-step rollout,reset,PPO 2 update / 384 transitions** 실행. 위치·방향 FK, finite tensor, sampled action 보존, world 목표 latch 검사 통과.
- Torch FK–PhysX 최대 TCP 위치 차이 **2.13µm**, quaternion dot 오차2.39e−7.
- gripper 최대 위치 오차 **3.226mm**, 검사 기준5mm 이내.
- 최종 기록: `logs/as2piper-low/forward_smoke_release_20261028/model_2.pt` 및 `smoke_contracts.json`. 검사용 출력이며 본 학습 resume에는 쓰지 않는다.

smoke 재실행:

```bash
WANDB_MODE=disabled AS2_SMOKE=1 \
WANDB_NAME=forward_smoke_check WANDB_RUN_ID=forward_smoke_check \
bash low-level-as2piper-forward/run_as2.sh --max_iterations 2
```

설정 비교는 새 run과 같은 PYTHONPATH에서 `python check_original_settings.py`로 실행한다. `audit_forward/settings_diff.json`에 B1 config와의 차이, height target 이외의 reward 및 terrain/DR/commands 등의 일치를 기록한다. 복제된 옛 `audit_port.py`는 이전 포팅용이며 이번 변경의 허용 조건에는 대응하지 않는다.

`audit_forward/command_workspace.png`는 실제 FK 경로와 head 기준 측면 투영이다. nominal 다리 자세에서 검사했으며 움직이는 다리·rough terrain·몸통 tilt·외력 아래의 동적 충돌이나 케이블/카메라 시야까지 보장하지 않는다. 유한 샘플은 연속 충돌의 증명이 아니다. 구현 diff는 `audit_forward/design.patch`에 있다.
