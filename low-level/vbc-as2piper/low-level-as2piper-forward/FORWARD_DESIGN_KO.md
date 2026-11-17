# AS2 PiPER 전진 plate-only: B1 구면 pose sampling

기존 `low-level-as2piper`와 B1 baseline은 수정/중지하지 않았다. 이번 변경은 독립 복제본 `low-level-as2piper-forward`에만 적용했다. 이전 관절 경로 bank 방식은 폐기하고 `audit_forward/before_spherical`에 보관했다. 활성 코드에는 bank 로드/선택이나 joint-space 목표 보간이 없다.

## 실행

```bash
cd /data/team/users/suhwan/vbc-as2piper
bash low-level-as2piper-forward/run_as2.sh
```

Task=`as2piper_plate_forward`, W&B project=**as2piper-low**. Run name은 `plate_forward_spherical_<UTC timestamp>_<PID>`, ID는 `as2piper_forward_spherical_<UTC timestamp>_<PID>`. 매 실행 새 ID를 쓰며 resume=false다. 6144env/rollout24step/45000iterations를 유지했다. 이번에는 짧은 smoke만 실행했고 장시간 학습은 시작하지 않았다.

## 장착과 좌표계

판은 140×120×6mm, 중심 x=-20mm 그대로다. PiPER만 +25mm 옮겨 실제 arm base를 AS2 body **[.005,0,.0820284782]m**에 둔다. 베이스 앞끝과 판 끝 사이 nominal 5mm 여유다. 실제 볼트 홀·가장자리 강도까지 승인한 최대치는 아니다. 판·UMI 형상·질량·관성은 바꾸지 않았다. 20 DOF/18 actions/그리퍼 열림±.049m를 유지한다.

구면 중심은 실제 arm origin이다. 몸통 roll/pitch/yaw 전체로 mount offset을 변환한다. **구면 샘플링 축은 B1과 같은 yaw 정렬·중력 기준 축**이다.

```
origin_world = body_position + R_body * [.005, 0, .0820284782]
goal_world = origin_world + R_body_yaw * sphere2cart(radius,pitch,yaw)
```

따라서 몸통을 숙이면 arm origin은 움직이지만 구면 축이 몸통 pitch를 따라 함께 기울지는 않는다. 기존 B1의 고정 world z=.7m 중심 대신 실제 AS2 arm 높이를 사용한다. 관측 위치는 위 world 목표를 실제 arm origin과 body 회전의 역변환으로 표현한다. 방향 관측 3칸은 **B1 원본처럼 0**, gait 관측은 켜서 총799차원이다. 앞서 추가한 rotation-vector 관측과 world-latch API는 제거했다.

## B1과 같은 목표 생성 알고리즘

- Radius/pitch/yaw 각각 독립 균등 sampling.
- Orientation delta roll/pitch/yaw도 각각 독립 균등 sampling.
- 이전 구면 목표→새 구면 목표를 **원본 linear torch.lerp**로 보간한다.
- `default_yaw=atan2(goal_y,goal_x)`, `default_pitch=-sphere_pitch+arm_induced_pitch`에 orientation delta를 더한다.
- 기존 trajectory [1,3]초, hold [.5,2]초와 timer 처리를 유지한다.
- IK는 실제 기존 팔 제어와 추가 후보 검증에 사용하며, IK/FK 결과를 목표 sampling 분포로 대체하지 않는다.

기반 클래스의 `_resample_ee_goal_sphere_once`, `_resample_ee_goal_orn_once`, `_resample_ee_goal`, `_update_curr_ee_goal`, `_collision_check`, `compute_reward` AST가 기존 AS2 복제 원본과 동일함을 검사했다. 새 subclass는 원본 sampling 함수를 상속하고 충돌 검사/실패 처리와 초기화만 보완한다.

| 파라미터 | AS2 PiPER 값 |
|---|---|
| 구면 radius | **.38–.52m** |
| 구면 pitch | **−.25–+.60rad** (약 −14.3–34.4°) |
| 구면 yaw | **−.40–+.40rad** (약 ±22.9°) |
| Orientation delta r/p/y | 각각 **−.15–+.15rad** (약 ±8.6°) |
| initial start/end sphere | [.427898468686,.421686284191,0] |
| arm_induced_pitch | **2.455216137744rad** |
| orientation_roll_offset | 0 |

PiPER TCP 축이 Z1과 다르므로 arm_induced_pitch를 실제 초기 FK에 맞췄다. 초기 default pitch=2.03352985rad이며 초기 TCP 위치와 방향이 기존 C 자세 `[0,1.1,-.55,0,0,0]`의 FK와 맞는다. start=end는 설정값으로 초기 hold를 만든 것이며 sampling/interpolation 알고리즘을 바꾼 것이 아니다. 구면 pitch 범위만 부호를 바꾸고 이 방향 offset을 그대로 두면 잘못된 orientation을 요구할 수 있다.

## 충돌 고려

**B1도 충돌 검사를 한다.** 원본은 구면 경로의 EE 점 10개가 금지 AABB 또는 underground limit에 들어가는지 검사한다. arm 전체 링크의 형상 충돌 검사는 아니다.

AS2에서는 중앙 장착으로 앞쪽 머리/몸통과 간섭할 수 있어 다음을 추가했다.

1. 기존 10점 EE 경로 AABB/underground 검사를 AS2 크기에 맞춰 유지한다.
2. 구면 경로 시작/중간/끝 **3개 pose**를 현재 팔 상태에서 시작한 배치 DLS IK로 검증한다. 관절 한계, 위치오차12mm, 방향오차5°를 검사한다.
3. 해당 해에서 **PiPER link2–6, 플랜지·그리퍼·열린 양 손가락**의 URDF collision 형상을 감싸는 OBB를 검사한다. AS2 전체 body/head STL AABB와 15축 SAT로 충돌을 판정한다. Arm box 2mm, body box 5mm 여유를 준 보수적 proxy다. 실제 mesh가 비어 있는 공간도 금지할 수 있다.
4. EE 목표 높이가 현재 terrain origin+30mm보다 낮으면 거절한다.
5. 원본처럼 위치 후보를 최대10회 재추출한다. **10회 모두 실패하면 이전 목표와 이전 orientation delta를 유지**한다. 원본은 마지막 거절 후보가 남을 수 있어서 이 부분만 충돌 처리 예외로 보완했다.

실제 목표 생성은 계속 독립 구면/방향 sampling이다. 충돌 조건 때문에 수락 분포가 제안 균등분포와 달라지는 것은 rejection의 결과다. 근사 IK 실패를 기구학적 도달 불가의 증명으로 해석하면 안 된다.

고정 mount/link1의 의도된 연결은 검사에서 제외했다. 동작 중 다리와 팔의 충돌, 팔 링크끼리의 모든 자기충돌, 케이블·물체·카메라, 연속 궤적 전체의 안전까지 증명하는 검사는 아니다. 물리 엔진의 원래 collision은 계속 활성화되어 있다. 원본보다 검사가 추가되어 목표 resampling 비용은 증가한다.

## 다른 설정과 검증

이전 전진 복제본의 하드웨어 보정은 유지한다: neutral legs hip0/thigh.8/calf−1.6, 발 FK 기반 초기 높이, 초기 팔 C 자세와 일치하는 EE 목표, body height target=.3244069m. 그 외 reward 공식·가중치, terrain/DR/보행command/PD/PPO는 유지한다. 기존 contact shaping 부호, 큰 DR 범위, 실제 arm torque 읽기 누락은 별도 미해결 항목이다. 새 subclass의 action clone은 PPO sampled action의 in-place 변경을 방지하는 기존 보정이다.

`audit_forward/validation.json`과 `command_workspace.png`는 **구면 방식의 현재 결과**다. 독립 제안1200개 중 guard 수락766개, IK 조건 거절152개, body/head proxy 거절331개(거절 사유 중복 가능). 수락한 endpoint의 별도 MuJoCo 충돌은0건이었다. 이것은 nominal stance의 표본 검증이며 모든 목표/움직이는 몸통의 안전 보장은 아니다. 원본55개 파일의 해시도 그대로다.

최종 GPU smoke 기록은 `logs/as2piper-low/forward_spherical_release_20261028/`의 `smoke_contracts.json` 및 `model_2.pt`다. 8env,350step rollout,reset,PPO2update를 검사한다. 미학습 zero-action 로봇이 넘어져 손가락을 누르는 경우는 전체 gripper 최대 오차로 별도 기록하고, upright 상태의 열림 유지 오차를 구분해 검사한다. 이 체크포인트는 학습된 정책이 아니며 본 학습 resume에 쓰지 않는다.

```bash
cd /data/team/users/suhwan/vbc-as2piper
umask 0002
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH="$PWD/mujoco_mount_compare/.deps" \
python low-level-as2piper-forward/validate_forward.py

# URDF 변경 시에만 collision proxy를 재생성한다.
PYTHONPATH="$PWD/mujoco_mount_compare/.deps" \
python low-level-as2piper-forward/build_collision_guard.py

WANDB_MODE=disabled AS2_SMOKE=1 \
WANDB_NAME=spherical_smoke_check WANDB_RUN_ID=spherical_smoke_check \
bash low-level-as2piper-forward/run_as2.sh --max_iterations 2
```

실행 시 collision metadata의 asset SHA256을 대조한다. `audit_forward/design.patch`는 복제 원본 대비 현재 코드 변경이며, 이전 bank 방식은 사용하지 않는다.
