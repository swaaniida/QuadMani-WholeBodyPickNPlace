# 7단계: 기본 PiPER 공식 SDK 대조 및 구동 경로 수정

## 결론

**확인된 코드 경로 오류와 위치 명령 속도 제한을 복제본에 수정했다.** 실기 펌웨어, 연속/피크 토크·열 한계, position drive의 실제 gain·가속도·지연은 확정하지 않았다. 해당 값이 확인되기 전 배포 적합성 및 torque-feasible workspace 판정은 보류한다. 학습·optimizer update·실기 통신은 수행하지 않았다.

작업 위치: `low-level-as2piper-head-orin-deploy`. 기존 45k 소스·실험·checkpoint에 이번 작업의 쓰기 없음. No-contact/755 관측·737 배포 입력·18 action·20DOF 설계와 Reward·PPO·DAgger·ROA 유지.

## 공식 근거와 해석

공식 저장소를 commit에 고정해 필요한 파일만 저장했다. SDK 설치나 전체 저장소 clone 없이 문서·소스만 취득했다. 경로·commit·SHA256은 `official/sources.json` 참조.

- [PiPER SDK 구동·제한·조회 구현](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/interface/piper_interface.py)
- [PiPER MIT 메시지](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/piper_msgs/msg_v2/transmit/arm_joint_mit_ctrl.py)
- [공식 pyAgxArm 펌웨어별 구동 범위](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/docs/piper/firmware_reference.md)
- [기본 PiPER 관절 범위](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/pyAgxArm/api/constants.py)

판정:

| 항목 | 확인된 근거 | 적용/보류 |
|---|---|---|
| 기본 PiPER 각도 범위 | 공식 preset과 현재 6관절 URDF 범위 일치 | 기존 각도 유지 |
| 위치 제어 최대 속도 설정 | SDK 설정 범위 0~3rad/s | 기존 5→3rad/s; 실기에서 실제 설정이 더 낮으면 추가 축소 필요 |
| 속도 raw 단위 | S-V1.5-8 전후로 0.001→0.01rad/s | 조회 도구에서 버전별 구분, 미상 버전은 환산 보류 |
| 가속도 | SDK 조회 가능 | 실제 기기값 미확정. 예제 기본 인자를 실측으로 간주하지 않음 |
| MIT gain | 명령 프로토콜 값 및 예시가 존재 | PhysX 등가 gain으로 직접 복사하지 않음 |
| MIT 토크 | SDK/펌웨어별 wire 폭·환산 계수가 다름 | 명령 표현 범위와 실제 모터 정격을 구분. 미확정 실기 한계 적용 없음 |

특히 구형 SDK의 public docstring에 남은 ±18과 실제 내부 encoder의 ±8도 서로 다르다. 최신 공식 문서는 펌웨어별 별도 환산을 설명한다. 따라서 문서의 한 숫자를 골라 URDF effort에 넣는 방식은 사용하지 않았다. MIT의 속도 표현 범위 ±45도 위치 제어의 3rad/s 설정 상한과 같은 의미가 아니다.

## B1 원본 → 기존 AS2 → 수정본

읽기 전용으로 `/data/team/users/suhwan/vbc-legacy/low-level/legged_gym/envs/manip_loco/manip_loco.py`와 현재 복제본을 대조했다.

| 경로 | B1 및 기존 AS2 | 이번 복제본 |
|---|---|---|
| Arm drive | `DOF_MODE_POS`, stiffness 400 / damping 40 하드코딩 | arm/gripper 별도 config로 이동. 수치는 미보정 legacy 값으로 유지 |
| `control.stiffness['piper_joint']=5`, damping=0.5 | 실제 arm position drive gain이 아님 | 기존 필드 유지하되 실제 gain 설정과 구분 |
| Arm Motor Strength DR | 0.7~1.3을 뽑아 action 경로에 곱하지만 arm action/torque가 0이라 actual drive에 무효 | actual PhysX arm stiffness/damping에 동일 표본 배율 적용 |
| Drive effort 포화 | URDF arm effort=100Nm | 수치 유지하되 실기 미확인 상태를 config에 명시 |
| Arm 속도 | URDF 5rad/s, IK 목표 변화량 제한 없음 | URDF·PhysX property 3rad/s, 전송 목표 변화율 3rad/s 제한 |
| Arm 가속도 | 제한 모델 없음 | 미확정 None. 임의 값 추가 안 함; 설정 시 무검증 적용을 거부 |
| Leg action delay | config=3이지만 실제는 초기 0 → 후반 1-step | config=1, warmup=240000 policy step으로 실제 동작과 일치. 임의 설정 step 인덱싱도 정상화 |
| Arm 명령 지연 | 다리 delay 경로를 우회 | 별도 필터/설정으로 분리, 기본 0-step 유지. 실기 지연 미측정 |
| Arm 실제 토크와 `self.torques` | 명시적 force tensor의 arm 성분은 0. position drive가 별도로 힘 생성 | 의미 유지. 해당 0을 실제 arm motor torque라고 보고하지 않음 |

### DR 수정의 정확한 의미

- 팔 6관절에만 기존 표본을 사용한다. stiffness=400×s, damping=40×s, s∈[0.7,1.3].
- 따라서 가능 범위는 stiffness 280~520 / damping 28~52다. PhysX property readback으로 적용을 확인했다.
- **이는 drive 응답 gain DR**이며, 식별된 토크 상수·전류 제한·마찰·backlash 모델이 아니다. effort 포화는 그대로 유지한다. 토크 능력 전체를 0.7~1.3배 물리 모델로 구현했다고 주장하지 않는다.
- 기존 다리 Motor Strength는 action amplitude에 곱해지는 경로를 유지했다. 다리 구동 모델까지 바꾸는 범위 확장은 이번에 하지 않았다.
- 관측/privileged 차원을 추가하지 않았다. 팔 DR이 새 실물 센서 입력을 요구하지 않는다.

### 목표 변화율·reset

기존 Arm DLS IK 해를 바꾸지 않고 IK 결과 뒤에 command filter를 적용했다. 이전 **전송 명령**과 다음 명령의 관절 차이를 3×dt로 제한한다. dt=0.02s에서 최대 0.06rad/step이다. 관절한계를 넘는 목표도 제한한다. reset된 환경만 명령·delay 이력을 현재 관절값으로 초기화해 이전 episode 목표를 재사용하지 않는다.

이는 전송 목표 변화율 제한이다. 충돌·외력·drive 오차 상황의 실제 관절 속도 및 가속도 보장으로 해석하지 않는다. 실제 펌웨어의 보간/필터/지연 모델은 추가 측정해야 한다. 지연 0-step도 측정 결과가 아니라 기존 동작 유지값이다.

### 토크와 측정

`_compute_torques`의 arm torque를 0으로 만드는 것은 정책 action을 가리는 기존 VBC 설계이며 position drive를 끄는 것이 아니다. 이 tensor를 clipping하는 것으로 별도의 position drive torque를 실제 정격으로 제한했다고 볼 수 없다. 별도의 drive effort 속성 및 구동 모델을 확인해야 한다.

Isaac Gym의 DOF force sensor도 문서상 DOF total force다. 이를 검증 없이 실제 motor current 기반 torque로 대체하지 않았다. Reward의 torque 계산 경로를 이번에 바꾸지 않았다. 실기 구동 측정에서는 current/torque 피드백의 firmware·계수·부하를 함께 확인해야 한다.

## 검증 결과

1. CPU 단위 검증:
   - 3rad/s command slew 및 관절한계, 부분 reset 이력 초기화.
   - 팔 delay 2-step 설정, 다리 delay 0/1/3-step 인덱싱 및 warmup.
   - 18 action 중 arm 6개 mask 유지.
   - arm drive gain DR 반영, leg/finger DOF 속성 및 effort 값 보존.
   - 공식 기본 PiPER 각도 preset 일치 및 firmware별 속도 단위 변환.
2. 실제 PhysX CPU 연결 검증:
   - 4환경, 100 policy step = 약 2초, 학습되지 않은 zero-action 정지 hold.
   - 정상 reset으로 초기화한 후 **reset 0**, observation/state/reward 유한값.
   - arm drive stiffness/damping의 표본별 적용과 velocity=3 property readback.
   - command slew 최대 약 3.00000024rad/s(부동소수점 오차), 관측된 arm 속도 최대 약 1.15014rad/s.
   - 755 observation / 18 action / 20DOF 유지. optimizer update 0.
3. 물리 hold의 짧은 정상 실행은 보행·작업공간 추종·실기 동작 검증이 아니다. GPU 학습/렌더 작업은 시작하지 않았다.
4. 이번 URDF 변경 전후 차이는 6개 velocity 속성뿐이며 link·mass·CoM·inertia·joint origin·한계각은 byte-equivalent XML 비교로 보존 확인.

## 기존 카메라 자산 변경 발견 및 보존 범위

1·2단계 당시 training URDF hash는 `8df7e...`였지만, 이번 수정 **전** 백업은 이미 `e091ee55b02783d129ff7d90b2220e767cf08350bd10993112d38e1b8661a300`이며 현재 기존 no-contact 코드의 URDF와 동일하다. 카메라 mount fixed joint의 위치·방향 변화가 확인됐다. 이 변화는 이번 구동 패치에 포함하지 않으며 되돌리지 않았다.

현재 복제본은 여기서 velocity만 변경한 `7e37cadb8e55fddb50e6bcd10440c772997c72e11f72c7afddeef281954de02b`를 사용한다. 명목 질량은 약 23.43034kg이다.

- `external_asset_updates.json`: 최초 manifest 대비 관측한 원본 자산 변경 8개와 각각의 hash. 변경 주체/원인은 로그만으로 확정하지 않음.
- 최초 manifest를 덮어쓰지 않았다. frame 검증은 좌표 불변성을 계속 확인하되 외부 자산 변경을 별도로 출력한다. 이후 새 변경은 기록한 hash와 비교해 다시 탐지한다.
- 기존 226 checkpoint는 크기·mtime 유지 확인. 이번 작업은 원본 코드/기존 checkpoint를 쓰지 않았다.
- **2단계 workspace 결과와 앞선 이미지는 이전 카메라 배치 기준**이다. 샘플러 적용 전에 현재 카메라 형상으로 재검사해야 한다.

## 실기에서 확인해야 할 항목

| 항목 | 현재 상태 | 확인 방법 |
|---|---|---|
| 기종 | 기본 PiPER로 사용자 확인 | 추가 추정 없음 |
| 펌웨어 | 미확정 | `SearchPiperFirmwareVersion`, `GetPiperFirmwareVersion` |
| 각도/속도 | 공식 preset·프로토콜 범위만 확인 | `SearchAllMotorMaxAngleSpd`, `GetAllMotorAngleLimitMaxSpd`로 기기값 확인 |
| 가속도 | 미확정 | `SearchAllMotorMaxAccLimit`, `GetAllMotorMaxAccLimit` |
| 배포 제어 모드 | 미확정 | MOVE J/CPV/MIT 중 실제 bridge 동작 확정 |
| 연속/피크/열 토크 | 미확정 | 해당 모터/펌웨어 공식 정격과 current/torque feedback 대조 |
| PhysX gain 등가성 | 미확정 | 위치 step/sine 응답 및 부하 시험으로 식별 |
| 지연 | 미확정 | 송신 시각과 feedback 응답 측정. 기존 CAN 주기 숫자를 end-to-end latency로 간주하지 않음 |

`read_piper_limits.py`는 명시한 CAN에 읽기 조회만 보내는 준비 도구다. Codex는 실행하지 않았다. SDK가 설치되고 기기 프로토콜에 맞는 V2 환경에서 사용할 수 있으며 enable/move/config/reset을 보내지 않는다. 응답 freshness는 사용자가 확인하도록 raw 값을 보존한다.

## 변경 파일·복구·재현

- `legged_gym/utils/piper_control.py`: 새 명령 필터·drive property helper.
- `legged_gym/envs/manip_loco/manip_loco.py`: 실제 적용 경로 및 reset.
- `legged_gym/envs/manip_loco/as2_head_orin_config.py`: explicit 구동/지연/미확정 상태.
- `resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf`: velocity 6개 수정.
- `CHANGES_KO.md`: 수정 이력.
- 수정 전 파일은 `before/`, diff는 `step7.patch`, 검사 결과는 `verification.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/usr/bin/python3 low-level-as2piper-head-orin-deploy/validation/deploy_step7/verify_control.py
```

**이번 단계에서 미확정 값을 추가 적용하거나 Full Training을 시작하지 않는다.** 펌웨어/실측 확인 후 gain·토크·가속도·지연 적용안을 확정하고, 현재 카메라 배치의 workspace 재검사로 돌아갈 예정이다.

## 주기 설명 보충

VBC 실기 논문은 다리 정책 50Hz / 독립 팔 IK·명령 제어 800Hz다. 본 보고서의 50Hz 및 0.06rad/step은 **공개 학습 코드의 IK 목표 갱신** 기준이다. 실기 팔 제어를 50Hz로 확정했다는 뜻이 아니다. 비교와 유지/수정 권고는 `CONTROL_RATES_KO.md` 참조. 학습 코드의 물리/drive 갱신은 200Hz이며, 별도 PiPER 실기 팔 루프는 아직 구현하지 않았다.
