# Effort·Gain 검증 및 변수 출처

## 결론

- **400/40은 VBC 원본의 PhysX 제어 설정값**이다. 값 선정 식이나 실측 식별 근거는 확인한 원본 코드에 없다. 무작위로 정했다는 증거도 없으므로 **선정 근거 미공개 튜닝 파라미터**로 분류한다.
- **PiPER effort=100Nm, velocity=5rad/s는 PiPER 제조사 원본 URDF에서 가져온 값**이다. B1/Z1에서 복사하거나 이번 작업에서 새로 임의 산정한 값이 아니다. 다만 URDF 값이 실제 연속/피크/열 정격을 증명하지는 않는다.
- VBC Z1 원본 URDF effort는 **[30,60,30,30,30,30]Nm**, velocity는 전 관절 **10rad/s**다. 이 또한 리소스 URDF 사용값이며 본 작업에서 실기 정격과의 일치를 검증한 값이 아니다.
- 현재 카메라 포함 모델의 중력 요구 토크와 시뮬레이션 응답은 산출했다. **실제 PiPER의 gain 등가성 및 모터 정격은 미확정**이며, 후보 시뮬레이션 effort를 하드웨어 확정값으로 적용하지 않았다.
- 학습 코드·reward·gain·effort를 이번 비교 중 추가 변경하지 않았고 optimizer update는 0이다. 사용자 요청대로 별도 가속도 제한기는 추가하지 않는다.

## 채워야 하는 변수와 근거

| 변수/설정 위치 | VBC B1+Z1 원본 실제 사용 | 현재 AS2+PiPER | 값의 성격·판단 |
|---|---|---|---|
| `control.arm_drive_stiffness` | 코드 하드코딩 400 | 400 | 시뮬레이션 position drive Kp. 제조사 공식 gain으로 확인한 값 아님; 응답 비교로 선정하는 튜닝 변수 |
| `control.arm_drive_damping` | 코드 하드코딩 40 | 40 | 시뮬레이션 position drive Kd. 도출 근거 미공개; 응답 비교로 선정 |
| URDF 각 arm joint의 `limit.effort` | [30,60,30,30,30,30]Nm | [100,100,100,100,100,100]Nm | 각 로봇 원본 URDF 출처. PiPER 100은 공식 모델 값이지 실기 정격 인증값은 아님 |
| URDF `limit.velocity` / `control.arm_command_velocity_limit` | URDF 10rad/s, 별도 목표 변화율 제한 없음 | URDF·drive·목표 변화율 3rad/s | PiPER 원본 URDF는 5였지만 공식 위치 명령 설정 범위 상한 3을 적용. 실제 기기 설정값은 별도 |
| `domain_rand.arm_motor_strength_range` | [0.7,1.3] 추출, 실제 arm drive에는 효과 없음 | [0.7,1.3]로 actual drive gain 배율 적용 | 학습용 DR 범위. 제조사 공차·실측 분포 아님. 실제로 활성화한 것은 원본 대비 동작 변경 |
| `control.arm_command_delay_steps` | 별도 arm 명령 delay 없음 | 0 | 학습 기준 원본 유지. 실기 지연 측정치가 아님 |
| `control.arm_command_acceleration_limit` | 별도 제한기 없음 | None | 사용자 결정에 따라 원본처럼 미적용. 채워야 하는 필수 변수가 아님 |
| IK damping | DLS λ=0.05, 식에는 λ²I | 동일 유지 | 수치 안정화 파라미터. 모터 정격/PD damping과 다른 값 |
| 학습 sim dt / decimation | 0.005s / 4 | 동일 유지 | 200Hz 물리, 50Hz 정책·IK. 모델/알고리즘 설정값 |
| 다리 action delay | 설정3, 실제 초기0→후반1-step | 설정1, 동일한0→20ms | 코드 실제 동작 정합성 수정. 실기 지연 도출식 아님 |
| URDF 관절 friction | Z1 [1,2,1,1,1,1] | PiPER arm joints에 별도 dynamics 미기재 | 원본 모델의 차이. 같은 gain이라고 실제 응답이 같아지는 것이 아님. 이번에 임의 마찰값 추가 안 함 |

**구분:** 공식 하드웨어 수치 / 공식 URDF의 시뮬레이션 모델 값 / 연구 코드의 제어 튜닝값 / 학습용 randomization 범위는 서로 다른 출처다. 공식 문서의 값이 아니라는 이유로 모두 무작위 값이라고 판단하지 않는다.

## 출처

읽기 전용 원본:

- `/data/team/users/suhwan/vbc-legacy/low-level/legged_gym/envs/manip_loco/manip_loco.py`: 400/40, arm torque mask, DLS IK, delay.
- `/data/team/users/suhwan/vbc-legacy/low-level/legged_gym/envs/manip_loco/b1z1_config.py`: DR, decimation. config의 z1 5/0.5는 실제 position drive 400/40과 구분.
- `/data/team/users/suhwan/vbc-legacy/low-level/resources/robots/b1z1/urdf/b1z1.urdf`: Z1 effort/velocity/friction.
- `/data/team/users/suhwan/robot-sources/agx_arm_urdf/piper/urdf/piper_description.urdf`: PiPER 원본 effort100/velocity5.
- [공식 PiPER 원본 URDF](https://github.com/agilexrobotics/agx_arm_urdf/blob/main/piper/urdf/piper_description.urdf)
- [공개 VBC 환경 구현](https://github.com/Ericonaldo/visual_wholebody/blob/main/low-level/legged_gym/envs/manip_loco/manip_loco.py)
- SDK 0.6.1 소스·사용자 확인 펌웨어 S-V1.5-8: 상위 `official/piper_sdk_0_6_1`, `confirmed_device_profile.json`.

## 현재 모델 정적 요구 토크

카메라 배치 갱신을 포함한 현재 URDF로 다시 산출했다. 네 발 고정 Leg IK + Arm 6D IK를 만족한 자세이며 조사 자세의 비인접 링크/지면 관통은 검출되지 않았다.

| 자세 | 몸통 높이 / pitch | Arm J2 중력 필요 토크 절댓값 | Arm J3 절댓값 |
|---|---|---:|---:|
| 초기 home | 0.3244m / 0° | 2.59Nm | 5.36Nm |
| 중간 목표 | 0.30m / 15° | 6.37Nm | 3.46Nm |
| 바닥 x=0.90m | 0.30m / 25° | 12.06Nm | 3.91Nm |
| 바닥 x=0.975m | 0.27m / 30° | 12.94Nm | 4.78Nm |
| 상단 z=0.80m | 0.3244m / -15° | 3.52Nm | 5.66Nm |

이는 물체 하중·열·마찰이 없는 모델 중력 요구량이다. 원격 서버의 계산으로 얻을 수 있으며 실제 출력 가능량과 구분한다.

## CPU PhysX 비교: 140조건

- 현재 카메라 포함 20DOF 모델, 고정 몸통, 5개 자세 × 7개 gain × 4개 진단용 effort = 140조건.
- gain: 100/10, 200/20, 280/28, 400/40, 520/52, 800/80, 400/10.
- effort: 100×6, [32×3,8×3], [16×3,4×3], [8×3,2×3]. **100 이외 프로파일은 민감도 비교를 위한 가정이며 공식 정격이라고 주장하지 않는다.**
- 모든 조건 동일 7초: 정지→작은 관절 step→0.5Hz 사인→정지. 목표 갱신50Hz, 물리200Hz, 명령 변화율3rad/s. DOF drive property readback 확인.
- 자기충돌은 이 고정 몸통 응답 비교에서 비활성화. 따라서 경로 충돌 검증을 대체하지 않는다. 실제 이동체의 전신 안정성/걷기 비교도 아니다.
- actual TCP와 현재 URDF FK를 별도 엔진에서 대조: 좌표계의 상수 평행이동을 정렬한 후 최대 잔차 약 0.0083mm. 모델·좌표 추출 정합성을 확인했다.

### 전방 0.975m 자세, effort100 비교

| Kp/Kd | 정지 TCP 오차 평균 | 사인 구간 TCP 오차 평균 | 관측 최대 관절 속도 |
|---|---:|---:|---:|
| 100/10 | 11.86cm | 12.13cm | 0.76rad/s |
| 200/20 | 6.51cm | 6.88cm | 0.76rad/s |
| 280/28 | 4.89cm | 5.31cm | 0.77rad/s |
| **400/40** | **3.66cm** | **4.13cm** | **0.77rad/s** |
| 520/52 | 2.99cm | 3.49cm | 0.77rad/s |
| 800/80 | 2.20cm | 2.77cm | 0.76rad/s |
| 400/10 | 3.17cm | 3.34cm | 1.99rad/s |

**판정:** 400/40은 이 모델의 짧은 응답 시험에서 발산하지 않지만, 원거리에서는 중력에 의한 정상 오차가 크다. 고정 gain을 높이면 이 시험에서는 오차가 줄었으나 그것만으로 실기 gain을 선정할 수 없다. 낮은 damping은 순간 속도와 모델 역동역학 추정 요구 토크를 높였다.

400/40의 초기 home 오차도 약 1.35cm다. 모든 자세에서 동일한 gain의 효과가 같지 않다. 실제 펌웨어가 어떤 중력 보상을 수행하는지 및 실제 위치제어 응답과 맞추지 않고, 이 결과만으로 800/80으로 올리지는 않았다.

### Effort 민감도

- 원거리 자세에서 J1~3 effort를 8Nm로 제한한 진단 조건은 약 44cm 정지 오차를 보였다. Kp를 높여도 해결되지 않는다. J2 중력 요구량 약13Nm보다 작은 출력 상한이므로 gain 변경으로 부족한 토크를 대체할 수 없다.
- 이 완만한 시험에서 16/4 및32/8 조건은100 조건과 비슷한 응답을 보였다. 따라서100Nm가 필요하다는 근거는 없다.
- 반대로 이 결과가 실제 정격16Nm를 입증하지도 않는다. 물체 하중·외란·빠른 전신 동작·열 제한은 포함하지 않았다.

## 토크 채널 검증 결과

실제 PhysX DOF force sensor를 활성화해 읽었다. 하중을 지지하는 정지 자세에서도 반환값이 모터 중력 토크와 달리 거의0이었다. **이 total generalized force를 실제 position drive motor torque라고 해석할 수 없다.** `self.torques`의 arm0을 실제 출력 토크로 해석할 수 없는 것과 같은 계측상의 구분이다.

추가로 MuJoCo 모델 역동역학을 이용해 샘플된 q/dq/수치미분 qdd의 필요 토크를 추정했다. 이는 모델 추정값이며 실제 측정값이 아니다. 50Hz 기록으로 계산했으므로 빠른 transient peak를 놓칠 수 있다. 토크 safety 인증이나 실제 current-limit 검증에 사용하지 않는다.

## 권고 및 남은 결정

1. 원본의 400/40은 임의의 무작위 수치가 아니라 코드에 고정된 제어 파라미터다. 선정 근거가 확인되지 않으므로 PiPER의 공식 gain으로 취급하지 않는다.
2. PiPER100Nm는 공식 URDF 모델값 출처가 명확하다. 단, 이를 실제 정격으로 확정한 것으로 표시하지 않는다.
3. 학습 전 최종 입력할 핵심은 **명목 Kp·Kd 및 관절별 effort**다. 속도3rad/s·원본 학습 주기·가속도 미적용은 이미 결정된 상태다.
4. 실기 식별을 제외하는 경우에는 “공식 모델값 + 시뮬레이션 검증” 수준으로만 선정할 수 있다. 실기 등가성을 검증했다고 주장할 수 없다. 임시 물리 상한을 쓸 경우 가정과 민감도 결과를 명시해야 한다.
5. 최종 gain/effort 및 DR 정책은 사용자 검토 전 추가 적용하지 않는다. 기존 45k 실험과 코드 보존, 학습 미실행.

산출물: `static_requirements.json`, `physx_signals.npz`, `bench_metadata.json`, `bench_results.json/csv`, `bench_traces.json`. 모든 실행은 CPU, optimizer update0. 실기·W&B 통신 없음.
