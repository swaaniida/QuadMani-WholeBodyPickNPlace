# VBC 원본과 AS2의 제어 주기·Action Delay 비교

## 핵심 구분

VBC 논문의 실기 배포는 **다리 정책 50Hz + 별도 IK 기반 팔 제어기 800Hz**다. 앞선 50Hz 설명은 공개 학습 코드에서 IK 목표가 갱신되는 주기이며, 실기 팔 제어 주기 설명으로 사용하면 부정확하다.

근거: [VBC 저자 공개 논문 Fig.12 및 Appendix G](https://zixuan417.github.io/assets/files/vbc.pdf).

## 코드와 논문 비교

| 계층 | VBC 공개 학습 코드 | VBC 실기 논문 | 현재 AS2 복제본 / 권고 |
|---|---|---|---|
| 다리 정책·관측/history | 50Hz | 50Hz | 50Hz 유지 |
| 물리 적분 | dt=0.005s, 200Hz, substeps=1 | 시뮬레이션 적분에 대응하는 항목 아님 | 200Hz 유지 |
| 다리 PD torque 계산 | 정책 action을 유지하며 물리 step마다 200Hz | 실제 모터 내부 주기는 별도 | 학습 200Hz 유지 |
| Arm IK 목표 새 계산 | decimation loop 밖, 정책 step당 1회 = 50Hz | 독립 IK 기반 팔 제어기 800Hz | 학습은 기존 50Hz 유지; 실기 bridge는 별도 빠른 팔 루프 필요 |
| Arm position target 설정 | 같은 목표를 decimation=4회, 200Hz로 설정 | 800Hz로 IK/명령 송신 | 현재 학습 코드 역시 200Hz 설정, 실기 bridge 미구현 |
| 내부 motor servo | PhysX drive | Z1 내부 제어기와 구분 필요 | PiPER 펌웨어 내부 servo 주파수는 이번 자료로 미확정 |

원본 근거(읽기 전용):

- `vbc-legacy/low-level/legged_gym/envs/base/legged_robot_config.py`: sim dt=0.005, substeps=1.
- `vbc-legacy/low-level/legged_gym/envs/manip_loco/b1z1_config.py`: decimation=4.
- `vbc-legacy/low-level/legged_gym/envs/manip_loco/manip_loco.py`: step 내 IK 계산이 for-decimation 밖, target 설정·torque 계산·simulate는 안쪽.
- `b1z1_interface.py`는 Gym environment를 step하는 스크립트이므로 파일 이름만으로 800Hz 실기 bridge라고 간주하지 않음.
- PhysX num_position_iterations=4는 solver 반복 횟수로, 200×4=800Hz 제어 주기라는 뜻이 아님.

## 다리 Action Delay의 의미

정책이 새로 만든 다리 action을 몇 policy step 늦게 적용하는 시뮬레이션 지연이다. 모터 제어 주파수·EE 목표 주기·팔 delay와는 다른 항목이다.

원본의 실제 동작:

- `global_steps < 10000*24`: 최신 action 적용 → 추가 delay 0ms.
- 이후: 직전 action 적용 → 추가 delay 20ms.
- 설정의 action_delay=3은 실제 인덱싱에 쓰이지 않아 60ms로 동작한 것이 아니다.
- 복제본은 action_delay=1 및 warmup=240000으로 실제 동작을 명시하고, 설정한 지연을 인덱싱에 사용하도록 수정했다. 기본 동작 0→20ms는 동일.
- 지연된 action은 매 5ms 물리 step에서 다리 torque 계산에 사용된다. 팔은 action 6개가 mask되고 별도 IK이므로 이 delay의 적용 대상이 아니다.

예: 50Hz에서 t에 계산한 다리 action a[t] 대신 a[t-1]을 적용하는 동안에도 다리 PD 계산과 물리 적분은 매 5ms 계속 수행된다.

## 유지·수정 방향

### 유지

- 학습 정책/관측/history 50Hz, sim dt=5ms, decimation=4. 공개 원본과 동일.
- 학습 Arm DLS IK 목표 계산 50Hz 및 PhysX position drive 200Hz. 이번 질문으로 바꾸지 않음.
- 기존 실제 다리 추가 delay 0→20ms. 실측 근거 없이 60ms로 늘리지 않음.
- 위치 속도 한계 3rad/s와 이를 시간 간격으로 환산하는 원칙.
- PPO/DAgger/ROA, action/observation 차원 유지.

### 배포를 위해 추가해야 할 부분 — 아직 미구현

- 50Hz 정책과 분리된 팔 상태 수신·IK·속도/가속도 제한·명령 송신 루프.
- PiPER에서 지원되는 제어 모드와 실제 joint-state freshness, CAN 대역폭·왕복 지연을 확인해 팔 루프 주기 결정.
- VBC와 같은 800Hz를 목표로 비교할 수 있으나, Z1이 가능하다는 이유로 PiPER의 유효 송신/응답 주파수를 800Hz로 확정하지 않음. 같은 오래된 상태를 반복해서 읽는 것은 800Hz feedback 제어가 아님.
- 명령 변화량은 `velocity_limit * actual_control_dt`로 계산. 3rad/s이면 50Hz에서 0.06rad, 200Hz에서 0.015rad, 800Hz에서 0.00375rad. **0.06rad를 실기 800Hz 루프에 그대로 사용하지 않음.**
- 독립 팔 루프에서도 몸통 움직임을 반영한 동일한 world/arm-base 변환과 시간 동기화 필요.

현재 추가한 ArmCommandFilter는 dt를 받아 계산하므로 0.06을 하드코딩하지 않았다. 그러나 현재 연결된 코드는 학습 루프이며, 완성된 실기 비동기 팔 제어기가 아니다.

이 문서는 비교·권고안이다. 이번 요청에서 제어 주기·알고리즘·gain을 추가 변경하거나 학습을 시작하지 않았다.
