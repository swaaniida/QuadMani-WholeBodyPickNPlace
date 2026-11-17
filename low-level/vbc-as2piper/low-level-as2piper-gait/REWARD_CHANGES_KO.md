# AS2 + PiPER gait 보상 수정 명세

## 보존 및 분리

- 기존 AS2 VBC 원복판: `../low-level-as2piper-hw`, task `as2piper_plate_vbc_port`.
- 새 gait 코드: 현재 디렉터리, public task `as2piper_plate_gait_stable`.
- 원본 B1: `/data/team/users/suhwan/vbc-legacy`. 읽기 전용이며 수정하지 않음.
- 정책18출력/시뮬레이션20DOF,799관측, AS2 URDF·질량·관성·PD·armature, PPO/network/storage는 유지.
- 새 task는 `AS2PiPERVBCPort`의 별도 subclass에서만 보상과 contact target을 바꿈.

## 항목별 변경

| 항목 | 기존 | 신규 | 의도 |
|---|---|---|---|
| 발 속도 | 초기 tensor-index 복사본 | 매 PhysX refresh 이후 현재 발 속도 선택 | stance 중 실제 발 이동이 penalty에 반영 |
| feet_jerk | 확인 객체와 저장 객체 불일치, 항상0 | 같은 environment에 이전 wrench 보관, 초기1초 제외 | 접촉력 변화에 대한 기존 의도 복구 |
| 정지 접촉 | desired contact 네 발 모두0.5 | 정지 시 모두1 | 네 발 지지 시 force penalty0 |
| planar tracking | 목표 이상 속도에서 포화 | `exp(-||v_xy-cmd_xy||² / 0.20²)`, weight2 유지 | 과속·저속·옆방향 이탈 모두 감소 |
| yaw tracking | denominator0.2, weight0.5 | sigma0.35rad/s, weight1 | 명령에 따른 회전 추종 유인 강화 |
| feet_height | 앞발2개 world-z norm | 네 발, stance0/swing sinusoid 최대4cm, sigma2.5cm | 위상에 맞춰 적당히 들어 올림 |
| feet_air_time | 앞발2개,0.5초 기준, weight2 | 네 발, `0.5/frequency=0.25초`, credit 상한+0.10초, weight0.5 | 2Hz/duty0.5와 시간 기준 일치, 긴 비행의 유인 제한 |
| walking_dof | home 근접 보상 weight1.5 | weight0.5 | home 유지가 실제 발 디딤을 과도하게 억제하지 않도록 완화 |
| base_height | world z와 고정 target 비교 | 지면 높이를 뺀 root 높이와 기존 AS2 target 비교 | 지면의 절대 고도가 바뀌어도 같은 높이를 평가 |

base height의 지면 기준 수정은7조건 평지 비교 완료 후 추가했다. 평지에서
ground height는0이므로 기존 식과 동일하며, 비교 결과를 바꾸지 않는다.
거친 지면의 초기 v1은 height grid의 보수적 3이웃 추정치를 사용했다.
하중이 걸린 발 바닥과 grid 추정치의 차이가 P95 4.35cm로 확인되어,
v2에서는 수직 접촉력 5N 초과 시 발 중심 높이에서 반지름 28mm를 뺀 값을
발별 지지 높이로 저장한다. 비접촉 중에는 마지막 접촉 높이를 유지한다.
몸통 기준 지면은 네 발 지지 높이 평균을 EMA 계수 0.2로 갱신한다.
Reset 시 해당 환경의 지면 원점 높이로 캐시를 초기화한다.
이는 접촉 기반 근사이며 경사면의 정확한 ray cast나 미래 착지 지형 관측이 아니다.

거친 지면에서는 swing clearance를 최소 높이 조건으로 바꿔 필요한 높은 발걸음을
벌하지 않는다. 평지에서는 기존의 대칭 Gaussian과 4cm 목표를 그대로 사용한다.
이 두 변경을 묶은 v2와 기존 grid v1을 같은 초기 checkpoint/Adam/seed,
4096환경/120회 추가 학습으로 비교한다. `stable_grid` preset은 v1 재현용이다. v2는 36회 넘어짐으로 v1의 27회보다
나빴으므로 기본값에서 제외했다. 최종 `stable`은 grid 방식이고
`stable_contact`로 v2 실험을 별도 재현할 수 있다.

기타 reward weight와 기능은 유지한다. `speed`는 tracking 두 항목까지,
`stable`은 clearance/air-time/posture/terrain-height 조합까지 적용한 조건이다.
각 조합 내부의 개별 weight가 최적이라는 주장은 하지 않는다.

## 환경 및 command 범위

Public task 기본값은 원복판의 rough terrain, robot DR, command sampler,
random EE sphere 범위를 유지한다. 온라인 후보 IK/충돌 guard를 다시 추가하지 않았다.
비교 및 첫 gait 학습의 `--controlled`는 평지·robot DR off이며,
원래 +/-20% joint reset·random EE·command 분포·상자 actor는 그대로다.
상자 질량 sampler도 원래 설정이다. 이 조건을 full-DR 수렴으로 표현하지 않는다.

## 로그 해석

호환성을 위해 기존 reward key를 유지한다. 새 task에서는:

- `rew_tracking_lin_vel_max`: 이름과 달리 symmetric planar Gaussian 보상.
- `metric_tracking_lin_vel_max`: planar 속도 오차 norm(m/s).
- `metric_tracking_ang_vel`: yaw 절대 오차(rad/s).
- `metric_feet_height`: 네 발의 목표 clearance 절대 오차 평균(m).
- `metric_base_height`: 지면 기준 root 높이(m).

episode metric은 기존 logger가 누적/정규화하므로 원시 물리량의 단순 평균과
혼동하지 않는다. 실제 속도·오차는 평가 JSON과 NPZ에서 직접 계산한다.

## 공개 실행 경로

`run_stable.sh`는 기존 VBC training entry와 target iteration 방식을 사용한다.
W&B project는 `as2piper-low`, 기본 host는 기존 CoreWeave host다.
`run_gait.sh`는 원인 비교용 explicit preset entry이며 `--iterations`는 새 업데이트 수다.
두 entry 모두 기존 run 디렉터리나 기존 프로세스를 지우지 않는다.

## Full-environment 추가 후보: progress (검증 중)

`stable`과 동일한 환경·보상 weight·phase 설정에서 planar/yaw tracking의 raw 값만
`0.5*Gaussian + 0.5*clip(실제 속도·명령 내적 / 명령 norm², -1, 1)`로 바꾼다.
정지 명령에는 기존 Gaussian을 사용한다. 목표에서 최대이고 과속에서는 Gaussian이
감소한다. 목표에서 먼 초기 속도에도 방향 진행에 따른 보상 차이를 제공하는 가설이다.
원래 VBC 총보상의 최종 `/100`, PPO, 관측, reset, EE sampling은 유지한다.
현재 기본 task에는 아직 적용하지 않았다. full-DR 학습·평가 후 채택 여부를 결정한다.

## 후속 접촉 입력·지지 높이 비교 (진행 중)

생산용 센서·관측·물리 설정은 원복판과 같다. 공식 force sensor flag나 새 contact filter는 측정 결과가 불충분하여 채택하지 않았다. `stable_proxy`는 force shaping 항의 입력만 기존 관측용 접촉 bit로 바꾼다. 이 bit는 힘의 크기가 아니다. `stable_proxy_ground`는 같은 bit로 마지막 접촉 발 높이를 기억하며, 거친 지면에서도 대칭 Gaussian clearance를 사용한다. 앞서 설명한 5N/단측 높이 v2와 별개다.

현재 공개 task의 기본 보상은 평지에서만 보행을 확인한 stable/grid다. full-environment 후보 채택 전이므로 이를 거친 지면·전체 DR의 검증 완료 버전으로 해석하지 않는다. [최신 조건·중단 및 재개 이력](experiments/gait_reward_proxy_v2/PROTOCOL_KO.md), [측정 결과](experiments/gait_reward_proxy_v2/RESULTS_KO.md)를 우선한다.
