# VBC 환경을 유지한 AS2 보상 비교

## 공통 조건

- AS2에 대한 기존 하드웨어 비례값 유지: 추가 질량 0~5.879kg, CoM 각 축 ±9.427cm, push 속도 최대 0.3964m/s. 사용자 확인 후 적용.
- VBC와 같은 지형 생성기·설정: rough flat 100%, trimesh, 요철 크기 0~0.1m, terrain curriculum off, 원래 10행×20열/8m 구역.
- 6144 병렬 환경, episode 10초, sim dt 0.005s, policy dt 0.02s.
- 전체 robot DR 활성: 마찰 0.3~3.0, 모터 0.7~1.3, base 질량/CoM, 그리퍼 질량, 8초마다 push.
- 원래 reset 랜덤화, 이동 명령 생성 및 초기 forward-only schedule, random EE sphere 유지.
- AS2 포트의 799관측/2Hz 위상 입력을 모든 군에 동일하게 유지. B1 원본 파일 기본값의 observe_gait_commands=False와는 다르므로 문자 그대로 동일한 B1 환경이라고 표현하지 않는다.
- 동일 seed7, PPO, 모델, URDF, 제어기, 정책 초기화. 기존 학습 checkpoint를 불러오지 않고 scratch 시작.
- 초기 정책의 history 외 tensor33개 일치. 지형과 환경 원점 해시 일치. HardwareRunner 초기 PhysX step 이후 DOF 해시는 다르므로 bitwise 물리 재현성은 주장하지 않는다.
- W&B disabled. 기존 online run/PID512627은 유지한다.

## 비교군

| Preset | 변경 | 추가 학습량 |
|---|---|---:|
| original | 보존한 AS2 VBC 포트의 보상 | 600회 |
| speed | live foot velocity + contact-change 상태 수정 + 정지 contact 목표 + symmetric planar/yaw tracking | 200회 |
| stable | speed + 네 발 phase clearance/air time + walking_dof weight 조정 + 지면 상대 base 높이 | 600회 |

200회 시점은 세 군의 동등 학습량 비교, 400/600회는 original/stable의 학습 추이 비교다.
각 조합 내부 모든 weight의 개별 인과 효과를 분리한 것은 아니다.
original은 앞서 보존한 포트이며 contact sign correction을 포함한다. upstream B1의 원시 부호까지 완전히 같은 대조군이 아니다.

## 평가

- 학습 지형과 별도로 고정 seed20261102로 생성한 rough terrain, 전체 DR, random arm goals.
- 동작당16환경, 총64환경×20초. 정지, 전진0.25m/s, yaw+0.6rad/s, 전진+회전.
- 속도 오차는 첫2초 제외, 넘어짐은 전체20초에서 집계.
- 한 번도 넘어진 적 없는 환경 비율·초기/후반 reset을 추가로 구분해 해석한다.
- 평가 환경 원점 기준 발 높이/몸통 높이는 실제 국소 지면 clearance가 아니다. rough에서는 넘어짐과 속도 추종을 우선 판단한다.
- 마지막 평가에서 실제 PhysX 지형 triangle mesh를 export해 영상에 사용한다. 평지로 바꾸어 렌더링하지 않는다.
- 평지 성능이나 학습 중 scalar reward만으로 rough 보행이 안정적이라고 판정하지 않는다.

## 100회 pilot 결과를 반영한 추가 비교군

`stable`의 32환경×20초 pilot에서 전진 명령 0.25m/s에 평균 약0m/s,
회전 명령 0.6rad/s에도 약0rad/s였다. 총 넘어짐은3회로 적었으나 정지에
머무는 정책이므로 보행 성공으로 보지 않는다.

추가 `progress` 군은 `stable`의 모든 설정을 유지하고 planar/yaw tracking의
raw reward만 `0.5*Gaussian + 0.5*clipped directional progress`로 변경한다.
방향 진행도는 실제 속도와 명령의 내적을 명령 norm제곱으로 나누고[-1,1]로
제한한다. 정지 명령에서는 기존 Gaussian과 같다. 목표 속도에서 최대이고,
과속 시 Gaussian 항이 감소하므로 기존 VBC의 완전 포화와 구별된다.
목표에서 먼 초기 속도에서도 목표 방향 이동에 따른 보상 차이를 제공하려는
가설이며, 수식 검사 통과를 보행 성공으로 표현하지 않는다.

`speed` 군200회가 끝난 뒤 GPU 슬롯을 재사용해 scratch/6144환경/seed7/200회로
학습한다. 기존 `stable`의200회와 비교하면 이 진행도 항의 조합 효과를 분리할 수 있다.
검증 전에는 기본 task 보상으로 적용하지 않는다.

## 저장 counter와 동일 학습량 비교

기존 runner의 periodic `model_100.pt`는101회 완료, `model_200.pt`는201회
완료 상태다. 200회로 종료한 speed의 최종 `model_200.pt`만 실제200회다.
파일명 대신 평가 JSON의 completed_updates와 checkpoint metadata로 구분한다.
- 네 군의100번 periodic 파일은 모두101회 완료 상태로 비교한다.
- original/stable/progress의200번 periodic 파일은 모두201회 완료 상태로 비교한다.
- speed 최종200회는 보조 결과로 명시하며201회와 엄밀한 동일 학습량 결과로 표현하지 않는다.
- progress는201회까지 학습해 이 counter 차이를 맞춘다. 이 변경은 대기 중인 자체 큐 helper만 교체한 것이며 실행 중 학습은 중지하지 않았다.

## Full-environment baseline 기준 (progress 평가 전 명시)

- 정지 vx MAE<0.08m/s, yaw MAE<0.15rad/s.
- 전진 vx MAE<0.10m/s 및 평균 속도와 목표의 차이<0.10m/s.
- 회전 vx MAE<0.10m/s, yaw MAE<0.25rad/s.
- 전진+회전 vx MAE<0.10m/s, yaw MAE<0.25rad/s.
- 전체 환경 중20초 동안 넘어짐이 한 번도 없는 비율>=95%.
- 각 명령에서 초기2초 이후 roll/pitch 절댓값<0.35rad인 비율>=99%.
- 각 명령에서 몸통 접촉력5N 초과 비율<1%.

거친 지면에서 환경 원점 기준 높이>0.22m 조건은 정확한 국소 지면 높이가
아니므로 baseline 판정에 쓰지 않는다. 원래 평지 판정은 별도로 보존한다.
전체 넘어짐0회 여부도 별도 strict_no_fall로 보고하며, baseline 통과를
전체 DR 조건에서 무실패라는 의미로 표현하지 않는다. 단일 seed 초기학습의
실험적 기준이며 실물 인증 기준이 아니다.

## 추종 보상 크기 분리: progress_gain2

progress의101회 pilot도 전진/회전 속도가 거의0이어서, 다음 군은
진행도/Gaussian 혼합식과 나머지 보상을 그대로 두고 planar tracking weight만
2→4, yaw tracking weight만1→2로 바꾼다. 정지 명령의 tracking 보상 크기도
함께 증가한다. 별도 속도 제한이나 명령 범위 축소는 하지 않는다.

progress201회 학습 종료 후 같은 GPU 슬롯에서 scratch/seed7/6144환경/201회로
실행한다. progress와 progress_gain2의 실제201회/64환경×20초 결과가
추종 보상 크기 조합의 주 비교다. gain2의100번 중간 평가는64환경 진단이며,
앞선32환경 pilot과 동일 표본 비교로 표현하지 않는다.

추종 gain2 군의 최대 학습량은600회로 확장했다. 201회 비교와 함께
original/stable/gain2의 실제600회 비교가 가능하도록 한 것이다.
별도 holdout 검증을 통과하면 완료된 optimizer update를 저장한 뒤
협조적 stop marker로 일찍 종료할 수 있다. 이를 위해 새 gain2 실행에만
--stop-file을 지정했고, 기존 실행 중 학습에는 손대지 않았다.

## Tracking 2×2 비교

모든 군에서 live foot velocity/정지 목표/phase clearance/air time/posture 수정은 공통이다.

| 조건 | Gaussian only | Gaussian + directional progress |
|---|---|---|
| planar2 / yaw1 | stable | progress |
| planar4 / yaw2 | gaussian_gain2 | progress_gain2 |

각 군의 실제201회 완료 모델을 동일한64환경×20초에서 비교한다.
새 gaussian_gain2는 자체 original600회 학습 종료 후 확보되는 GPU 슬롯에서
scratch/seed7/6144환경/201회로 실행한다. 기존 사용자 온라인 학습의 종료를
기다리거나 그 프로세스를 조작하는 것이 아니다. 이 대조군으로 추가 진행도
항이 필요한지, 추종 weight 조정만으로 충분한지 구분한다.
