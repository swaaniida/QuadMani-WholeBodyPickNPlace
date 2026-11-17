# Observation / PPO 경로 검증

## 결론 및 범위

기존 URDF·20 DOF/18 action·TCP/IK·좌표계·joint mapping·초기 자세·Reset/Push 및 Reward 수정 유지. Reward scale과 PPO 수식 유지. 확인된 입력·저장·배포 경로 오류만 수정했다.

실제 PPO 학습은 실행하지 않았다. 16환경 × 4초 rough PD/IK rollout과 저장 데이터 기반 forward/backward 검사를 실행했다. PPO.update와 update_dagger의 실제 backward를 통과하되 optimizer.step 직전 예외로 중단했다. 모델 파라미터 해시 불변, 두 Adam state 비어 있음, optimizer step 0회. 진단용 새 모델의 경로 검증이며 학습 성능 검증은 아니다.

## 1. Foot-contact 불일치

| 항목 | 기존 observation | Reward 접촉 상태 / 수정 observation |
|---|---|---|
| 신호 | 발에 부착한 6축 force sensor 전체 norm | 발 링크 net contact force의 XYZ norm |
| 임계값 | >1.5, 힘·토크를 혼합 | >2 N |
| 시간 처리 | 현재 표본 | 현재 OR 직전 표본 |

기존 센서는 외력·접촉력·관절 내부 solver force를 포함한 링크의 합력을 읽는다. 정적 지지에서도 내부 반력이 접촉력과 상쇄되므로 지면 normal force와 같지 않다. 전체 6축 norm은 N과 Nm도 혼합한다. 로컬 Isaac Gym 문서 `third_party/isaacgym/docs/programming/forcesensors.html`의 rigid-body sensor 설명과 실제 계측이 일치한다. 센서 원점 변경이나 threshold만으로 해결할 문제가 아니다.

기존 약56%는 이전 rough 조건의 결과다. 이번 수정 전 재검사 observation 대 contact_filt 불일치는 42.41%; 수정 후 0%. 최종 재검사에서 기존 sensor 대 순간 net-contact 불일치는 52.22%. 비교 신호·필터·물리 실행이 달라 이전56%와 같은 값일 필요는 없다. 0%는 동일 상태 공유의 정합성을 의미하며 물리 접촉 정답률을 의미하지 않는다. Rough mesh net-contact 자체의 누락·스파이크 가능성도 남는다.

수정: 기존 필드 이름 `foot_contacts_from_sensor`를 유지하고 값은 `contact_filt`로 연결. 4개 위치·차원·FR/FL/RR/RL 순서 유지. Air-time/drag 등 접촉 bool 소비자는 같은 상태를 사용한다. Contact-force penalty 등 연속 힘 reward는 필터 bool 대신 실제 force를 사용하는 것이 정상이다.

실기기: Isaac의 net-contact tensor를 직접 얻을 수 없다. AS2의 발 force 제공 여부·단위·좌표·offset·주기 또는 토크 기반 contact estimator를 실제 SDK/센서와 검증해야 한다. 2N은 시뮬레이션 기준이며 하드웨어 임계값으로 인증되지 않았다. 같은 current OR previous 필터·순서·50Hz 입력 계약을 맞춰야 한다.

## 2. 744차원 observation 및 배포

인덱스는 Python 반개구간 `[start:end]`다.

| 구간 | 차원 | 값 / 정규화 |
|---|---:|---|
| 0:2 | 2 | 몸통 roll, pitch [rad] |
| 2:5 | 3 | 몸통 좌표계 angular velocity ×1 |
| 5:23 | 18 | q−default_q ×1, 다리12+팔6 |
| 23:41 | 18 | dq ×0.05, 같은 joint 순서 |
| 41:53 | 12 | 가장 최근 발행한 다리 action, policy 순서 |
| 53:57 | 4 | contact_filt, FR/FL/RR/RL |
| 57:60 | 3 | vx/vy/yaw-rate command ×[1,1,1] |
| 60:63 | 3 | 실제 arm base에서 본 EE 목표 위치 [m], 몸통 full rotation 반영 |
| 63:66 | 3 | 원본부터 의도된 0: EE 방향을 직접 관측하지 않음 |
| 66:84 | 18 | privileged physics |
| 84:744 | 660 | 이전 10개 66차원 frame, 과거→최근 |

Proprio66 + privileged18 + history660 =744. 누락된55차원 zero padding 구조가 아니다. Base linear velocity 직접 관측도 원본처럼 없고 history로 추론한다. EE orientation 목표는 IK에 전달되지만 정책 입력에는 명시되지 않는다. 이번에는 원본 구조를 유지했다.

Privileged 순서: 추가 base mass1, base COM delta xyz3, 추가 gripper mass1, friction1, native joint 순서 leg motor-strength−1의12. 합계18. 단위 기반 고정 scale을 쓰며 running mean/std normalization은 없다. 기본 noise=False, observation/action clip ±100. privileged와 history에도 별도 RMS 표준화는 없다.

History: 반환 observation에는 갱신 전 history가 포함된다. Reset 직후 반환 history는0, 내부 history는 현재 frame 10복사로 채운다. 다음 호출에서 그 history가 반환된다. 원본 시간 계약 유지. 이전 action 구간은 지연 전 최근 발행 action이므로 지연 후 실제 적용 action과 다를 수 있다.

훈련 actor: 현재66+privileged latent20. History actor: 현재66+history latent20. Critic: 현재66+privileged18, 2개 value head. History encoder는 privileged latent를 teacher로 DAgger 학습한다. Privileged18은 실기기에 요구되지 않는다.

배포 입력: `cat(obs[:,:66], obs[:,84:])` →726. 새 JIT exporter는 history branch를 명시하고 입력726을 검사한다. Native history와 JIT 출력 최대 오차 0.0; privileged 값 변경 시 history 출력 변화0. Native inference 기본값도 history=True로 연결했다. 기존 JIT 파일은 새 exporter로 다시 생성해야 한다.

실제 배포에는 IMU·joint state·contact estimator·command·FK 기반 목표·history buffer를 구성하는 AS2/PiPER 하드웨어 adapter가 필요하다. 현재 `b1z1_interface.py`는 Isaac 환경을 감싸는 인터페이스이며 실제 AS2 motor bridge가 아니다. 텐서/JIT 경로 검증 통과를 실제 배포 완료로 해석하면 안 된다.

## 3. Action 및 최소 수정

| 항목 | 확인 / 조치 |
|---|---|
| 18 action | policy FR/FL/RR/RL leg12 + arm6. native joint 순서 재배열·다리 action scale 유지 |
| 팔 mask | 원본부터 arm6=0. 팔은 DLS IK+position drive로 움직임. 이번에도 유지 |
| 그리퍼 | action에 없음. 별도 열린 위치 제어 유지 |
| action alias 오류 | env.step의 in-place mask가 PPO.transition.actions까지 바꾸던 문제. mask 전 clone하여 원래 sampled action과 old logprob의 일치 복구 |
| delay | 원본 config=3이지만 실제 코드는 global_steps<240000이면0, 이후1 policy-step 지연. 임의로3-step으로 바꾸지 않음 |
| delay=-1 | history 최신 슬롯이 갱신되지 않아 observation action이 stale하던 오류 수정 |
| observe_priv=False | observation 재조립이 if 내부에 있어 stale observation을 반환하던 오류. 최신 proprio/history + zero privileged18로744 유지 |
| JIT slice | interface의 privileged 제거 시작을18→84로 수정. instance args 사용 |
| JIT export | raw actor의 privileged branch 대신 history branch wrapper. bool 타입 주석 추가 |
| 파일 연결 | 예전 JIT 파일이 없으면 현재 로드 모델에서 policy_1.pt export 후 사용 |
| arm default 계수 | 5개를 추출하던 slice를 [12:18]의6개로 수정. 현재 torque supervision=False로 비활성 |

실제 step에서 early/late/disabled 분기를 검사: 발행 leg action0.02일 때 적용값0.02/0/0.02, 관측 최신 action은 모두0.02. caller action 변형 없음; 적용 arm action 모두0.

팔 출력6이 가려져도 확률적 logprob와 surrogate에는 남는다. 팔 출력 gradient가 물리 팔 제어 기여를 뜻하지 않는다. 불필요한 분산이 생길 수 있으나 출력/알고리즘 재설계이므로 이번 범위에서 변경하지 않았다. Arm motor-strength random 값도 현재 IK position-drive gain을 직접 변화시키지 않는다. 실제 PiPER gain/torque/latency 검증은 후속 항목이다.

## 4. Reward / advantage / gradient

- Leg/Arm reward를 각각 scale 합산 후 /100, Nx2로 storage에 저장한다. 두 value head 및 각 채널 GAE와 timeout bootstrap 경로 확인.
- Advantage 정규화는 채널별이 아니라 전체 NxTx2 공동 mean/std다. 원본과 동일하다. 한 채널 평균이 양수, 다른 채널 평균이 음수가 될 수 있으며 단독으로 오류가 아니다.
- 혼합계수 m은 PPO counter 기준0→3000에서0→1. Leg=A_leg+m*A_arm, Arm=A_arm+m*A_leg. 분리 branch 검사에서 m=0의 leg-only loss는 arm head gradient0, arm-only loss는 leg head gradient0; m=1에서는 양쪽 전달 확인.
- PPO backward에서 leg/arm actor head와 dual critic gradient 모두 유한·비영. History encoder gradient0은 PPO branch에서 정상. 별도 DAgger backward는 history encoder만 비영, actor backbone/priv encoder/critic은0.
- Runner는 iteration%20==0에 DAgger만 실행하며 iteration0도 해당한다. 나머지 iteration에서 PPO 업데이트. Privileged regularization 계수는3000까지0, 이후10000까지0.1로 상승. 원본 유지.
- 실제 `ppo.py`, `rollout_storage.py`는 보호된 원본과 SHA256 동일. PPO 수식·혼합·정규화·mask 구조·reward scale 변경 없음.

정량 gradient norm 전체는 `ppo_path.json`에 기록했다. 진단 sampled action과 실제 PD reward를 재생한 경로 시험이므로 loss 크기를 on-policy 학습 안정성으로 해석할 수 없다.

## 5. 정지 정책 위험

`walking_dof=exp(-0.05 Σ|q_leg−q_default|)`는 보행 command 중 기본 자세를 유지하면 높아진다. 이름과 달리 실제 걸음 발생을 요구하지 않는다. +1.464는 준정적 PD에서도 얻을 수 있다. 반면 전진 tracking은 양의 목표에 대해 `min(vx,v_cmd)/(v_cmd+1e-5)`이므로 정지에서 약0, 약간 후진하면 음수다. 관측된−0.012는 보상 계산 누락보다 실제 속도 부족에 해당한다.

위험은 있다. 초기 탐색에서 자세 보상은 쉽게 얻고 움직임에 따른 자세·에너지·접촉·reset 손실은 즉시 생기므로 정지 근처에 머물 수 있다. 다만 절대 reward baseline은 GAE/value로 상쇄되므로 +1.464 대−0.012만으로 정지가 전역 최적이라고 판정할 수 없다.

다른 항목과 walking_dof를 고정한 반사실 계산: vx command0.3 m/s, velocity scale2에서 vx=0/0.05/0.15/0.3이면 속도 reward=0/0.333/1/2. 두 항목 합=1.464/1.797/2.464/3.464. 실제 보행 시 나머지 penalty 변화는 포함하지 않았으므로 이 표는 학습 예측이 아니다. 학습 후 moving-command 구간의 속도·정지비율·보상별 변화·reset을 함께 보아야 한다. 근거가 부족하므로 scale 유지.

## 6. 재현 및 보존

프로젝트 루트에서 다음을 각각 실행한다. 학습 실행 명령이 아니다.

```bash
cd low-level-as2piper-head-orin
export PYTHONPATH="$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python"
export MAX_JOBS=2
python audit_observations.py --label after
python audit_ppo_path.py
```

`before/`, `after/`에 rollout/결과, `ppo_path.json`에 gradient/JIT/optimizer 결과, `changes.patch`에 이번 변경, `diff_stat.json`에 파일별 변경량, `preservation.json`에 원본/URDF 보존 해시를 저장했다. Git status는 저장소 dubious ownership으로 실패하여 공유 Git 설정을 변경하지 않고 파일 diff/hash로 확인했다. 보호된 B1+Z1 원본 소스 불변, training URDF 불변.

사용자 검토 전 실제 PPO 학습·optimizer 업데이트·장시간 실험을 시작하지 않는다.
