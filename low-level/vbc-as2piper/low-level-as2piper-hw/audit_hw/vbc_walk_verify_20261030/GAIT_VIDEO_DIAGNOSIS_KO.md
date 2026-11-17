# model_313 보행 영상 원인 분석

## 판정

실제 정책은 지정한2Hz 대각선 trot 대신 약3.25Hz의 같은 쪽 앞·뒷발 지지가 많은 패턴을 보인다. 발을 과하게 들어 올리고 몸통이 좌우로 흔들리며, 정지·회전 명령을 충분히 따르지 않는다. 추가 계측에서 실제 보상 신호를 훼손하는 코드 결함2개와 정지 접촉 목표의 불일치를 확인했다. 단순한 초기 학습 부족으로만 설명하면 안 된다.

분석 대상은 새 `as2piper_plate_vbc_port`의313회 체크포인트다. 현재 실행 중인 구형 제한 범위 task의 정책 영상을 분석한 결과는 아니다.

## 영상·접촉 계측

평지 nominal, 조건별8환경10초 중 최초2초 제외. 분석 데이터는 `eval_313`에 보존했다.

| 항목 | 측정 |
|---|---:|
| 목표 발 주기 |2Hz|
| 실제 전진 발 접촉 신호의 우세 주파수 |약3.25Hz|
| 목표 접촉 위상과 일치율 |53.6%|
| 같은 쪽 앞·뒷발 두 개로 지지한 시간 |49.2%|
| 대각선 두 발로 지지한 시간 |5.0%|
| 한 발만5N 초과 지지한 시간 |24.9%|
| 네 발 모두5N 미만인 시간 |3.7%|
| 발 높이99백분위 |약7.8–11.5cm|
| 단일 발 최대 Fz |약674N|

지지 비율의 분모는 전체 평가 시간×8환경이다.5N 기준은 접촉력 분류 기준이며, 한 발 지지나 순간 무접촉 자체만으로 실패를 정의하지 않는다. 영상에서 보이는 흔들림·명령 추종 실패와 함께 해석했다. 대표 전진 환경의 body z 범위는0.302–0.343m, roll RMS4.37도였다.

## 확정 결함1: contact velocity reward가 초기 복사본을 사용

위치: `legged_gym/envs/manip_loco/manip_loco.py:761`, 소비: `legged_gym/envs/rewards/maniploco_rewards.py:330`.

`self.rigid_body_state[:, self.feet_indices, 7:10]`에서 tensor index를 사용하는 advanced indexing은 원본 공유 view가 아닌 복사본을 만든다. 이 값을 초기화 때 한 번만 `self.foot_velocities`에 저장하고 이후 갱신하지 않는다. 원본 Isaac Gym tensor를 refresh해도 이 복사본은 갱신되지 않는다.

4초 추가 계측 결과:

- 캐시 속도 변화 최대값:0.0m/s.
- 실제 속도 변화 최대값:3.944m/s.
- 캐시와 실제 속도의 성분 RMS 차이:0.885m/s.
- 대표 환경의 초기 캐시 네 발 수직 속도는 모두 약+1m/s로 남아 있다.
- 전진 조건의 현재 weighted velocity term 평균:-0.95855. 같은 상태에서 현재 속도로 식만 재계산하면-0.53616.

두 값은 공통 `/100` 적용 전이며, 재계산은 정책/물리 상태를 바꾸지 않은 오프라인 비교다. 현재 속도에 기반한 penalty가 아니므로 접지 중 발 움직임을 올바르게 억제하지 못한다. `foot_positions`도 같은 방식으로 고정돼 있지만 이 task의 해당 보상은 live rigid-body 위치를 읽으므로 직접 영향은 구별해야 한다.

## 확정 결함2: feet_jerk penalty가 항상0

위치: `legged_gym/envs/rewards/maniploco_rewards.py:191`.

조건문은 reward container인 `self`에 `last_contact_forces`가 있는지 확인하지만, 값은 environment인 `self.env.last_contact_forces`에 저장한다. container에는 속성이 만들어지지 않아 계속 초기 분기로 들어간다. 실제 계측에서도 weighted `feet_jerk`는 전체 구간0이었다. 발 접촉력 변화 패널티가 작동하지 않는다.

이 두 결함은 원본 B1 VBC의 같은 소스에도 존재하며 AS2 복제본으로 계승됐다. 기존 B1 코드는 수정하지 않았다.

## 확정 설계 불일치: 정지 시 접촉 목표와 부호 수정의 조합

위치: `manip_loco.py:973` 및 `as2piper_vbc_port.py:122`.

정지 때 gait phase를0에 고정하면 Gaussian smoothing 결과 네 발 desired contact가 모두0.5다. 실제 계측에서도 모두0.5였다. 새 port는 contact shaping의 raw sign만 뒤집고 scale-2를 유지하므로, 이 상태에서 하중을 지지하는 발도 force penalty를 받는다.

- 네 발 정지 접지: force term 약-1.0.
- 두 발 정지 접지: force term 약-0.5.
- 가정: 접촉력이 sqrt(sigma)=약0.277N보다 충분히 크고 속도 term은 별도 비교. 실제 하중 지지력에서는 포화 조건을 만족한다.

이 항목만 보면 정지 시 네 발을 모두 지면에 두는 것보다 발을 드는 쪽이 유리하다. 복원 과정에서 정지 시 contact target=1 처리를 제거하면서 부호 보정과의 충돌을 놓친 부분이다. 실제 정지 평가의 force term 평균은-0.4463이었다. 이 불일치가 존재한다는 것은 확인했지만, 이 항목만으로3.25Hz pace가 발생했다고 인과를 단정하지는 않는다.

## 보상 설계상 추가 제약

- `tracking_lin_vel_max`: 목표0.25m/s에 도달하면 포화한다.0.25와0.552m/s 모두 이 항목에서 거의 같은 최대 보상을 받는다. 초과 속도를 이 항목이 교정하지 않는다. 전진 평가 평균 weighted term은1.954/최대2.0이었다.
- `walking_dof`는 실제 보행 여부가 아니라 초기 관절 자세와의 근접도를 보상한다. 전진 평균1.379/최대1.5이며, 이 값이 높다고 좋은 보행을 의미하지 않는다.
- `feet_height`: 앞발2개의 world z norm이0.18855m에 도달하면 penalty가 사라진다. 발별 swing/stance 위상, 지면 상대 높이, 발 반지름을 고려하지 않는다. 한 앞발을 높게 들거나 몸통 높이를 바꿔도 항목이 개선될 수 있다.18.9cm는 개별 발 clearance 목표가 아니다.
- `feet_air_time`: 앞발만0.5초 기준을 사용한다. 목표2Hz/duty0.5의 swing 시간은0.25초이므로 해당 gait의 swing 시간과 직접 맞춘 보상이 아니다. 전진 평균 weighted term은-0.0744였다.

이 항목들은 원본 설계를 유지한 부분이다. AS2에서 어떤 항목이 주된 원인인지, weight를 얼마나 바꿔야 하는지는 별도 비교 실험이 필요하다. 이번에는 production reward를 바꾸거나 재학습하지 않았다.

## 배제하거나 한정한 설명

- foot 순서는FL,FR,RL,RR이며 목표 trot pair도FL+RR/FR+RL로 맞는다.
- 영상의 MuJoCo FK 발 높이와 저장된 PhysX 발 높이 오차는 최대2.38e-7m였다. 발 높이를 잘못 재생한 영상 문제는 아니다.
- 전진 평가에서 torque limit90% 이상 샘플은0%, 최대사용비율은71.3%였다. 현재 자료는 torque clipping을 주원인으로 지지하지 않는다. PD gain의 적절성 전체를 검증한 것은 아니다.
- force sensor 기반 접촉과 net Fz>5N 분류의 일치율은약96.5%다. 접촉 감지가 전부 고정돼 있던 문제는 아니다.
- arm contact>1N은 감시한 link에서 관찰되지 않았다. 모든 자기충돌을 배제하는 검사는 아니다.

## 수정 우선순위

1. 매 physics refresh 후 live foot velocity를 사용하도록 데이터 경로 수정.
2. feet_jerk의 저장/확인 객체를 일치시키고 reset 초기화 처리.
3. 부호 보정된 contact reward와 정지 시 네 발 지지 목표를 일관되게 구성.
4. 먼저 동일 조건에서 위 수정만 비교. 이후 과속, 발 높이/air time 설계 변경은 원본 대비 차이를 명시한 별도 실험으로 검증.

## 산출물

- `gait_video_contact_sheet.jpg`: 실제 영상 발췌.
- `gait_phase_diagnosis.png`: 목표/실제 접촉 순서와 발 높이.
- `video_gait_diagnosis.json`: 접촉·몸통·주기 계측 및 식 검산.
- `reward_inspection/reward_summary.json`: 기존 보상 반환값을 바꾸지 않고 기록한 항목별 평균.
- `cache_inspection/cache_findings.json`: 캐시 vs live 속도 직접 비교.
- `inspect_policy_rewards.py`: 계측 wrapper. production 코드에는 monkeypatch를 저장하지 않으며 진단 프로세스 안에서만 적용.

기존 학습 프로세스 및 B1 source는 변경하지 않았다. 분석 파일만 추가했다.
