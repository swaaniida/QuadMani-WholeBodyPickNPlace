# VBC reward 원본 대조 및 AS2 계측

## 판단과 적용

| 항목 | 판단 근거 | 적용 |
|---|---|---|
| Feet height | 원본은 앞발 2개 중심의 월드 Z 벡터 norm과 0.3 m를 비교한다. 두 발 높이가 같으면 각 중심 약 0.212 m, 평지 clearance 약 0.184 m에 해당한다. AS2에 큰 높이 유도이지만 현재 PD rollout은 보행 성능을 검증하지 않는다. | 지면 상대 좌표로 수정. 0.3 m와 scale +1 유지. 새 목표값 튜닝은 보행 데이터 확보 이후 판단 |
| Contact force | sensor norm은 평지 정지에서 약 2 N인데 실제 net contact force는 발별 약 41~72 N이다. 기존 penalty는 센서 상쇄 때문에 0이었다. | 실제 foot contact force를 사용. soft excess threshold 40 N, scale −0.001 유지 |
| Base height | root world Z를 0.3244 m와 비교해 rough의 지형 고도 영향을 받았다. | 발밑 지면 대비 몸통 높이로 수정. 기하학 목표 0.3244069024 m, scale −5 유지 |
| EE tracking | exp(−2 × TCP 위치 L1 오차 / 1). 작은 오차에 0.95 전후의 값이 실제 기록됐다. | 식·scale 0.8 유지 |
| Velocity tracking | vx는 방향별 min 비율과 정지 exp 분기를 사용한다. yaw는 exp(−yaw 오차²/0.2). 미추종과 일치 입력에 정상 반응한다. | 식·scale 유지 |
| Contact shaping | raw mismatch가 음수이고 scale도 −2여서 활성 시 mismatch가 양의 reward가 된다. cached 발 속도도 초기값 복사본이다. | raw mismatch 부호와 현재 발 속도 읽기를 수정. observe_gait_commands=false 활성 상태 유지 |
| Feet jerk | container의 last_contact_forces 존재를 검사하면서 실제 값은 env에 저장해 항상 0이었다. | env의 이전 실제 contact force를 비교. scale −0.0002 유지 |
| Air time / drag | rough에서 sensor norm 기반 접촉과 실제 contact가 약 56% 불일치했다. | 기존 net-contact 1-step filter(contact_filt)를 reward 접촉 판정에 사용. 관측 차원·관측 foot-contact 필드는 변경하지 않음 |

40 N은 하드웨어 허용 접촉력이나 종료 조건이 아니다. 정상 정지 하중도 이 값을 넘으므로 교정 후 정지 penalty가 발생한다. 기존 식은 soft penalty이며 B1에서도 정상 하중이 기준을 넘을 수 있다. 정상 정지 penalty를 무조건 0으로 만들 근거는 없으므로 새 force 기준을 임의로 확정하지 않았다.

## 실제 rollout

평지·rough 각각 16개 환경, 10초, seed 20261111. 조건은 정지·전진 명령 0.3 m/s·yaw 명령 0.6 rad/s·전진+회전이며 각 4개 환경이다. 초기 다리는 공칭 자세, DR/push는 끄고 랜덤 EE 목표를 생성했다. Rough는 원본 생성기의 4×4 패치다. 다리 action=0인 기본 PD로 reward의 계산·분기를 확인했으며 학습된 보행 성능 측정은 아니다. 아래 값은 2~10초에서 산출했다.

| 지형·조건 | vx 명령 / 실제 (m/s) | yaw 명령 / 실제 (rad/s) | vx reward raw | yaw reward raw | EE L1 오차 (cm) / raw |
|---|---:|---:|---:|---:|---:|
| 평지 stand | 0.00 / -0.0015 | 0.00 / 0.0002 | 0.9973 | 0.9999 | 2.32 / 0.9548 |
| 평지 forward | 0.30 / -0.0018 | 0.00 / -0.0003 | -0.0059 | 0.9998 | 2.64 / 0.9489 |
| 평지 turn | 0.00 / -0.0017 | 0.60 / 0.0002 | 0.9975 | 0.1655 | 2.18 / 0.9574 |
| 평지 forward_turn | 0.30 / -0.0018 | 0.60 / 0.0001 | -0.0060 | 0.1655 | 2.40 / 0.9533 |
| rough stand | 0.00 / -0.0016 | 0.00 / -0.0014 | 0.9951 | 0.9993 | 2.03 / 0.9604 |
| rough forward | 0.30 / -0.0018 | 0.00 / -0.0026 | -0.0059 | 0.9993 | 2.01 / 0.9606 |
| rough turn | 0.00 / -0.0022 | 0.60 / -0.0052 | 0.9967 | 0.1603 | 2.22 / 0.9567 |
| rough forward_turn | 0.30 / -0.0018 | 0.60 / 0.0014 | -0.0059 | 0.1668 | 2.32 / 0.9549 |

- 평지 정지 실제 발 접촉력 평균은 [71.66, 72.03, 43.71, 41.39] N이다. 센서 norm 평균은 [2.36, 2.23, 1.97, 1.98] N이다.
- 평지 정지 base height 평균은 0.30883 m이며 기하학 목표와 약 1.56 cm 차이가 있다. 초기 위치 0.32641 m와 reward 목표 0.32441 m는 구별한다.
- 평지 정지 발 clearance P95는 거의 0 m다. 전진 명령을 주어도 다리 action=0이므로 현재 시험의 발 움직임으로 적절한 swing target을 확정할 수 없다.
- 모든 조건에서 reset은 0이다. 모든 활성 reward·metric 값은 유한하며, raw × scale의 합을 100으로 나눈 값과 실제 Leg/Arm reward buffer가 일치한다.
- net contact force와 발 센서 값은 20 ms 제어 시점에서 기록했다. Rough 접촉은 진동과 제어 주기 사이의 impulse sampling 영향을 받으므로, 해당 평균을 정적 중량 분배와 동일시하지 않는다.

## 계측으로 확인한 계산 오류

- 같은 로봇과 지면의 높이를 함께 1 m 이동시키면 원본 base-height reward는 최대 약 0.97, feet-height reward는 약 0.26 변했다. 교정 후 차이는 1e-7 이하로, 지면 기준 좌표 이동에 불변이다.
- 활성화한 synthetic swing 접촉 mismatch의 weighted 값은 원본 +2.0, 교정 후 −2.0이다. stance 속도 mismatch는 원본 +1.7293, 교정 후 −1.7293이다. 현재 production gate=false이므로 contact shaping의 실 rollout 기여는 계속 0이다.
- 실제 접촉 50 N × 4발, 센서 2 N 입력에서 force excess는 원본 0, 교정 후 40 N이다. 이는 Σ max(F−40,0)와 일치한다.
- 이전 접촉력 0에서 50 N × 4발로 변한 입력에서 force-change metric은 원본 0, 교정 후 200 N이다. 이 term의 기존 이름은 feet_jerk지만 수식은 kinematic jerk가 아닌 step 간 접촉력 변화다.

## 활성화·scale 전체 대조

원본 설정의 49개 Leg/Arm scale 항목 중 25개 함수가 호출된다. 그 중 contact shaping 2개는 gait gate로 0이므로 실질 활성은 23개다. scale=0 및 None인 항목은 _prepare_reward_function에서 제거되어 metric도 기록되지 않는다. 상속된 주석의 “0 scale도 metric 로깅”과 실제 실행은 다르다. 아래 scale은 /100 이전 값이다.

| 항목 | scale | 실행 상태 | 전진 조건 weighted 평균 |
|---|---:|---|---:|
| leg/action_rate | -0.015 | called_finite | -0 |
| leg/alive | 1.0 | called_finite | 1 |
| leg/ang_vel_xy | -0.2 | called_finite | -7.73801e-05 |
| leg/base_height | -5.0 | called_finite | -0.0785964 |
| leg/base_height_standing | 0.0 | inactive_scale | — |
| leg/base_height_walking | 0.0 | inactive_scale | — |
| leg/collision | -10.0 | called_finite | -0 |
| leg/delta_torques | -1e-07 | called_finite | -5.35802e-10 |
| leg/dof_acc | -7.5e-07 | called_finite | -5.30477e-07 |
| leg/dof_default_pos | 0.0 | inactive_scale | — |
| leg/dof_error | 0.0 | inactive_scale | — |
| leg/dof_pos_limits | -10.0 | called_finite | -0 |
| leg/energy_square | 0.0 | inactive_scale | — |
| leg/energy_square_standing | 0.0 | inactive_scale | — |
| leg/energy_square_walking | 0.0 | inactive_scale | — |
| leg/feet_air_time | 2.0 | called_finite | 0 |
| leg/feet_contact_forces | -0.001 | called_finite | -0.068975 |
| leg/feet_drag | -0.08 | called_finite | -0.01563 |
| leg/feet_height | 1.0 | called_finite | -0.260473 |
| leg/feet_jerk | -0.0002 | called_finite | -0.00012877 |
| leg/hip_pos | -0.3 | called_finite | -1.14708e-05 |
| leg/lin_vel_z | -1.5 | called_finite | -9.27349e-05 |
| leg/orientation | 0.0 | inactive_scale | — |
| leg/orientation_standing | 0.0 | inactive_scale | — |
| leg/orientation_walking | 0.0 | inactive_scale | — |
| leg/penalty_lin_vel_y | 0.0 | inactive_scale | — |
| leg/roll | -2 | called_finite | -0.00335016 |
| leg/stand_still | 1.0 | called_finite | 0 |
| leg/torques | -2.5e-05 | called_finite | -0.00946506 |
| leg/torques_standing | 0.0 | inactive_scale | — |
| leg/torques_walking | 0.0 | inactive_scale | — |
| leg/tracking_ang_vel | 0.5 | called_finite | 0.499903 |
| leg/tracking_contacts_shaped_force | -2.0 | inactive_gait_gate | -0 |
| leg/tracking_contacts_shaped_vel | -2.0 | inactive_gait_gate | -0 |
| leg/tracking_lin_vel_max | 2.0 | called_finite | -0.0117683 |
| leg/tracking_lin_vel_x_exp | 0 | inactive_scale | — |
| leg/tracking_lin_vel_x_l1 | 0.0 | inactive_scale | — |
| leg/walking_dof | 1.5 | called_finite | 1.46436 |
| leg/work | -0.003 | called_finite | -0.0183562 |
| arm/arm_energy_abs_sum | None | inactive_scale | — |
| arm/arm_orientation | None | inactive_scale | — |
| arm/arm_termination | None | inactive_scale | — |
| arm/tracking_ee_cart | None | inactive_scale | — |
| arm/tracking_ee_orn | 0.0 | inactive_scale | — |
| arm/tracking_ee_orn_ry | None | inactive_scale | — |
| arm/tracking_ee_sphere | 0.0 | inactive_scale | — |
| arm/tracking_ee_sphere_standing | 0.0 | inactive_scale | — |
| arm/tracking_ee_sphere_walking | 0.0 | inactive_scale | — |
| arm/tracking_ee_world | 0.8 | called_finite | 0.759086 |

- tracking_ee_orn은 scale 0, arm_energy_abs_sum은 None으로 비활성이다. 팔 방향은 IK가 추종하지만 현재 orientation reward는 없다. 팔 position drive torque는 env.torques의 arm 구간에 나타나지 않으므로 현재 torques penalty는 실제 팔 drive 토크를 벌점으로 주지 않는다.
- 비활성 sphere/cart/orientation 파생 reward 중 일부는 env.get_ee_goal_spherical_center, env._reward_* 또는 ee_goal_orn_euler 등 현재 환경에 없는 경로를 참조한다. 현재 scale에서 호출되지 않는다. 활성화 전에 별도 수정·검사가 필요하며 이번 단계에서 활성화하지 않았다.
- walking_dof는 보행 명령 중 기본 관절 자세에 가까울수록 양의 reward다. 이는 원본 설계다. 실제 PPO가 속도 추종과 어떻게 절충하는지는 후속 학습 검증 대상이다.

## 비정상 몸통 접촉 termination 검토

- 현재 실제 terminate_after_contacts_on 목록은 비어 있다. 비정상 접촉은 scale −10의 collision reward에서 body/thigh/calf를 대상으로 처리한다.
- 이번 평지·rough 시험에서 base_link contact peak는 0 N이다. rough에서는 calf 접촉이 관측됐다. 몸통 접촉 종료에 필요한 force·지속시간을 이번 표본으로 확정할 수 없다.
- 몸통을 바닥에 지지하거나 장애물에 걸린 상태를 조기에 끊는 용도로 contact termination을 검토할 수 있다. 실제 충돌·지속 접촉 자료와 단발 mesh contact를 구분하는 검증이 필요하다. 현재 body 목록과 termination threshold는 변경하지 않았다.

## 파일

- reward_inventory.json에 전체 scale, 호출 상태, 원본·수정 수식 소스 및 실측값을 기록했다.
- before/after 각 지형 폴더에 result.json과 samples.npz가 있다. Leg/Arm raw·metric, 높이·clearance·접촉과 body별 힘이 저장돼 있다.
- function_checks.json과 body_contact_summary.json은 독립 입력 검증 및 body별 접촉 최고값이다.
- 변경된 reward는 본 프로젝트의 legged_gym/envs/rewards/maniploco_rewards.py에 있다. Reward scale 및 target 상수는 이번 단계에서 변경하지 않았다.
