# Gait Clock / Contact Shaping 활성화 검증

## 변경

작업 대상은 low-level-as2piper-head-orin-nocontact. Contact observation4+history40 제거 상태를 유지하며 gait index1+clock4만 추가했다. Proprio67 + privileged18 + history670 = Training755. Privileged를 제외한 Deployment737. Contact zero padding 없음.

정확한 구간: IMU0:5, q5:23, dq23:41, 이전 leg command41:53, velocity command53:56, EE goal56:59, 기존 orientation zero59:62, gait index62:63, clock63:67, priv67:85, history85:755. 배포는 cat(obs[:,:67],obs[:,85:]). Actor 입력67+latent20=87, Critic67+priv18=85, History frame67·10개→latent20. Action18 유지.

AS2 config observe_gait_commands=True, num_proprio=67, num_observations=755. 생성자가 cfg.num_proprio에5를 반복 더하던 방식을 실제 contact/gait flag에서 차원을 유도하도록 바꾸어 config 재사용 시 중복 증가를 방지했다. 기존 Actor/Critic·History·DAgger·JIT 코드는 config 차원을 따라 연결되며 PPO 파일은 수정하지 않았다.

실제 센서에 의존하는 legacy foot_contacts_z 함수는6축 force_sensor_tensor를 읽는다. 기존에도 scale 미등록으로 비활성이었으며 이번에 scale0을 명시했다. 그 외 active contact 함수는 net contact force/contact_filt 또는 live rigid-body foot velocity를 쓰므로 유지했다. 실제 발 센서가 필요한 활성 reward는 남아 있지 않다. 시뮬레이터의 센서 생성 자체는 유지하되 actor/배포 입력으로 쓰지 않는다.

Config 비교 결과 변경된 설정은 env.observe_gait_commands, env.num_proprio, env.num_observations, rewards.scales.foot_contacts_z 네 항목뿐이다. 나머지 reward 값·gain/DR·initial pose·URDF·PPO 설정은 동일하다. feet_height_target=0.30, max_contact_force=40, shaping scale 각각−2, frequency2Hz 유지. 이전 추천값0.15/200N은 적용하지 않았다.

## Contact Shaping

기존에 수정해 둔 함수를 변경 없이 활성화했다.

- Force raw = mean[(1−desired_contact)×(1−exp(−||net_contact_force||²/gait_force_sigma))].
- Velocity raw = mean[desired_contact×(1−exp(−||live_foot_velocity||²/gait_vel_sigma))].
- Raw mismatch≥0, scale−2이므로 weighted≤0. Mismatch 증가 시 penalty 증가. 원본 B1의 음수raw×음수scale 문제는 재도입하지 않았다.
- Force는 contact_forces[:,feet_indices,:]의 실제 simulator net contact vector, velocity는 rigid_body_state[:,feet_indices,7:10]을 매 호출 직접 읽는다. 이전 foot_velocities cache는 사용하지 않는다.

각 실제 reward 호출을 독립 analytic 식과 비교하여 일치 확인. 센서 tensor와 stale velocity cache를 NaN으로 채운 shadow 환경에서도 정상 계산했다. Probe 결과 (/100 이전): swing 중50N force weighted−2, stance 중50N force0, stance 중1m/s weighted−1.729329, stance0m/s0. 즉 올바른 접촉 phase와 실제 신호 변화에 반응한다.

16환경×4초 rough PD/IK rollout에서 정지/전진/회전/복합 command를 각각4환경에 공급했다. 학습된 보행 정책 시험이 아니며 다리action=0, 팔IK다. 비영 출력과 phase 변화만 확인한다. 아래 weighted는 /100 이전 값이다.

| 명령 | force raw | force weighted | vel raw | vel weighted |
|---|---:|---:|---:|---:|
| stand | 0.19467 | -0.38934 | 0.01093 | -0.02187 |
| forward | 0.18653 | -0.37307 | 0.01108 | -0.02216 |
| turn | 0.29368 | -0.58735 | 0.00398 | -0.00795 |
| forward_turn | 0.16987 | -0.33974 | 0.00312 | -0.00624 |

정지에서도 원본 함수와 같이 별도 walking gate가 없어 shaping penalty가 발생한다. 해당 동작은 이번에 임의 변경하지 않았다. Force sigma0.5도 유지되어 작은 접촉력에서 지수항이 빠르게 포화한다. 계산 정상 여부와 보행 성능·reward 최적성은 별개이며 실제 학습 성능은 검증하지 않았다. Rough net-contact 보고의 불확실성도 유지된다.

## 관측·배포·학습 경로 검증

- 전체755 field·normalization·history 위치·reset history 시간 계약 검사. Clock5가 이동 명령에서 실제 변화하고 관측62:67과 일치함을 확인했다.
- Contact bit 반전 및 sensor NaN 주입 시 전체755 observation 불변. History도67개만 저장하며 contact4를 재삽입하지 않는다.
- 실제 runner 생성: storage[24,16,755], native inference action[16,18], Adam state 비어 있음.
- PPO/DAgger 실제 forward/backward 경로 실행 후 optimizer.step 직전에 중단. Leg/Arm actor·dual critic gradient, history student gradient 및 advantage routing 검사 통과. 모든 파라미터 불변, optimizer step0.
- Native history와 JIT737 출력 최대오차0. Privileged 값 변경 시 history 출력 변화0. 이전700/744 Actor/Critic 입력,682/726 JIT 입력 거부 확인. 기존 checkpoint/JIT는 새 입력 계약과 호환되지 않으며 padding/부분 이식하지 않았다.

실물 clock는 접촉 센서 없이 command와 dt로 생성한다. 원본 _step_contact_targets와 같이 이동 시 phase=(phase+dt×2)%1, 정지 명령이면 phase0. Clock native foot order는 FL/FR/RL/RR이며 대각 다리에0.5 phase offset을 둔다. 정책 joint 배열 순서와 clock 배열 순서를 혼동하지 말고 학습 순서 그대로 구성해야 한다. 실제 AS2/PiPER SDK motor bridge는 이번 작업 범위가 아니다.

## 보존·재현

before_source/에 이번 변경 전700차원 설정·관측 코드·진단 스크립트 백업. changes.patch와 diff_stat.json에 diff. config_changes.json에 실제 설정 차이. preservation_after.json에 기존 B1 checkpoint·744 AS2 버전·URDF·gain·reward 소스·PPO 보존 hash 결과. 이전 contact_removal/ 보고서·데이터는 보존했으며 현재 코드 상태는 이 gait_clock 보고서가 우선한다.

새 복제본 루트에서 PYTHONPATH="$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python"로 python audit_observations.py, python audit_ppo_path.py를 실행하면 구조 검증을 재현한다. 감사 스크립트의 optimizer는 차단되어 weight를 업데이트하지 않는다. 실제 학습용 run_head.sh와 learn()은 실행하지 않았다.

Git status는 기존 dubious ownership 오류로 실행되지 않아 Git 설정을 바꾸지 않고 파일 hash와 별도 diff로 확인했다. 사용자 검토 단계에서 중단한다.
