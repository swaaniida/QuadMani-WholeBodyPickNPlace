# AS2 head + Orin VBC low-level 변경 기록

## 현재 적용 상태 — B Full Baseline

현재 학습 대상으로 선택한 조건은 **B 하나**다. 아래 값이 최신 상태이며, 뒤의 절들은 시간순 변경 이력이다. 이력에 적힌 “당시 미적용/학습 미실행/0회”는 그 검증 당시의 기록이며 현재 run의 진행 상태를 뜻하지 않는다. 이번 문서 감사에서는 학습 프로세스를 점검하지 않았다.

| 항목 | 원본 B1+Z1 / 과거 상태 | 현재 AS2 B 조건 |
|---|---|---|
| 하드웨어 | B1+Z1·그리퍼1 DOF | 머리 PiPER+UMI·등 Orin, 총20 DOF/정책18 action |
| 접촉 관측 | 현재4+history40 | 완전 제거, contact zero padding 없음 |
| Gait 입력 | 기본off, 실제 B1 45000 checkpoint는on | on, phase1+sin clock4,2Hz |
| 관측 차원 | 원본기본744/배포726; B1완료모델799/781 | proprio67+priv18+history670=755, 배포737 |
| Feet height | target0.30 | target0.18m, scale+1.0 유지 |
| Force reward | sensor6 norm−40, scale−0.001 | simulator net force norm−200N, scale−0.001 유지 |
| Contact shaping | 원본 negative raw×negative scale | mismatch raw≥0×−2, 정지/deadband는walking_mask로0 |
| Contact timer/drag/jerk | sensor/이전상태 오류 포함 | net contact_filt/net force/live foot velocity; 아래 세부 이력 참조 |
| PPO | 원본 dual reward/GAE/mixing/DAgger | 알고리즘 유지, 관측/저장/배포 경로 오류만 수정 |
| 학습 조건 | 원본6144환경·45000iteration | 동일, seed1 공통초기weight strict load |
| 실사용 진입점 | 기존 train.py/run_head.sh | run_force_ablation.sh --mode train --condition B |
| W&B | 시스템 SDK0.15.12 | 전용.wandb_deps SDK0.22.3, project as2piper-low, entity quadruped_capstone |
| Run 표시 이름 | 실험별 | as2piper_headpiper_orin_200N_0.18m |

`run_head.sh`의 기본 태스크는 공통 No-contact 태스크로 threshold40N이며, **B=200N은 force_ablation/configs.py와 전용 진입점에서 선택**한다. A/C는 비교 구성만 보존하며 학습 대상으로 선택하지 않았다.

기본 train.py의 wandb.save 처리와 별개로, 실제 B ablation 경로는 전체 설정·공통초기weight hash·source hash를 run manifest에 저장하고 추가 지표를 W&B/ablation_metrics.jsonl에 기록한다. Native PPO/DAgger 업데이트·checkpoint 저장은 기존 runner를 사용한다. 새 run 경로는 기존 이름이 있으면 덮어쓰지 않는다.

## 원본 대비 소스 감사 및 문서 범위

- [변경 파일·함수 목록](validation/source_audit/SOURCE_INDEX_KO.md)
- [현재 통합 런타임 diff](validation/source_audit/original_vbc_to_current.patch)
- [파일별 SHA256·심볼 목록](validation/source_audit/source_changes.json)

원본 low-level과 rsl_rl을 실제 파일로 대조해 수정/추가된 런타임 Python과 launcher를 목록화했다. 예전 changes.patch는 중간 단계 자료이므로 현재 전체 변경으로 간주하지 않는다. URDF·메시 교체, 제어 gain/물성/DR 산출, 패키지·검증 스크립트와 검증 결과는 아래 설명 및 연결된 보고서 범위다. 과거 문서의 상대 경로 중 복제본에 없는 geometry/workspace/rollout 검증 파일은 원래744버전 `../low-level-as2piper-head-orin/`에 보존되어 있다.

실제 PiPER torque/gain/latency·실기기 motor bridge·장착 하중 경로의 현물 검증은 완료 항목이 아니다. 학습용 URDF 수치·시뮬레이션 검사와 실제 하드웨어 검증을 구분한다.


## 기준 코드와 복제

- 기준 코드 `/data/team/users/suhwan/vbc-legacy/low-level` 및 `third_party/rsl_rl`을 이 디렉터리에 독립 복사했다.
- head+Orin URDF, 메시 및 기존 Python 3.8 보조 의존성 `.deps`를 독립 복사했다. Isaac Gym 런타임은 기존 공유 설치 경로를 참조한다.
- 원본 Python 소스 해시는 `baseline_hashes.json`에 기록하고 수정되지 않았음을 확인했다. 기존 파일의 변경 diff는 `changes.patch`에 있다. 새 설정·검증 파일은 아래에 별도 기록했다.

## 적용한 변경

| 대상 | 변경 내용 | 이유 |
|---|---|---|
| 로봇 자산 | `as2piper_head_orin_training.urdf` 사용, 총 질량 약 23.328 kg | 머리 PiPER·UMI와 등 뒤쪽 Orin 물성 반영 |
| Fixed joint | TCP 및 네 발 joint에 `dont_collapse=true` | TCP·발 링크와 센서 기준 보존 |
| 초기 다리각 | hip 0, thigh 0.8, calf −1.6 rad | AS2 기하학 기준 자세 |
| 초기 팔각 | [0, 1.1, −0.55, 0, 0, 0] rad | 계산한 초기 TCP와 CoM 기준 자세 |
| 초기 그리퍼 | [+0.049, −0.049] m | UMI 열린 상태 |
| 높이 | 접지·reward 기준 0.3244069024 m, reset 0.3264069024 m | 네 발 collision 구의 밑면 접지 및 2 mm 여유 |
| 팔 기준점 | [0.22435, 0, 0.1319] m | head 장착 URDF의 실제 arm base 위치 |
| 좌표 변환 | 팔 기준점 위치에 몸통 전체 quaternion 반영, 구면 목표 축은 yaw 기준 | 몸통 roll/pitch 시 장착점 이동 반영 |
| 구면 중심 | 원본의 지면 기준 z=0.7 m 중심을 실제 arm base 위치로 변경 | 목표 생성과 관측의 기준점 정합 |
| 초기 구면 목표 | [0.4278984687 m, 0.4216862841 rad, 0] | 초기 실제 TCP 위치와 목표 일치 |
| 목표 방향 | roll offset 0, `arm_induced_pitch=2.4552161377` rad | PiPER TCP 축 방향과 초기 목표 회전 일치 |
| EE 범위 | 반경 [0.40,0.48] m, 고도각 [0.10,0.50], 방위각 [−0.25,0.25] rad | 후보 범위 중 IK·충돌 표본검사 통과 범위 적용 |
| 방향 변화 | roll/pitch/yaw 각각 ±0.08 rad | 선택 위치 범위와 결합한 IK 수렴 확인 |
| 경로 충돌 검사 | 원본 구면 보간·10점 검사를 사용하고 검사 좌표를 실제 몸통 좌표로 변환 | 몸통 자세 변화와 새 장착 위치 반영 |
| 검사 경계 | 몸통·머리·Orin을 포함한 보수적 AABB 적용, 지면은 terrain 높이 측정값 사용 | 새 구조의 TCP 경로 제한 |
| 관절 계약 | 실제 joint 이름·순서 assertion 추가, 20 DOF·18 action·그리퍼 2개 | 배열·Jacobian slice·제어 대상 정합 |
| reset | 팔·그리퍼는 초기 FK 자세로 설정, 전체 관절 한계 clamp | 초기 EE 목표 불일치 및 한계 초과 방지 |
| 팔 제어 | IK 위치 목표를 joint limits로 clamp | 유효한 관절 위치 목표 전송 |
| 그리퍼 제어 | 두 손가락의 position target을 열림 값으로 유지 | 원본의 0 target이 닫힘을 유발하는 문제 수정 |
| 발 센서 | local z offset −0.05 → −0.028 m | AS2 발 collision 구 반경 반영 |
| 질량·CoM 대상 | `props[1]` 대신 이름으로 찾은 `base_link` 인덱스 사용 | fixed joint 병합 후 인덱스 변화 대응 |
| 다리 Kp | hip/thigh/calf [80,80,92.48896] Nm/rad | 정적 하중·처짐·관성 기반 계산 |
| 다리 Kd | [2.94118,2.77572,2.78504] Nms/rad | 대각 관성 모델의 감쇠비 0.7 가정 |
| 다리 armature | [0.026,0.026,0.038] kg·m² | AS2 공식 시뮬레이션 프리셋 반영 |
| 추가 몸통 질량 | 0~2.33281 kg | 전체 질량의 10%를 설계 범위로 사용 |
| CoM 변동 | base_link 기준 x ±0.02809 m, y/z ±0.02192 m | 전체 CoM 이동을 지지 반폭의 5% 기준으로 환산 |
| Push 설정 | 축별 기본 속도 ±0.39639 m/s | B1/AS2 높이 비의 제곱근으로 속도 환산 |
| 실행 기본값 | `stop_update_goal=false`, `pitch_control=false`, `env_spacing=3` | CLI 외 직접 환경 생성 경로의 누락 설정 보완 |
| 평지 생성 | plane 분기와 평지 env origin 처리 추가 | 원본의 trimesh 고정 생성 경로 보완 |
| 지면 조회 | `measured_heights`가 없으면 env origin 높이 사용 | 평지 EE 재샘플링 오류 수정 |
| 진입점 | `as2piper_head_orin` 태스크 및 `run_head.sh` 추가 | 독립 실험 실행 경로 |
| 실행 경로 | 로컬 코드·`.deps`·복제 rsl_rl 우선, 전용 Torch extension 경로 | 복제본 의존성 연결 |
| W&B 소스 기록 | 새 `as2_head_orin_config.py`도 저장 | 실행 설정 추적 |
| 토크 계측 | 요청 시 PhysX DOF force sensor 활성화 | 팔 position drive의 실제 출력 기록 |

## 상속한 설정

PPO 알고리즘, reward scale, 다리 action scale, 팔 DLS IK와 position drive 400/40, terrain 생성 설정과 gait 관측 기본값은 원본에서 상속했다. 높이 reward 기준은 위 AS2 값이다. 마찰계수 0.3~3.0, motor strength 0.7~1.3, 다리 reset 0.8~1.2배도 원본 범위다.

초기 검증은 원본 정지 push 2.5배를 사용했다. Reset/Push 후속 검증 이후 현재 기본 배율은 1.0이며 축별 상한은 ±0.39639 m/s다. 2.5배는 별도 stress 조건으로 기록했다.

## 검증 및 검증 전용 변경

- `check_geometry.py`는 초기 FK·목표 회전·접지 높이·관절 한계·메시 경로를 확인한다. 위치 오차 약 2.32e-8 m, 방향 행렬 최대 차이 약 8.34e-8이다.
- `validate_workspace.py`는 학습 밖에서 후보 범위별 위치 4³ × 방향 3³ = 1,728개 자세를 검사한다. 선택 범위와 128개 보간 경로에서 수치 IK 실패·비인접 링크 충돌은 0개다. 직접 연결된 링크의 조립 중첩은 제외한다. 충돌 모델은 URDF collision 형상의 MuJoCo convex mesh 표현이다.
- `validate_rollout.py`는 조건별 16개 환경 × 12초를 평가했다. 정지 및 랜덤 EE 목표는 reset 0개, 평지 DR 및 소규모 rough DR은 push 이후 각각 reset 3개다. 팔 접촉·측정 effort 포화는 0이다.
- rollout은 다리 action=0의 기본 PD와 팔 IK로 수행했다. hold/ee의 초기 다리는 공칭 자세, 초기 속도는 0으로 설정했다. DR은 원본 다리 reset·초기 속도 변동을 포함한다. rough 검증은 원본 생성기를 4×4 패치로 줄였다. 이 시험 전용 설정은 학습 config의 terrain 크기 등을 덮어쓰지 않는다.
- 검증 전용 DOF force sensor로 팔 position drive 토크도 측정했다. 자세·오차·접촉·토크·reset 시각은 `validation/*_rollout.json` 및 NPZ에 기록했다.
- 범위 검사 결과·계산 가정·시험 조건·전체 수치는 `validation/REPORT_KO.md`, `workspace.json`, `hardware.json`에 있다. `validation/hardware.json`은 실제 태스크 config가 읽는 gain·외란 설정 파일이다.
- 원본 보존 결과와 검증용 자산 해시는 `validation/preservation.json`에 기록했다.

## 남은 작업

- Push 이후 reset 원인 분석과 학습된 보행 정책의 외란 회복 평가.
- 학습 중 보행·EE·접촉 보상 신호의 정상 작동 및 PPO 업데이트 확인.
- 지면 줍기용 EE 범위와 전신 동작 확장·검증.
- 실제 PiPER 토크 정격·구동 gain·하드웨어 지연 확인. URDF의 팔 effort 100 Nm는 실측 정격으로 확인되지 않았다.
- 장시간 학습, 학습된 정책의 평지·rough·이동 중 EE 추종 평가.

기하학 검사와 표본 rollout 결과를 기록했으며, 학습된 보행 성능과 연속 공간 전체의 무충돌은 아직 검증하지 않았다.

## Reset/Push 후속 반영

- Config 임계값 실제 연결, 발밑 지면 상대 높이 종료, 원인별 episode 로그와 종료 상태 기록을 추가했다. 초기 자세·reset 높이·URDF·TCP·IK·관측·joint mapping은 유지했다.
- 정지 판정의 signed-sum 상쇄 문제를 수정하고 standing_push_multiplier 기본값을 1.0으로 조정했다. 실제 이전/적용/delta 속도를 기록한다.
- 독립 강도 시험에서 평지 0/1/1.5배는 reset 0/32, 2.5배는 4/32다. Rough 0/1배는 reset 0/32, 2.5배는 6/32다. 최종 시험의 push 전 reset은 없었고 종료 원인은 roll/pitch였다.
- 자세 임계값 0.8 rad, 종료 상대 높이 0.1 m를 유지했다. 접촉 종료 대상은 기존처럼 비어 있다.
- 시험 조건·이벤트·잔여 한계는 validation/reset_push/REPORT_KO.md에 있다. Reward/PPO/학습 검증은 후속 단계다.

## Reward 후속 검토·수정

- 원본 scale 49개와 활성 함수 25개를 대조하고 평지·rough 각각 16환경 × 10초를 계측했다. Contact shaping 2개는 현재 gait gate로 0이다.
- 몸통·발 높이는 지면 상대 좌표로 수정했다. Feet contact force는 센서 net-force 상쇄 대신 실제 foot contact force를 사용한다. Air-time/drag reward의 접촉 판정에는 기존 contact_filt를 사용한다.
- Feet jerk의 잘못된 이전 상태 존재 검사와 force 신호를 수정했다. Contact shaping은 mismatch 부호와 현재 발 속도 읽기를 고쳤다. 활성 flag와 관측 차원·관측 foot-contact 필드는 유지했다.
- feet_height_target=0.3, max_contact_force=40 및 reward scale은 유지했다. 학습된 보행 없이 새 swing 목표·force threshold를 임의로 정하지 않았다.
- 몸통 접촉 termination은 검토 결과만 기록하고 설정을 유지했다. 계산식·활성 상태·실측·판단 근거는 validation/reward_review/REPORT_KO.md와 reward_inventory.json에 있다.

## Observation / PPO 경로 후속 검증

- 위 Reward 단계의 “관측 foot-contact 필드 유지”는 당시 상태다. 이번에는 차원·위치를 유지하면서 값만 reward와 같은 contact_filt로 연결했다. Rough observation/reward 불일치 42.41%→0%; 기존 sensor는 힘·토크 norm과 내부 반력 상쇄로 접촉과 불일치한다.
- 744=현재66+privileged18+history660 전체 필드·joint 순서·고정 scale·history reset/time 계약 확인. Privileged 비활성 시 stale observation 수정. 배포는 privileged를 제외한726이며 실제 AS2 센서/모터 adapter는 아직 없다.
- env.step mask 전 clone으로 PPO 저장 action의 in-place 변형 수정; delay=-1 history 갱신 오류 수정. 원본 delay0→1step 스케줄과 arm6 mask/IK 구조 유지.
- JIT history branch export·726 입력 계약·privileged 제거 slice·native history inference 기본값 수정. Arm default position slice5→6 보정. 예전 JIT는 재생성 필요.
- PPO/DAgger 실제 backward·Leg/Arm advantage routing·critic·timeout bootstrap·JIT 동일성 확인. optimizer step0, 파라미터 불변, Adam state 비어 있음. ppo.py/storage.py 원본 SHA256 동일; reward scale 유지.
- walking_dof는 보행 자체를 요구하지 않아 정지 탐색 위험이 있다. PD 수치만으로 학습 후 정지 여부를 확정할 수 없어 scale 유지. 팔 actor6은 원본처럼 마스킹되어 gradient와 실제 제어 기여를 구분해야 한다.
- 상세 근거·한계·재현 명령: validation/observation_ppo/REPORT_KO.md. 실제 PPO 학습은 사용자 검토 이후 진행한다.

## Contact observation 제거 전용 복제본

- 원본744 버전은 low-level-as2piper-head-orin에 보존. 이 디렉터리는 low-level-as2piper-head-orin-nocontact, task=as2piper_head_orin_nocontact.
- Contact4+history40 실제 제거: proprio62, privileged18, history620, 학습700, 배포682. 삭제 슬롯 zero padding 없음. 입력 크기 검사를 추가해 이전744/726을 거부.
- Actor82/Critic80/history frame62, 실제 runner/storage700, DAgger·PPO backward 및 JIT parity 검사. optimizer0, 파라미터 불변.
- Reward 소스·scale·gait(False)·PPO·URDF·gain·초기 자세 유지. 실물 foot sensor는 새 actor 입력에 없음.
- B1 45000의 swing 높이 계측과 AS2 후보 사후 비교를 수행. Height0.15, force200N은 후속 실험 추천이며 반영하지 않음. 현재0.30/40N 유지.
- 상세: validation/contact_removal/REPORT_KO.md. 실제 학습 미실행.

## Gait Clock 755/737 후속 활성화

- Contact4+history40 제거 유지. Clock index1+sin4 추가: proprio67, priv18, history670, 학습755/배포737. 삭제 contact 슬롯 padding 없음.
- observe_gait_commands=True. Config 초기화 때 차원 반복 증가 방지.
- 수정된 shaping force/vel 활성화. Raw≥0×scale−2→penalty, 실제 net contact force/live foot velocity 참조 확인. Reward source와 기존scale는 변경하지 않음.
- Sensor-only foot_contacts_z는 기존 미등록 비활성 상태를 scale0으로 명시. 기타 simulation contact reward 유지.
- 실제 runner·Actor/Critic·History·DAgger·JIT 및 센서 교란 독립성 통과. optimizer0, 파라미터 불변. 기존700/682 checkpoint/JIT와 호환되지 않음.
- Feet height0.30, force40N,2Hz 및 PPO/다른reward 유지. 실제 학습 미실행. 상세 validation/gait_clock/REPORT_KO.md.

## Feet Height 0.18 / Contact Shaping 정지 마스크

- feet_height_target0.30→0.18. Base-height reward 비0.590, 다리 길이 비0.608 근거. Feet height scale+1.0 및 수식 유지.
- Shaped force/vel에 기존 walking command mask 적용. 정지/deadband0, 전진·후진·순수 회전·복합에서 기존 penalty 계산.
- 관측755/배포737, contact44 제거, Gait/나머지 reward/PPO/초기 자세 유지. 실제 학습 없음.
- 16환경×4초 계측 및 probe 검증: validation/feet_height_walkmask/REPORT_KO.md. 이전 정지 shaping penalty 설명을 대체.

## Contact Force Reward Ablation 준비

- 기존production코드수정없이force_ablation모듈·run_force_ablation.sh추가. A40N/B200N/Cforce scale0을별도태스크/run으로분리.
- Height0.18·scale+1, shaping walking_mask·scale−2, DR/Gait/PPO/755·737입력유지.
- Seed1공통미학습weight strict load,초기/최종모델hash및DR tensor hash가세조건동일. 실제reward합산·정지shaping0·150/250N probe검증.
- 속도/EE/reset/force분포/reward기여도계측및checkpoint평가경로준비. Native PPO/DAgger유지.
- Prepare·소규모PD검증·미학습fixture평가만실행. 실제학습/optimizer step0. 상세force_ablation/VALIDATION_KO.md.

## W&B 초기화 호환성 수정

- 구형SDK0.15.12 server-settings 조회 오류 확인. 전역SDK대신 .wandb_deps/wandb0.22.3을전용launcher에서사용. Entity quadruped_capstone 명시.
- 실패로그/manifest보존. Offline backend·prepare검사통과, 온라인인증은root사용자재실행에서확인필요. PPO/DR/reward/공통가중치불변, 학습재시작하지않음. 상세force_ablation/WANDB_FIX_KO.md.

## B 재시작 요청: 정지 push 2.5배 복구

- standing_push_multiplier1.0→2.5 복구. 기존 velocity-push 정의 유지, 이동 명령의push 배율1은불변. Config대조·같은샘플probe로정지만2.5배확인. 나머지B200N/height0.18/6144환경/45000iteration/PPO/scale불변.
- 사용자요청으로현재B의학습로그·checkpoint·W&B run삭제후공통초기weight(seed1)에서iteration0 새시작준비. 기존B1+Z1 baseline은삭제대상아님.
- 현재학습PID1255859가root소유여서suhwan은kill권한이없고sudo도없음. 아직stop/delete/restart실행하지못함. root전용restart_B_push25.sh가대상B와logger만종료→해당remote run및local학습기록삭제→detached B재시작을수행하도록작성. --dry-run으로삭제범위검증.
- 원래표시이름유지,삭제된W&B run ID재사용충돌방지위해새randomID를manifest에기록. 학습설정변경아님.
- 변경전config/run.py코드는force_ablation/before_push25_*.txt로보관. 검증결과force_ablation/push25_verification.json.

## 2026-11-14 바닥 중심 EE trajectory 비교 (평가 전용)

- ground_trajectory_compare.py / render_ground_compare.py 추가. 10k·20k·30k·45k 동일 지면고정 EE pose 경로64초, base velocity command0, nominal extrinsics·DR/push off.
- 상단부터 바닥1cm까지, 48초는바닥근처동작. 실제 PhysX기록·CPU replay 영상, 초기상태편차와실패reset별도기록. 학습코드·reward·PPO·URDF·checkpoint 변경없음.
- 산출물: ../reports/ee_ground_compare_20261114/REPORT_KO.md 및 ee_ground_compare_64s.mp4 (프로젝트루트reports하위).

## 2026-11-14 기록 기반 3D 재생 뷰어

- reports/ee_ground_compare_20261114/viewer/ 추가. 원본 시각 형상31개·25Hz/64초 body pose를 내보내 Three.js로 재생.
- 단일/4분할 checkpoint 전환, 연결 시점·회전·확대·시간/속도 조절. 한국어 글꼴·JS 로컬 포함. 로컬 HTTP8774.
- 새 학습·물리 rollout·URDF/보상/PPO/기존 checkpoint 변경 없음. Chromium/WebGL 실제 조작 검증 결과는 viewer/browser_checks.json.

## 2026-11-14 EE+보행 36초 평가 추가

- locomotion_trajectory_compare.py 추가. vx/vy/yaw 전후·좌우·양회전·대각·복합 명령과 바닥 EE 동시 평가. 같은 상대EE궤적, xy는body/yaw 추종·z는ground 기준.
- 8774 뷰어에 motion36 실험선택·명령/실제속도·명령화살표·로봇별따라가기 추가. ground64 기록 보존.
- 학습코드·보상·PPO·URDF·checkpoint 변경없음. 자료 reports/ee_locomotion_compare_20261114/, 3D데이터 기존viewer/datasets/motion36/.

## 2026-11-14 보행+EE 평가 vy 명령0 복구

- locomotion_vy0_compare.py 추가. 36초 vx/yaw 일정·EE경로유지,vy명령만전구간0. 물리step/저장기록 전체에서비영vy명령0개확인. 실제vy는강제고정하지않고표시.
- render_locomotion_compare.py 추가. 명령/실제vx/vy/yaw를표시하는영상재생.
- 8774기본 motion36_vy0로갱신,기존motion36도새vy0로alias. 이전비영vy자료는보관. 학습코드·PPO·보상·URDF·checkpoint변경없음.
