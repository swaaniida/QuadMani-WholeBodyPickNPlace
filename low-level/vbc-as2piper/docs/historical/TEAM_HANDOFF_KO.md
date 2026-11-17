# AS2 + Head PiPER + UMI + Orin — 현재 VBC Low-level 학습 인계

기준 시각: **2026-11-12T10:27:06.640906+00:00 (UTC)**. 프로젝트: `/data/team/users/suhwan/vbc-as2piper`.

## 현재 사용할 버전

- 학습 코드: `low-level-as2piper-head-orin-nocontact/`.
- 프로젝트: **VBC low-level + ROA**, AS2 머리 PiPER/UMI + 등 중앙 Orin. Deep WBC 및 Unitree 공식 gait 학습과 별도.
- 실행 조건: **B(발 접촉력 threshold 200N)** 단독. A=40N/C=scale0은 구성만 있으며 실행 대상 아님.
- W&B project: **as2piper-low**, 표시 이름: **as2piper_headpiper_orin_200N_0.18m**.
- 현재 run ID: **gdr8lou4**. [W&B 실행](https://forge.coreweave.com/wandb/quadruped_capstone/as2piper-low/runs/gdr8lou4).
- 문서 작성 시 로그 iteration: **18924/45000**. 최신 저장 체크포인트: **model_18800.pt**. 계속 진행하는 실행이므로 고정 최종값이 아님.
- 기존 B1+Z1 원본 `/data/team/users/suhwan/vbc-legacy`는 읽기 전용 보존 대상.
- 과거 AGENTS.md/실험 문서의 중앙 장착·744차원 등 수치는 이 실행의 최신 설정과 구분. 실제 run manifest와 아래 스냅샷이 현재 구성의 기준.

## 같이 전달할 문서

- [전체 필수 파일·URDF 메시 37개·SHA256 목록](docs/current_training/ESSENTIAL_FILES.md)
- [현재 run의 전체 설정 스냅샷 및 Python 패키지 버전](docs/current_training/TRAINING_CONFIG_SNAPSHOT.md)
- [원본 VBC 대비 변경 이력](low-level-as2piper-head-orin-nocontact/CHANGES_KO.md)
- [10,000회 평가 보고서](low-level-as2piper-head-orin-nocontact/experiments/contact_force_ablation/evaluations/B_iter10000_report_20261112/REPORT_KO.md)

이 Markdown들은 파일 내용 전체를 대체하는 실행 패키지가 아니다. 다른 서버에서 실행하려면 파일 목록의 코드·URDF·메시·선택 체크포인트와 의존 환경을 함께 준비해야 한다.

## 로봇 모델과 좌표

실제 학습 URDF: [`low-level-as2piper-head-orin-nocontact/resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf`](low-level-as2piper-head-orin-nocontact/resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf)

| 항목 | 현재 값 |
|---|---|
| 전체 질량 | 23.328064kg, URDF 명목값 |
| Orin box | 0.8725kg, 크기 0.110 × 0.110 × 0.07165m. 해당 URDF에서 복사한 모델값 |
| PiPER arm base / AS2 base_link | (0.22435, 0, 0.1319)m |
| Orin box origin / AS2 base_link | (-0.03, 0, 0.111875)m |
| 관절/정책 action | 20DOF / 18 action: 다리12 + 팔6, 그리퍼2는 별도 열림 제어 |
| TCP | piper_umi_nominal_tcp. 실제 EE 참조는 이 URDF 링크 |
| 초기 다리 | 각 다리 hip0 / thigh0.8 / calf-1.6 rad |
| 초기 팔 | [0, 1.1, -0.55, 0, 0, 0] rad |
| 초기 손가락 | +0.049 / -0.049m |
| reset 초기 base world z | 0.3264069024m + 지형 origin 처리. termination 높이와 다름 |
| base-height reward 목표 | 0.3244069024m |
| 종료 높이 | 지형 상대 높이 0.1m 미만. roll/pitch threshold 각각 0.8rad |

실측이 확정되지 않은 토크·gain·장착 강도를 이 모델값으로 인증하지 않는다. PiPER 실제 torque/gain/latency 반영과 실물 배포 검증은 후속 항목이다.

## 코드 진입점과 수정 위치

| 파일(학습 코드 폴더 기준) | 역할 |
|---|---|
| `run_force_ablation.sh` | 로컬 의존성·SDK·rsl_rl import 경로를 설정하는 권장 launcher |
| `force_ablation/run.py` | train/evaluate/validate/prepare 진입점, manifest, 초기 가중치, W&B |
| `force_ablation/configs.py` | A/B/C 분리. 현재 B를 명시적으로 선택 |
| `force_ablation/runtime.py` | 원래 환경/runner에 물리·reward 계측 추가 |
| `legged_gym/envs/manip_loco/as2_head_orin_config.py` | AS2 URDF·관측·gain·초기 자세·EE 범위·push·height override |
| `legged_gym/envs/manip_loco/b1z1_config.py` | 상속하는 원본 reward/PPO/DR/gait 기본 설정 |
| `legged_gym/envs/manip_loco/manip_loco.py` | 관측·history·reset·termination·push·IK·action 적용 |
| `legged_gym/envs/rewards/maniploco_rewards.py` | 실제 reward 계산식·walking mask |
| `validation/hardware.json` | 명목 질량·AS2 gain/armature·DR 스케일의 설계 계산값 |
| `rsl_rl_local/rsl_rl/modules/actor_critic.py` | Actor/Critic·privileged/history encoder |
| `rsl_rl_local/rsl_rl/algorithms/ppo.py` | PPO·Leg/Arm advantage·ROA regularization·DAgger |
| `rsl_rl_local/rsl_rl/runners/on_policy_runner.py` | rollout/update 전환·로그·checkpoint |
| `rsl_rl_local/rsl_rl/storage/rollout_storage.py` | rollout buffer·returns·advantage |
| `legged_gym/utils/helpers.py` | JIT export 등 공통 도구 |

`run_head.sh`는 공통 태스크의 40N 설정이므로 현재 B=200N 학습을 재현하는 진입점으로 사용하지 않는다.

## Observation · ROA · Action

| 경로 | 차원·동작 |
|---|---|
| Training observation | 현재67 + privileged18 + history670 = **755** |
| Deployment observation | 현재67 + history670 = **737** |
| History encoder / adaptation module | 과거67×10 → latent20. 별도의 두 직렬 모듈이 아님 |
| Privileged encoder | 학습 전용 물리 정보18 → latent20 |
| Actor backbone 입력 | 현재67 + latent20 = **87**, 90 아님 |
| Critic 입력 | 현재67 + privileged18 = **85** |
| 배포 | history latent 사용. privileged 물리값·발 접촉력 센서 입력 불필요 |
| Action | 18출력 중 다리12를 적용. 팔6은 원본 마스킹 경로 유지, 실제 팔은 DLS IK/position drive |

관측 구간: IMU0:5, q5:23, dq23:41, 이전 leg command41:53, velocity command53:56, EE goal56:59, 원본 orientation zero59:62, gait index62:63, clock63:67, privileged67:85, history85:755. 원본 orientation zero는 contact 제거용 padding이 아니다. Contact 4 + history40은 실제 제거했으며 zero padding하지 않는다.

ROA는 `dagger_update_freq=20`: iteration %20==0에서 history 경로 rollout 및 DAgger adaptation update, 나머지에서 privileged 경로 PPO update를 수행한다. Privileged 정규화 계수는 원본 schedule [0,0.1,3000,7000]. 실물에서는 history latent 추론을 쓰며 온라인 gradient update는 없다.

## 학습 조건과 보상

| 항목 | 현재 설정 |
|---|---|
| 환경/iteration/seed | 6,144 / 45,000 / 1 |
| rollout | 환경당24 step, physics dt0.005s, control decimation4 → policy dt0.02s |
| checkpoint 주기 | 200 iteration |
| feet height | target0.18m, scale+1.0 |
| feet contact force | net contact norm 초과분, threshold200N, scale-0.001 |
| contact shaping force/vel | 수정된 positive mismatch × scale-2.0. 정지 walking_mask로 0 |
| gait clock | 5차원 활성, 주파수2Hz |
| sensor-only foot_contacts_z | scale0. net contact/contact_filt 기반 다른 보상 유지 |
| push | base xy 속도 대입 방식, 이동±0.3963886m/s, 정지 multiplier2.5 → ±0.9909716m/s. N 단위 힘 아님 |
| PPO | learning rate2e-4, epochs5, minibatches4, clip0.2, gamma0.99, lambda0.95, 원본 알고리즘 유지 |

EE 명령은 원본 구면 샘플링 알고리즘 유지: r[0.40,0.48]m, pitch[0.10,0.50]rad, yaw[-0.25,0.25]rad, orientation delta 각±0.08rad. 초기 구면 [0.42789847,0.42168628,0]. Arm origin은 base 전체 회전을 반영하고 구면 방향은 yaw 기준으로 적용한다. 전체 상속 설정·DR 범위·reward scale은 설정 스냅샷을 참조한다.

## 서버 환경과 파일 전달

- 학습: Python3.8 / PyTorch1.13.1+cu117 / 기존 Isaac Gym PhysX. 현재 launcher의 로컬 `.deps`, `.wandb_deps`, `rsl_rl_local`, `third_party/isaacgym/python` 경로 우선순위를 유지한다.
- W&B: 프로젝트 로컬0.22.3 사용. Forge endpoint는 `https://api.forge.coreweave.com`. 사용자별 API 인증은 각자 설정하며 인증 파일은 공유하지 않는다.
- `setup.py`의 일반 dependency 선언만으로 현재 버전/바이너리 조합이 고정되는 것은 아니다. 스냅샷의 실제 버전을 기준으로 환경을 맞춘다.
- `.deps`의 CPython38 바이너리와 Isaac Gym의 기존 바이너리를 Python3.11 환경에 그대로 사용하지 않는다.
- 폴더 구조·URDF 상대 mesh 경로를 유지한다. 제3자 SDK는 호환 환경과 이용 조건에 따라 준비한다.
- launcher의 gymtorch 캐시는 형제 폴더 `low-level-as2piper-head-orin/.torch_extensions`를 사용한다. 새 서버에서는 쓰기 가능한 해당 경로를 준비한다.
- 코드·자산을 다른 위치로 옮긴 뒤 URDF 경로·PYTHONPATH·캐시 경로를 확인한다. 과거 run manifest의 절대 경로는 원래 서버의 기록이다.

## 실행 명령

현재 실행을 추가로 시작할 필요는 없다. 아래 train 명령은 **새 실험을 처음부터 시작할 때만** 사용한다. 기존 출력 디렉터리 이름은 재사용할 수 없다.

```bash
cd /data/team/users/suhwan/vbc-as2piper
umask 0002
training_run_name="as2piper_headpiper_orin_200N_0.18m_$(date -u +%Y%m%dT%H%M%SZ)"
log_path="$PWD/low-level-as2piper-head-orin-nocontact/experiments/contact_force_ablation/${training_run_name}.log"
nohup env WANDB_MODE=online WANDB_BASE_URL=https://api.forge.coreweave.com \
  WANDB_ENTITY=quadruped_capstone PYTHONUNBUFFERED=1 \
  bash low-level-as2piper-head-orin-nocontact/run_force_ablation.sh \
    --mode train --condition B --seed 1 --project as2piper-low \
    --name "$training_run_name" > "$log_path" 2>&1 < /dev/null &
echo "PID=$! LOG=$log_path"
```

현재 실행 로그:

```bash
tail -f /data/team/users/suhwan/vbc-as2piper/low-level-as2piper-head-orin-nocontact/experiments/contact_force_ablation/B_push25_train_20261111T135342Z.log
```

기존 GPU 메모리 부족이 확인되어 추가 평가/영상 실행은 중단한 상태이다. 평가 명령은 실행 가능한 자원이 확보된 뒤 사용한다.

```bash
cd /data/team/users/suhwan/vbc-as2piper
bash low-level-as2piper-head-orin-nocontact/run_force_ablation.sh \
  --mode evaluate --condition B --seed 1 --terrain flat --seconds 40 \
  --checkpoint /data/team/users/suhwan/vbc-as2piper/low-level-as2piper-head-orin-nocontact/experiments/contact_force_ablation/runs/as2piper_headpiper_orin_200N_0.18m/model_10000.pt \
  --name "B_10000_flat_$(date -u +%Y%m%dT%H%M%SZ)"
```

Rough는 `--terrain rough`로 별도 실행한다. 평가 코드는 optimizer update 없이 history latent로 추론하며, 학습 source hash와 일치를 검사한다.

## 체크포인트·재개·배포의 현재 범위

- 실행 폴더: `low-level-as2piper-head-orin-nocontact/experiments/contact_force_ablation/runs/as2piper_headpiper_orin_200N_0.18m/`.
- `manifest.json`: 실제 설정·source hash·W&B ID. `model_*.pt`: 모델 전체 state(History/Privileged encoder 포함), main optimizer state, iteration, infos.
- 공통 최초 가중치: `experiments/contact_force_ablation/seed_1/initial_policy.pt`. 초기화 hash는 각 run의 `initial_model_sha256.txt`에 기록.
- 현재 `force_ablation/run.py --mode train`은 공통 seed1 초기 가중치에서 새로 시작한다. 기존 checkpoint resume용 CLI 옵션은 제공하지 않는다.
- 원본 checkpoint 포맷은 history encoder의 별도 optimizer state, ROA counter, RNG, 환경/observation history 상태를 저장하지 않는다. 따라서 모델 weight는 불러올 수 있지만 중단 직전 학습 상태를 완전히 동일하게 복원하는 포맷은 아니다.
- 755/737 입력과 JIT parity는 검증했으나, 이것은 실물 AS2/PiPER SDK·지연·모터 통신까지 완성된 배포 패키지를 의미하지 않는다.
- `restart_B_push25.sh` 및 `force_ablation/restart_push25.py`는 과거 명시적 요청으로 만든 기존 B 실행 중단·삭제·새 시작 도구이다. 일반적인 실행/재개 도구로 사용하지 않는다.

## 검증 상태와 미해결 항목

- 10,000회: 평지/rough 각각32환경·명령당8환경·40초, 초기2초 제외. 평지 전진0.4 명령에 실제0.673m/s, rough0.704m/s로 과속. 순수 회전0.6 명령은 평균0.607/0.600rad/s.
- EE 평균오차 약1.5~1.7cm. 실패 reset은 평지 복합1건, rough 정지1건. 각 지면 timeout96건은 정상 episode 종료로 구분.
- 정지 yaw RMS는 평지0.538/rough0.515rad/s. 정지 흔들림과 과속은 미해결이며 보상·PPO를 임의로 변경하지 않았다.
- Contact Shaping은 정지0·이동음수 정상. Rough net-contact force의 과소 집계 가능성 때문에 실제 하중·안전성을 판단하지 않는다.
- 10,000회 loss·모델 유한값 확인. 명확한 발산·지속 정지 정책은 관찰되지 않아 학습 유지.
- 영상 작업은 추가 GPU 메모리 부족 후 사용자 요청으로 중단. `reports/B_iter10000_videos_20261112/`의 미완성 파일을 완성된 검증 영상으로 배포하지 않는다.
- 후속: 실제 PiPER torque/gain/latency, ground-picking workspace/whole-body 확대, 실물 상태 입력·안전 제어 검증.
