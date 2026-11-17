# EE pose command 설계 검증 — 100 PPO iteration

**판정: 명목 조건의 좌표계·팔 추종은 동작하지만, 현재 버전 전체를 검증 완료된 장시간 학습 설계로 승인할 수 없다.** 구면 범위 자체가 전부 잘못된 것은 아니다. 다만 시작점 방향 충돌의 재추출 처리와 초기 EE 목표 구성에 구체적인 문제가 있으며, 실제 전신 학습에서는 다양한 EE 목표를 거의 경험하지 못했다. 이 상태를 그대로45,000회 돌리는 것은 권하지 않는다.

## 실행 조건과 측정 방법

- `low-level-as2piper-forward`, `as2piper_plate_forward`, W&B **disabled**.
- 기존과 같은 **6144환경×24control step×100 PPO iteration = 14,745,600 transitions**. 100iteration은 전체 학습 수렴 검사가 아니다.
- wall time **1652.92초, 약27.55분**. 기존 AS2 학습과 병행했으며 기존 프로세스를 정지하지 않았다.
- 검사 중 runtime URDF/config/reward/PPO를 변경하지 않았다. 별도 관측 wrapper만 사용하고, 진단 프로세스에서만 DOF force sensor를 활성화했다.
- 매 step 시작 때 실제 적용 목표를 복사하고, 물리 계산 후 reset 전에 실제 EE와 비교했다. reset 직후 목표가 원위치로 바뀌면서 오차가 가려지는 문제를 피했다.
- 위치 오차는 **world L2 거리(m)**, 방향 오차는 quaternion 최단 회전각(°). 원본 reward의 L1 오차와 구별한다.
- 전체6144환경에서 reset·sampling 횟수를 집계하고, 균등 간격512환경의 모든2400step을 저장해 오차 분포를 계산했다. 아래 quantile은1,228,800개 저장 상태의 통계이며 독립 episode/목표 성공률이 아니다.
- Runner가 시작할 때 episode counter를 무작위로 초기화하므로, 저장된 done sequence에서 실제 경과 시간을 다시 계산했다. 원본 counter의 첫 episode 길이를 실제 생존 시간으로 오해하지 않았다.
- upright 진단 조건은 |roll|,|pitch|<.35rad 및 terrain origin 대비 body 높이>.22m. late는 reset 후 실제 .5초 이후다.

## 결과

| 조건 | 위치 평균 | 위치95백분위 | 방향 평균 | 방향95백분위 |
|---|---:|---:|---:|---:|
| 몸통 고정, nominal 질량/CoM, 무작위 구면 목표 | **1.57cm** | **2.82cm** | **4.27°** | **8.22°** |
| 위 조건의 목표 hold 구간 | 1.42cm | 2.01cm | 3.99° | 5.73° |
| 몸통 자유, 평지·nominal, zero action·초기 C 목표 | **1.13cm** | **1.16cm** | 미계측 | 미계측 |
| 실제 PPO 검사 전체 저장 상태 | **8.36cm** | **20.12cm** | **14.89°** | **33.41°** |
| 실제 PPO, upright·.5초 이후 | 5.78cm | 10.58cm | 10.34° | 18.29° |
| 실제 PPO, upright·무작위 목표·.5초 이후 | **4.28cm** | **9.37cm** | **10.97°** | **25.94°** |

몸통 고정 실험은64환경·30초이며 통계는4초 이후를 사용했다. 같은 URDF, 구면 sampler, DLS IK, arm400/40·gripper1800/35 position drive를 사용했다. 몸통을 고정하고 DR/PPO를 제거해 팔 목표와 제어만 분리했다. 해당 실험의 arm 관련 net contact>1N은 관측되지 않았다. 초기 mixed 구간은 자동 goal timer가 먼저 끝날 수 있으므로 순수 home-only 결과로 사용하지 않았다.

몸통 자유 nominal 대조군은16환경·10초, 평지·명목 질량/CoM/motor, 정확한 neutral joint reset, zero policy action으로 초기 목표를 유지했다. **reset0회, upright100%, roll 절댓값95백분위0.36°**, 평균 body 높이.3049m였다. 즉 현재 기구와 기본 자세가 명목 조건에서도 반드시 넘어지는 구조라고 볼 근거는 없다. 이 대조군은 실제 DR·terrain·학습 정책 조건과 다르므로 특정 DR나 reward를 실패 원인으로 확정할 수는 없다.

실제 전신 검사에서는 오차가 크게 늘었다. 단, upright·무작위 목표·late 표본은 **434개 상태**뿐이다. 그중 위치5cm/방향10°를 동시에 만족한 비율은51.6%였다. 이는 희소하고 시간적으로 상관된 상태 통계이며 파지 성공률이 아니다. 새 목표 전체의 추종 성능을 충분히 평가했다고 할 수 없다.

## 실제 학습이 EE command를 얼마나 경험했는가

- 전체543,540회 episode 종료. 배타적으로 분류하면 roll408,970회, pitch133,749회, 낮은 높이6회, timeout815회다.
- 마지막 로그의 평균 episode 길이는 **.6018초**였다. 종료의 거의 전부가 roll/pitch 한계 초과다. **EE 추종 실패 직접 reset 조건은 비활성화**되어 있다.
- 기록 상태 중 새 무작위 EE 목표를 적용 중인 비율은 **.2987%**였다. 나머지는 대부분 초기 C 목표다. 초기 목표도 실제 위치·방향 명령이며 명령이 없다는 뜻은 아니다.
- 전체 새 목표 요청1453건 중622건 수락,831건 이전 목표 유지: **fallback57.2%**.
- 마지막20iteration에서는 요청56건 중3건 수락,53건 fallback이었다. 표본 수가 적지만 검사 후반에도 다양한 목표가 거의 공급되지 않았음을 보여준다.

따라서 PPO100회 결과로 “무작위 EE command가 전신 동작에서도 잘 추종된다”고 결론 낼 수 없다. 기본 보행/자세 안정성을 먼저 확보하거나, 초기 목표 단계에서도 작은 연속 이동을 경험하도록 구성한 뒤 같은 지표를 다시 봐야 한다. 100회만으로45,000회 학습의 최종 수렴 가능성을 부정하는 것은 아니지만, 아래 필터 문제는 학습량과 별개다.

## 확인된 설계 문제1: 위치 재추출로 해결되지 않는 시작점 방향 거절

원본 B1은 orientation delta를 한 번 뽑고 위치만 최대10회 재추출한다. 원본 충돌 검사는 EE 위치 경로 중심이라 이전 시작점이 안전하면 대체로 이 방식으로 진행할 수 있었다.

AS2 추가 검사는 t=0,.5,1의 **위치와 방향**을 IK/OBB로 검사한다. 이때

```
sphere(t) = (1-t)*start + t*goal
sphere(0) = start
orientation(0) = f(start.pitch, start.yaw, 새 orientation_delta)
```

즉 새 방향이 시작점에서 거절되면 목적지 위치를 다시 뽑아도 시작점 조건은 바뀌지 않는다. 현재 코드는 같은 새 방향으로10회 위치만 바꾸므로, 이 경우 재시도로 해결할 수 없다. 이전 목표 유지가 안전 측면의 fallback이기는 하지만 목표 다양성과 계산 비용에 손해가 생긴다.

실제 몸통 고정 hold 상태를 재생한921개 새 방향 제안에서 **317개(34.4%)가 시작점에서 거절**됐다. 같은 시작점에 대해 IK를12→64회로 늘리면 **88개가 통과**했다. 따라서 일부는 짧은 수치 풀이 때문에 거절된 경우이며, 나머지 거절도 기구학적 불가능의 증명은 아니다. 보수적인 box proxy 영향도 있다. 이 검사는 전체 runtime fallback의 모든 원인을 분류한 것이 아니라, 위치만 재추출해서 해결할 수 없는 메커니즘이 실제 입력에서 발생함을 입증한 것이다.

근거 파일: `../ee_fixed_base_20261029/guard_start_analysis.json`. 코드 위치는 `legged_gym/envs/manip_loco/manip_loco.py:1253`의 방향1회/위치반복, `as2piper_forward.py:84`의 시작점 포함 pose 검사다.

**필요한 수정:** B1의 독립 구면·방향 proposal과 구면 보간은 유지하되, 시작점 방향 조건이 실패했으면 방향도 재추출해야 한다. IK 실패에는 수렴 정도·다른 seed/추가 반복을 구분해 과도한 rejection을 줄여야 한다. 충돌 조건을 유지하면서 실패 조건에 맞게 재추출을 바꿔야 한다.

## 확인된 설계 문제2: 초기 start=end로 인한 목표 경험 부족

현재 `init_pos_start`와 `init_pos_end`가 모두 `[.427898468686,.421686284191,0]`이다. 따라서 초기 trajectory1–3초와 hold.5–2초 동안 팔 목표가 그대로다. 첫 새 무작위 목표까지 최소1.5초가 필요한데, 실제 episode는 약.6초다. 이로 인해 초기 단계가 사실상 학습 EE task의 대부분이 되었다.

초기 목표 점프를 막으려면 **start를 실제 초기 FK와 맞추면 된다. end까지 동일할 필요는 없다.** 원본 구면 보간을 유지하면서 end를 가까운 다른 유효 pose로 두면 연속적인 작은 초기 arm motion을 만들 수 있다.

검토용 end 후보 `[.46,.25,0]`를 만들었다. 해당 경로41점의 IK/몸통 proxy 검사에서는 거절0건이었다. 최종 arm-local TCP는 `[.445700,0,.113806]m`다. **이 후보는 현재100회 실행에 적용하지 않았고 동적 검증도 완료하지 않았다.** `initial_motion_proposal.json`은 다음 수정의 구체적 검토안이다.

## 판단과 다음 우선순위

1. 명목 조건의 좌표계·기본 자세·구면 pose 생성은 추종 가능한 출발점이다.
2. 방향 거절을 위치만 바꿔 반복하는 문제를 먼저 수정한다.
3. 초기 FK start는 유지하되 검증된 다른 end를 사용해 연속적인 초기 목표 변화를 만든다.
4. 그 후 전신 자세 안정성과 충분한 무작위 목표 경험을 확보하고, upright random hold의 위치·방향 오차 및 실제 접촉을 다시 평가한다.

검사 중 reward·DR·PPO를 튜닝하지 않았다. 원본 contact shaping 부호, 큰 추가 질량/CoM randomization 등 기존 검토 항목도 그대로다. 이 검사만으로 어느 하나를 전신 불안정의 단독 원인이라 확정하지 않는다. 실제 모터 사양·카메라 질량·구조 체결 강도까지 승인한 결과도 아니다.

DOF force tensor도 기록했지만 이는 API가 반환하는 **DOF에 작용하는 total force**이며 모터 구동 torque만 분리한 값이 아니다. 이를 전류/gear torque 한계 검증으로 사용하지 않았다. 로컬 Isaac Gym 문서 `third_party/isaacgym/docs/_sources/programming/forcesensors.rst.txt`의 Joint Force Sensors 설명에 따른다.

## 산출물과 재현

- `summary.json`: 최종 상태 오차·reset·sampling 통계.
- `training_diagnostics.png`: iteration별 오차/종료 원인/목표 요청.
- `ee_initial_replay.mp4`, `ee_late_replay.mp4`: 실제 PhysX 기록 상태의 초기/후기4초. 빨강은 적용 목표, cyan은 실제 EE. MuJoCo는 재생·렌더에만 사용했고 다시 물리 계산하지 않았다. 원래 rough terrain 형상은 표시하지 않아 평면은 높이 참고면이다.
- `control_comparison.jpg`: nominal free-base, fixed-base arm, 실제 PPO 상태 비교.
- `ee_fixed_base_tracking.mp4`: 몸통 고정 상태에서 실제 구면 목표를 추종하는4초 재생. 실제 PhysX 위치·관절 상태이며 MuJoCo 재시뮬레이션이 아니다.
- `tracking_samples.npz`, `trajectories.npz`, `iterations.jsonl`, `metadata.json`: 원시 기록과 계측 정의.
- 체크포인트: `logs/as2piper-low/ee_observe_100_20261029/model_100.pt`. 장시간 학습 승인된 정책이 아니다.
- 별도 대조군: `../ee_fixed_base_20261029`, `../ee_nominal_stand_20261029`.

```bash
cd /data/team/users/suhwan/vbc-as2piper/low-level-as2piper-forward
umask 0002
export PYTHONPATH="$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python"
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions" MAX_JOBS=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled WANDB_DISABLE_GIT=true
export EE_AUDIT_DIR="$PWD/audit_forward/ee_100_repeat"
python audit_forward/observe_ee_training.py --task as2piper_plate_forward \
  --proj_name as2piper-low --exptid ee_observe_100_repeat --headless --num_envs 6144 --max_iterations 100
```

재실행에는 새 출력 경로와 exptid를 사용해야 한다. 기존100회 기록을 덮어쓰거나 이어 붙이지 않는다. 기존 `low-level-as2piper`와 B1 baseline은 수정/중지하지 않았으며,100회 진단 프로세스는 정상 종료했다.
