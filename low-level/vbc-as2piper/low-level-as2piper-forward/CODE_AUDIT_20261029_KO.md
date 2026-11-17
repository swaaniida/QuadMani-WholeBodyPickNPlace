# 새 AS2+PiPER 모델 학습 준비 상태와 VBC 비교

## 판정

현재 `low-level-as2piper-forward`는 Isaac Gym에서 실행되고 PPO 업데이트까지 가능한 복제본이다. 그러나 **현재 상태 그대로 45000 iteration 장시간 학습을 시작하는 것은 권하지 않는다.** 100 iteration 검사에서 드러난 EE sampling/초기 목표 문제를 아직 수정하지 않았기 때문이다. 이번 감사에서는 학습 코드·config·URDF를 수정하거나 학습을 시작하지 않았다.

실제 `/data/team/users/suhwan/vbc-legacy`의 소스와 직접 비교했다. 과거 W&B B1 run의 실행 중 override까지 동일하다는 뜻은 아니다. `legged_gym` 및 `rsl_rl_local/rsl_rl`의 Python 파일 중 기존 대응 파일과 동일한 것은34개, 변경3개, 신규5개다. 원본 AS2 소스55개도 이전 SHA와 동일하다.

## 실행 구조

```text
run_as2.sh
  ├─ 프로젝트 전용 PYTHONPATH / rsl_rl_local / gymtorch cache
  ├─ prepare_asset.py → 자체 plate-forward training URDF
  └─ legged_gym/scripts/train_as2.py
       ├─ W&B init (as2piper-low, 새 name/ID)
       ├─ task_registry → AS2PiPERForward + AS2PiPERPlateCfg/PPO
       ├─ AS2PiPERForward : ManipLoco : LeggedRobot : BaseTask
       │    ├─ reset / goal / observation / collision의 AS2 전용 override
       │    └─ Isaac Gym PhysX: 다리 PD torque + 팔 IK/position drive
       └─ OnPolicyRunner → ActorCritic + PPO + rollout storage
            ├─ 6144환경 × 24control step
            ├─ PPO / 매20 iteration history encoder 학습
            └─ W&B 기록 / 매200 iteration checkpoint
```

`arm_collision.py`는 추가 후보 검사용 배치 FK·IK/OBB이고, `urdf_chain.py`는 URDF FK 유틸이다. MuJoCo는 asset 기하 검증/시각화와 충돌 proxy 생성에 쓰이며 실제 PPO 학습 물리는 Isaac Gym PhysX다. 과거 joint-space command bank는 활성 경로에서 사용하지 않는다.

## 원본 VBC와 동일한 부분

다음 파일은 실제 B1 baseline과 바이트 단위로 동일하다.

- `rsl_rl_local/rsl_rl/algorithms/ppo.py`
- `rsl_rl_local/rsl_rl/runners/on_policy_runner.py`
- `rsl_rl_local/rsl_rl/modules/actor_critic.py`
- `legged_gym/envs/rewards/maniploco_rewards.py`
- `legged_gym/envs/manip_loco/b1z1_config.py`
- `legged_gym/utils/terrain.py`

상속한 PPO 설정도 experiment_name 외에는 같다. 환경6144개, rollout24step, 총45000 iteration, Adam LR0.0002, epoch5, minibatch4, gamma.99, lambda.95, clip.2, seed1을 유지한다. 기존 DR·terrain·보행 command·termination·noise·normalization 설정도 비교 검사에서 동일했다.

네트워크의 history/privileged encoder, leg/arm head 및 reward 혼합·regularization schedule도 원본이다. **PPO 소스가 같다는 것과 환경의 동작까지 완전히 같다는 것은 구별해야 한다.** 아래에는 명시적인 동작 변경이 있다.

## 변경한 부분과 이유

| 영역 | B1 원본 → 새 AS2 모델 | 이유와 영향 |
|---|---|---|
| 기구 | B1+Z1+단일 gripper DOF → AS2+PiPER+UMI, 20DOF | 다리12+팔6+finger2; V7 형상 유지 |
| 액션 | 정책18차원 유지, finger는 별도 열림 제어 | 기존 WBC 인터페이스 유지; finger 목표±.049m |
| 장착 | AS2 기존 arm base x=-.020 → +.005m | 판140×120×6mm는 연장하지 않고 PiPER만25mm 전진. bolt/hardware 승인값은 아님 |
| 링크 계약 | AS2 joint/foot 이름과 순서 검사, UMI TCP·foot fixed joint 보존 | DOF slicing, Jacobian, 접촉 센서가 다른 링크를 참조하지 않도록 함 |
| arm/hand 제어 | 팔400/40 position drive 유지, finger1800/35 분리, arm target joint-limit clamp | 2개 prismatic finger를 별도 열림 유지하고 잘못된 joint target 제한 |
| mass/CoM DR 적용 | 하드코딩 body index1 → 실제 `base_link` index | B1의 rigid-body 순서를 AS2에 그대로 적용하지 않도록 함. DR 범위 자체는 유지 |
| 발 센서 | foot frame z=-.05 → 0 | AS2 foot frame에 맞춤. 센서 wrench moment 기준점도 달라짐 |
| 초기 다리·팔 | 다리[hip0,thigh.8,calf-1.6], arm C=[0,1.1,-.55,0,0,0] | AS2 기구와 초기 TCP FK 기준. 팔 reset noise는 제거하고 다리 noise는 유지 |
| 초기 높이 | B1 고정 .5m → .3264069m 명목, reset별 발 FK로 보정 | 최저 발 구 바닥이 terrain origin보다2mm 높도록 계산. rough terrain 국소 접촉을 보장하는 방식은 아님 |
| base height reward 기준 | .55 → .3244069m | AS2 다리 길이·자세에 따른 기하학 기준. reward 수식·scale는 유지 |
| EE 중심 | yaw+고정 world z=.7m 중심 → 실제 arm mount origin | `origin = root_pos + R_body*[.005,0,.0820284782]`; 몸통 pitch/roll에 따른 장착 위치 변화 반영 |
| EE 축 | B1처럼 yaw/중력 정렬 유지 | `goal = origin + R_yaw*sphere2cart(r,p,y)`; full body rotation으로 구면 축을 기울이지 않음 |
| EE 관측 | 실제 arm origin을 빼고 body rotation 역변환 | command/control/observation 좌표계 일관화. 방향3칸은 원본처럼0 |
| EE 방향 | 고정 roll π/2 → 0, induced pitch .38 →2.4552161 | Z1과 PiPER/UMI TCP 축 차이를 반영; 초기 FK 방향에 맞춤 |
| 구면 범위 | r .4–.95 → .38–.52m, p 약−1.257–1.047 →−.25–.60rad, y±1.2 →±.40rad | 더 작은 PiPER와 중앙 장착의 작업 범위 반영. 방향 delta 각±.5 →±.15rad |
| 추가 충돌 검사 | 기존 EE10점 금지 box/floor + 새3pose IK·arm/body OBB | AS2 몸통/머리와 팔·그리퍼 간섭을 후보 생성에서 제외하려는 목적 |
| 실패 fallback | 마지막 거절 후보 대신 이전 목표/방향 유지 | 무효 후보를 바로 제어에 넘기지 않기 위함. 움직이는 몸통에서 이전 목표의 안전성이 영구 보장되는 것은 아님 |
| PPO action 보호 | `step(actions.clone())` | 원본 환경의 arm action in-place masking이 PPO에 저장한 sampled action을 덮어쓰지 않도록 수정 |
| 실행·기록 | 자체 task/asset/logger 경로, 매 실행 새 W&B ID | B1 및 기존 AS2 run과 파일·실험 분리 |

### 관측799차원의 의미

현재 디스크 B1 기본 설정은 gait 관측false이며 proprio66, private18, history10이므로744차원이다. 새 모델은 gait phase/clock5개를 켜서 proprio71, 총 `71×11+18=799`다. **추가55칸은 gait5개×현재/과거11개이며 단순 zero padding이 아니다.** 방향3개가 현재/과거에 모두0인 것은 원본 B1 관측 계약을 유지한 것으로 별개다. 과거 외부 W&B B1 run이 어떤 override를 사용했는지는 이 파일 비교만으로 확정하지 않는다.

gait flag는 관측만 바꾸지 않는다. 원본 contact-shaped force/velocity 함수의 조기0 반환도 해제한다. 따라서 **reward 함수와 scale는 같아도 이 두 항의 활성 여부는 현재 디스크 B1 기본값/기존 AS2 run과 다르다.** 원본 수식은 음수 raw 값과 음수 scale의 곱이며 이 부호도 수정하지 않았다.

### EE 중심 변경의 의미

B1 원본은 관측용 arm offset을 쓰지만, 목표 구면 중심의 world z는 .7m로 고정한다. 새 모델은 실제 arm origin을 사용해 root 높이/roll/pitch에 따라 목표 중심도 이동한다. 이는 단순히 B1 offset 숫자만 바꾼 것보다 큰 과제 정의 변경이다. 바닥의 물체에 대해 world-fixed 목표를 유지하는 작업은 현재 sampler만으로 구현됐다고 볼 수 없다.

## 지금 장시간 학습을 시작하면 안 되는 구체적인 이유

### 1. 시작점 방향 실패를 위치 재추출로 해결하려 한다

원본은 방향delta1회 → 위치최대10회다. 새 검사는 t=0,.5,1의 pose를 검사한다. t=0에서 새 방향이 IK/충돌 검사에 실패하면 목적지 위치만 바꿔도 t=0은 변하지 않는다. 이 불일치가 아직 남아 있다. 실제 전체 목표 요청1453건 중831건(57.2%)이 fallback이었다.

몸통 고정 hold 상태921개에서 시작점 방향 제안317개가 거절됐고, IK 반복을12→64로 늘리면 그중88개가 통과했다. 유한 반복 실패를 물리적 불가능으로 간주해서는 안 된다. 해결은 독립 구면·방향 proposal과 구면 lerp는 유지하되 실패 원인에 맞게 방향도 재추출하고, IK 수렴 실패를 보완하는 것이다. arm self/leg/object/cable 또는 연속경로 clearance는 지금3점 box 검사로 보장되지 않는다.

### 2. 초기 EE start=end

둘 다 `[.427898468686,.421686284191,0]`이다. 초기 trajectory1–3초+hold.5–2초 동안 목표가 정지한다. 첫 random goal까지 최소1.5초인데 100회 테스트의 마지막 평균 episode는.6018초였다. 기록 상태 중 random goal 적용은.2987%뿐이었다.

초기 start를 실제 FK에 맞추는 것은 타당하지만 end까지 같을 필요는 없다. 검토 후보 end `[.46,.25,0]`는41점 IK/body-proxy 검사에서 거절0이었지만 **미적용·동적 검증 미완료**다. 초기 목표 이동은 reward/PPO를 바꾸지 않고 config와 기존 구면 보간으로 구성할 수 있다.

### 3. 새로운 전신 동작의 성능 근거 부족

| 기존 검사 | 위치 평균/p95 | 방향 평균/p95 |
|---|---|---|
| 명목 몸통 고정·random EE | 1.57/2.82cm | 4.27/8.22° |
| 평지 명목·free base·초기 목표 | 1.13/1.16cm | 미계측 |
| 실제 PPO·upright·random goal·reset후.5초 이후 | 4.28/9.37cm | 10.97/25.94° |

마지막 행은434개 상관된 상태뿐이어서 workspace 전체 성능/파지 성공률로 일반화할 수 없다. PPO100회가 수렴 학습은 아니지만, sampler 문제는 수렴 시간과 별개로 먼저 고칠 수 있다.

## 그대로 유지했지만 검증이 필요한 부분

- 큰 B1 DR: 추가mass0–15kg, CoM 각축±.15m, friction.3–3, motor.7–1.3, push8초/.5m/s. AS2 명목 대조군과 다른 조건이다. 이것이 넘어짐의 단독 원인이라고 입증하지는 않았다.
- feet height target.3m, contact force 기준40N 등 B1의 길이/하중 관련 상수도 요청에 따라 유지했다. AS2 적합성이 모두 입증됐다는 뜻은 아니다.
- 출력18차원 중 arm6개는 원본처럼 환경에서 mask되고, 실제 arm은 IK/position drive로 제어한다. 18개 모터 전부를 정책이 직접 torque 제어하는 구조가 아니다.
- EE orientation 관측은0이며 orientation reward scale도0이다. 위치 기반 전신 보조와 명시적6D pose 정책을 동일시하면 안 된다.
- `env.torques`의 arm 부분은0이며 position-drive가 실제로 내는 torque가 자동으로 reward에 포함되지 않는다. 원본과 동일한 제한이고, 이번 감사에서 reward/torque supervision을 바꾸지 않았다.
- 손목 camera image/물체 인식/파지·dragging policy는 현재 low-level policy 입력·훈련 과제에 포함되지 않는다. 렌더링 데모와 학습 정책은 구별해야 한다.
- bolt pattern/하중 경로, 실제 motor limit, 카메라 질량은 시뮬레이션 로딩 성공으로 검증되지 않는다.

## 다음 실행 전 필요한 조치

1. 새 복제본에서 위 sampler 실패 처리와 초기 end를 수정한다. 기존 B1/기존 AS2는 보존하고 reward/PPO는 유지한다.
2. FK/PhysX 정합, 초기 경로·후보 rejection과 fallback, 제어 연속성을 짧은 물리 테스트로 확인한다. 무작위 목표를 실제로 충분히 경험하도록 명목 대조군과 전신 학습 결과를 구별한다.
3. 장시간 실행 전에 checkpoint에 history optimizer, schedule/environment counters, RNG와 실제 config/asset SHA를 추가 저장한다. 이는 과거 checkpoint에 없는 상태를 소급 복구하지는 않는다.
4. 터미널과 분리된 실행 및 stdout/stderr 파일 기록을 사용한다. 현재 `run_as2.sh`만 전경에서 실행하면 이전 VS Code 종료 문제를 예방하지 못한다.
5. 그 뒤 W&B `as2piper-low`에서 새 run으로 시작한다. 원본744와 새799 관측 계약/기구가 달라7000 checkpoint를 새 모델에 그대로 strict load하는 재개는 아니다.

## 이번 감사의 검증과 산출물

- `check_original_settings.py` 재실행 PASS: 허용한57개 환경 설정 차이, PPO experiment_name 외 동일.
- 실제 baseline Python 파일 비교:34동일/3변경/5추가. collision JSON의 asset SHA 일치.
- 활성 config/subclass/base/reward/URDF5개 SHA가100 iteration 테스트 당시와 동일함을 재확인. 알려진 문제가 아직 남아 있다는 뜻이기도 하다.
- `audit_forward/code_audit_20261029.json`: 파일별 SHA와 baseline 비교, 원본55개 보존 결과.
- `audit_forward/ee_100_20261029/REPORT_KO.md`: 실제100회·대조군의 측정 방법과 수치.
- 기존 baseline 및 AS2 runtime에는 변경 없음. 새 감사 문서/JSON과 자체 settings 비교 산출물만 작성했다. 저장소가 대부분untracked이므로 tracked `git diff --stat`만으로 변경량을 해석하면 안 된다.
