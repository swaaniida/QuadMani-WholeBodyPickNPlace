# 보행 수정 및 실제 정책 평가

아래는 과거 gait/transfer 진단 기록이다. 현재 장시간 실험은 `VBC_PORT_AUDIT_KO.md`의 원본 VBC 의미 복구 task를 체크포인트 없이 학습한다. 이 문서의 개별 reward 변경이 보행 성공의 필수 원인으로 분리 입증된 것은 아니다.

## 확인한 문제

이전20 iteration 모델은 초기 상태 검사였으며, 수렴된 보행 정책이라고 판단할 수 없었다. 그 모델은 서기를 유지했지만 전진.15m/s 명령에도 실제 속도가 거의0이었다.

추가로 다음 설정은 보행을 불리하게 만들 수 있었다.

1. `walking_dof`: 걷기 명령 중에도 기본 관절각을 유지할수록 보상을 준다.
2. 발 공중시간: 원본은 착지 시 `air_time-.5`를 사용한다. 현재2.523Hz/duty.5의 예상 swing은.198초이므로 정상 주기에도 발당 약−.302가 된다.
3. Feet height: 앞발 두 개의 절대 높이 norm은 네 발의 stance/swing 위상에 맞는 발 들림과 다른 목표다.
4. 작은 명령에 대한 속도비 reward와 과도하게 줄인 초기 탐색은 정지 행동을 벗어나는 학습을 어렵게 만들 수 있다.

위 설정의 모순은 코드로 확인했지만 어느 하나가 유일한 원인이라고 입증한 것은 아니다. 학습량·과제 분리·reward·탐색을 함께 바꾼 결과이며 원인별 ablation 실험은 아니다.

## 수정 범위

기존 B1, 기존 AS2, forward 실험은 보존했다. HW 복제본에 별도 `as2piper_plate_gait` task를 추가했다. 기존 하드웨어 gain·armature·URDF·20DOF/18action·799관측·PPO 업데이트 구조는 유지했다.

- 먼저 팔을 C 자세로 유지하고 보행을 학습한다. 팔·그리퍼 질량과 관성은 실제 모델에 그대로 있다.
- 기본 관절각 고정을 유도하는 `walking_dof`는 보행 task에서0.
- 속도비 reward 대신 기존 Gaussian XY 속도 추종 함수를 사용, 가중치3/tolerance².04.
- Yaw 추종 가중치1, 별도 tolerance².0625.
- 공중시간 기준은 `.5/gait_hz`로 변경하고 네 발을 사용, 가중치.5.
- 발 높이는 네 발의 현재 gait phase에 따른 최대40mm swing clearance를 추종한다. 높이는 발 반경/ground origin을 반영한다.
- 초기 다리 action std를.3→.6으로 늘려 실제 발 디딤을 탐색한다. 다리 gain·action scale·effort cap은 유지한다.
- 명령은 전진.15–.4m/s, yaw±.5rad/s이며 stand/회전/전진/전진+회전을 섞는다.
- 이 단계는 added mass/COM/gripper mass/push를 끄고, 기존 축소 friction/leg-strength DR은 유지한다.

가중치/수식 변경은 새 보행 task에 한정한다. 기존 VBC reward를 모두 그대로 유지했다고 설명하지 않는다. 발 높이 목표는 reward 기준이며 scripted joint trajectory로 로봇을 움직이는 코드가 아니다. 실제 다리 action은 학습된 PPO 정책이 출력한다.

## 학습

`gait_pilot500_20261029`, W&B disabled,2048환경×24step×500iteration=24,576,000 transitions. 처음부터 학습했고 기존 B1 가중치를 가져오지 않았다. 기록된 training loop 시간은791.18초이며 동시 평가의 영향이 포함된다.

최종 파일:

`logs/as2piper-low/gait_pilot500_20261029/model_500.pt`

41개 model tensor가 finite이고 algorithm counter=next_iteration=500이다. History/PPO Adam과 RNG/counter도 함께 저장돼 있다.

## 최종 기본 보행 평가

명목 평지, deterministic **history encoder**, 팔 C 자세 유지,32환경×10초. 명령별8환경이며 앞2초를 제외한 평균이다.

| 명령 | 실제 vx 평균 | 실제 yaw 평균 | 넘어짐 |
|---|---:|---:|---:|
| 정지 |−.001m/s|+.007rad/s|0|
| 전진 .25m/s |**.256m/s**|.000rad/s|0|
| 제자리 좌회전 +.4rad/s |.023m/s|**+.387rad/s**|0|
| 전진 .25 + 좌회전 .4 |**.252m/s**|**+.395rad/s**|0|
| 제자리 우회전 −.4rad/s |.014m/s|**−.372rad/s**|0|
| 전진 .25 + 우회전 −.4 |**.255m/s**|**−.396rad/s**|0|

전진의 instantaneous 속도 MAE는.0129m/s이고, 네 발의 clearance p95는 약35/29/24/39mm다. 전진 접촉 위상 일치율은82.2%, 최대 적용 leg torque26.88Nm였다. 따라서 단순 standing/관절 포즈 재생이 아니라 발을 들고 명령 속도로 이동하는 정책을 확인했다.

구형 발의 바닥 접촉점 속도는 `v_COM + omega × [0,0,-.028]`로 근사했다. 발 중심 이동과 실제 바닥 미끄러짐은 같지 않다. Fz>5N 전체 구간의 평균은 전진.133m/s, 충분히 하중을 지지하는 중간 stance(Fz>20N, desired contact>.9)는.005m/s였다. 접촉 전환부의 속도/접촉 지연이 남아 있으므로 완벽한 무미끄럼 gait라고 주장하지 않는다.

평가는10초이고 rough terrain, backwards/sideways walking, 실제 하드웨어 인증이 아니다. 제자리 회전에서도 작은 전진 drift는 남아 있다.

## 팔 변화와 WBC 연결

팔 고정 보행 성공을 무작위 팔 목표에서도 검증된 것으로 간주하지 않았다.500 모델에 random EE 목표를 켜면 nominal 전진 평균이.075m/s로 낮아졌다. 팔 움직임/arm state 입력 변화가 있는 조건에서 추가 학습이 필요했다.

`as2piper_plate_walk_wbc_nominal`은 수정한 보행 reward를 유지하고 random EE 목표를 다시 켠다. 이를500 checkpoint에서 이어 학습해 검사한다. `as2piper_plate_walk_wbc`는 이후 hardware-sized payload/COM/push DR까지 활성화한 task다. 정지 명령에서는 네 발의 desired contact를1로 두어 frozen trot 전환점의.5 contact를 강요하지 않는다.

WBC500→502 smoke에서 모델/optimizer를 불러와 실제 업데이트·저장을 통과했다. 이후 측정 결과는 `audit_hw`의 해당 run/evaluation 파일에 기록한다. Resume 후 음수 ETA를 출력하던 문제도 새 복제본의 logging 식만 수정했다. PPO 업데이트 식은 변경하지 않았다.

## WBC 전환에서 확인한 입력 분포 문제

500 gait 모델을 그대로 가져와100회 더 학습한 `walk_wbc_nominal600_20261029`는 random arm 평가에서 전진 평균.095m/s에 머물렀고 회전 조건에서 넘어짐도 발생했다. 이 branch는 권장 모델로 채택하지 않는다. 학습 횟수를 늘리는 것만으로 충분하지 않았다.

Gait 학습 중 arm joint position/velocity와 EE command 입력은 거의 일정했다. 이 입력들이 WBC에서 크게 변하면 그 방향으로 충분히 학습되지 않은 input weight가 기존 leg policy의 출력을 흔들 수 있다. 네트워크 전환 초기화를 별도 복사본에서 검사했다.

```text
old first layer: h = W_body*x_body + W_arm*x_arm + b
reference:      x_arm = gait reference state
new W_arm = 0
new b = b + old W_arm * reference
```

Actor backbone, history encoder의 입력층, critic backbone에 적용한다.71차원 proprio 중 arm q17:23, arm dq35:41, EE position60:63이다. 기준값은 실제 gait 평가 궤적에서 계산했다. 다른 입력은 보존한다. 기준 입력에서 첫 층의 최대 출력 차이는4.8e-7 이하였고, 새로 초기화한 부분의 Adam moment만 초기화했다.

이는 관측을0으로 덮어쓰거나 gradient를 차단하는 방식이 아니다. 실제 arm/EE 입력은 그대로 들어가며 해당 weight는 이후 WBC 학습으로 다시 학습된다. Source gait checkpoint와 초기화된 checkpoint는 별도 파일이다.

초기화 직후 random arm 평가 결과는 다음과 같다(명목 평지,32환경10초, history encoder).

| 명령 | 실제 vx | 실제 yaw | 넘어짐 |
|---|---:|---:|---:|
| 전진 .25 |.250m/s|−.001rad/s|0|
| 제자리 회전 +.4 |.024m/s|.382rad/s|0|
| 전진 .25 + 회전 .4 |.236m/s|.393rad/s|0|

EE 위치 평균 오차는 약1.7–1.9cm였다. 동일한 기구/명령 범위에서 입력 전환 초기화만으로 회복된 결과는 입력 분포 변화가 큰 원인이었음을 뒷받침한다. 실제 모든 arm pose/지형에 대한 보장으로 일반화하지 않는다.

`gait_arm_transfer500_20261029/model_500.pt`가 이 초기화 복사본이다. 이후 `walk_wbc_transfer520_20261029`에서256환경×24step×20iteration의 실제 WBC 학습을 추가했다. 해당 arm 입력 weight norm은0에서 actor.441/history encoder.229로 변해 계속 학습 가능함을 확인했다. 최종 추가 학습 후 평가는 `audit_hw/wbc_transfer520_*`에 저장한다.

### 실제 WBC 업데이트 후 최종520 모델

권장 starter checkpoint:

`logs/as2piper-low/walk_wbc_transfer520_20261029/model_520.pt`

명목 평지·random EE 목표·history encoder·32환경10초의 결과:

| 명령 | 실제 vx 평균 | 실제 yaw 평균 | 넘어짐 |
|---|---:|---:|---:|
| 정지 |−.001m/s|+.006rad/s|0|
| 전진 .25 |**.268m/s**|−.040rad/s|0|
| 제자리 좌회전 .4 |.025m/s|**.357rad/s**|0|
| 전진 .25 + 회전 .4 |**.241m/s**|**.356rad/s**|0|

EE 위치 평균 오차는 전진1.6cm, 회전/전진+회전1.7cm였다. 팔 고정 평가도 넘어짐0이었지만 전진.291m/s로 약.041m/s overshoot가 남았다. 고정 팔 보행 자체의 정확도는 gait500 모델이 더 좋고, random arm 상태 대응은 transfer520 모델이 개선됐다. 완전 수렴이나 모든 명령에서의 최적 성능으로 주장하지 않는다.

동일520 모델에 payload/COM/push DR와 random arm 목표를 함께 적용하면 이동 중 넘어짐0이었고 전진.233m/s, 전진+회전.207m/s/.369rad/s였다. 정지 중에는1건의 넘어짐이 있어 full DR 수렴 정책으로 승인하지 않는다.

최종 영상은 `audit_hw/wbc_transfer520_arm/learned_gait.mp4`다.10초 중 뒤3초, 전진과 전진+회전을 나란히 보여준다. 원시 오차는 매 step 시작의 실제 적용 목표와 reset 전 실제 EE를 비교한 `policy_evaluation.npz`에서 집계했다. 최종 trajectory 파일의 보조 scalar도 이 원시 측정과 맞췄으며 원래 시각화용 v1 기록은 별도로 보존했다.

```bash
# W&B 인증된 vbc 환경에서 nominal WBC 학습을 이어갈 때
cd /data/team/users/suhwan/vbc-as2piper
AS2_TASK=as2piper_plate_walk_wbc_nominal bash low-level-as2piper-hw/launch_hw.sh \
  --resumeid walk_wbc_transfer520_20261029 --checkpoint 520 --max_iterations 45000
```

이번 진단 학습들은 정상 종료했고45000회 online 학습은 시작하지 않았다. 기존에 확인한 suhwan W&B 인증 부재를 우회하거나 인증 파일을 읽지 않았다.

기본 gait model의 payload/COM/push 스트레스 검사에서는 전진.243m/s, 전진+회전.234m/s/.402rad/s였고 이동 중 넘어짐은 없었다. 정지 중 push에서는2개 환경이 넘어졌다. 이 때문에 명목 보행/WBC 전환과 full DR 강건성 학습은 구분한다.

## 재실행

```bash
# 이미 검증한 기본 보행 checkpoint 평가
cd /data/team/users/suhwan/vbc-as2piper/low-level-as2piper-hw
export PYTHONPATH="$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python"
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python evaluate_hw.py logs/as2piper-low/gait_pilot500_20261029/model_500.pt \
  --vx .25 --yaw .4 --output audit_hw/gait_eval_repeat
```

`launch_gait.sh`와 `launch_hw.sh`는 장시간 작업을 터미널과 분리하고 파일 로그를 남긴다. W&B online은 인증된 Linux 사용자 환경에서 사용해야 한다. 이번 진단은 인증 파일을 읽거나 복사하지 않고 W&B disabled로 수행했다.

실제 상태 영상: `audit_hw/gait_eval500_history/learned_gait.mp4`. PhysX에서 기록한 root/joint 상태를 MuJoCo로 재생했으며 렌더 중 물리를 다시 적분하거나 로봇 root/관절 경로를 연출하지 않았다.
