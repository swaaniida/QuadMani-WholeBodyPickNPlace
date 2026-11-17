# AS2+PiPER 시스템 감사 — 2026-10-28

이번 작업은 진단과 보고서 작성이다. 실행 중인 학습, reward/config/정책 코드는 수정하지 않았다. 원본 B1+Z1도 읽기 전용으로 조사했다. 아래는 코드·URDF·로컬 실행 기록으로 확인한 사실이며, 실제 하드웨어와 실제 reset 원인별 통계는 별도 측정이 필요하다.

## 1. 먼저 바로잡을 사실: B1 실행과 AS2 실행은 같은 조건이 아니었다

이전 답변의 “B1에서도 contact shaping이 0이었다”는 설명은 틀렸다. 실제 `original-run01` stdout의 시작 부분과 reward 기록에 반대 증거가 있다. config 파일의 기본값만 읽고 실행 시 CLI override를 놓쳤다.

- B1 run: `original-run01`, W&B ID `auh8t2og`, project `quadruped_capstone/b1z1-low`.
- B1 실제 argv: `--headless --exptid original-run01 --proj_name b1z1-low --task b1z1 --sim_device cuda:0 --rl_device cuda:0 --observe_gait_commands`.
- 시작 로그 24행: `Observe gait commands!`; 207~208행: contact force/vel reward 각각 0.4802/0.4582. 이후에도 변한다.
- B1 로컬 W&B `files/config.yaml`의 args에도 해당 flag가 있다. config.py의 `False`는 CLI 처리에서 `True`로 바뀐다.
- AS2 실제 argv에는 `--observe_gait_commands`가 없다. 현재 contact shaping 0은 이 차이로 설명된다. 센서가 없어서가 아니다.
- B1 proprio=71, total obs=(71×11)+18=799. AS2 proprio=66, total obs=744. gait phase 1개와 clocks 4개가 빠진다.
- 따라서 `check_original_settings.py`의 클래스 기본값 동등성 검사는 실제 B1 실행과의 동등성을 증명하지 못했다.
- W&B 저장 `.py`는 원본 파일을 가리키는 symlink라 과거 코드의 불변 snapshot도 아니다. 실제 argv·stdout·모델 입력 크기를 함께 봐야 한다.
- 제공받은 Forge 프로젝트 URL은 웹 도구에서 열리지 않았다. 대신 위 run ID와 연결된 서버의 원시 W&B 실행 기록으로 확인했다. 다른 run 선택 여부까지 웹 UI에서 확인한 것은 아니다.

근거:
`/data/team/users/suhwan/vbc-legacy/experiments/vbc-lowlevel/original-run01.log`
및 `/data/team/users/suhwan/vbc-legacy/low-level/legged_gym/envs/logs/wandb/run-20261026_192834-auh8t2og/files/config.yaml`.

## 2. 시작 자세와 EE command — 직접 계산

현재 training URDF와 같은 kinematic chain의 MuJoCo FK로 계산했다. root=(0,0,0.5), yaw=0, leg hip=±0.2/thigh=0.8/calf=-1.5, arm C=(0,1.1,-0.55,0,0,0)인 이상적인 reset 한 개다. 실제 reset에는 각도/위치/속도 perturbation과 terrain offset이 추가된다.

| 항목 | 계산 결과 | 의미 |
|---|---|---|
| 발 중심 높이 | 약 0.216m | 반경 0.02m를 반영하면 발바닥 약 0.196m 부유 |
| 정지 착지에 맞는 root 높이 | 약 0.304m | reset z=0.5는 거의 20cm 낙하를 만든다 |
| reset TCP | (0.3704, 0, 0.7572)m | 후보 C의 초기 위치 |
| EE 초기 시작 목표 | (0.7619, 0, 0.8913)m | 실제 TCP와 0.4139m L2 / 0.5257m L1 차이 |
| EE 초기 종료 목표 | (1.0, 0, 0.7)m | standing fixed-base 위치 IK 잔차 약 0.2039m |
| 시작 goal pitch | -0.0127rad | `-π/8 + 0.38` |
| reset TCP pitch | 약 2.034rad | 시작 방향 오차 약 117° |
| far-front 목표 | (1.25, 0, 0.7)m | fixed-base 위치 IK 잔차 약 0.4539m |
| high 목표 | sphere radius .95 / pitch π/3 | fixed-base 위치 IK 잔차 약 0.3244m |
| 전방 바닥 (0.61,0,0.035) | upright fixed-base IK 잔차 약 0.190m | 별도의 body tilt/height coordination 필요 |

IK 수치는 관절 한계를 적용한 12개 시작점의 position-only least-squares 최선값이다. 충돌·orientation을 무시한 완화 문제이므로 실제 조건은 더 어렵다. IK 탐색 잔차를 모든 몸통 자세에 대한 불가능 증명으로 해석하지 않는다. 다만 초기 종료 목표와 현재 고정 베이스에서는 arm-base→goal 거리 약 1.027m가 전체 PiPER→TCP translation 길이 합의 느슨한 상한 0.9391m보다 길다.

AS2 다리 legal joint range의 5,000 표본에서 수직 지지 길이 최대는 앞/뒤 약 0.420/0.423m(foot 반경 근사 포함)였다. 이는 정확한 전역 최대 인증이 아니지만, 정상적인 약 0.3m stance에 B1의 0.55m 높이 목표를 그대로 쓰기 어렵다는 충분한 경고다.

## 3. 우선 해결할 불일치

### 추가 대조: B1 시작 조건

동일한 root z=0.5m, yaw=0, 각 모델의 명목 기본 관절각에서 계산한 결과다. B1은 별도 CPU PhysX 진단에서도 EE=(0.760149,-0.000000,0.912846)m, 앞발 중심 z=0.023836m로 FK와 일치했다. 진행 중인 학습 환경에 접속하거나 변경한 검사는 아니다.

| 항목 | B1+Z1 | AS2+PiPER |
|---|---:|---:|
| 발 collision sphere 하단 z | -0.0162m | +0.1882m |
| 초기 EE 목표와 위치 L2 차이 | 0.0216m | 0.4139m |
| 초기 EE 목표와 방향 차이 | 1.30° | 117.24° |

B1 URDF의 mount origin z 문자열에 `0.09>>` 오타가 있지만 Isaac Gym이 실제 import한 kinematics는 z=0.09 해석과 일치함을 위 CPU 검사로 확인했다. AS2 발 반경은 실제 URDF sphere 값으로 계산했으며 앞 표의 +0.196m는 0.02m 반경 근사값이다. 정확한 현재 URDF 비교값은 +0.1882m다. terrain/각도 perturbation 전의 명목조건이므로 모든 env의 실제 초기 간격을 뜻하지는 않는다.

744차원은 799차원 중 55개 zero padding이 아니다. gait flag에 따라 현재 관측 5개와 history 10회분의 50개가 벡터/네트워크 입력에서 빠진다. 현재 AS2 network log의 history encoder 입력은 66, B1은 71이다.

B1의 같은 초기 log block에서 force metric=-0.2401, force weighted reward=+0.4802, velocity metric=-0.2291, weighted reward=+0.4582를 확인했다. 두 metric도 episode 누적/10 값이지만 동일 normalization이므로 scale -2로 부호가 뒤집혔다는 실행 증거가 된다.

현재 reset 식에는 EE tracking error가 없다. `ik_fail`은 주석이며 직접 조건은 roll/pitch, world-z, timeout 등이다. EE mismatch가 자세 불안정을 거쳐 간접 원인이 될 수 있으나 “못 따라가면 즉시 reset”하는 로직은 아니다.

| ID | 중요도 | 확인된 문제 | 영향 / 필요한 조치 |
|---|---|---|---|
| P01 | 최우선 | B1는 gait flag ON, AS2 OFF | shaping 2개와 관측 55차원 차이. 같은 비교 조건을 원하면 실제 runtime args부터 맞춰야 한다. 변경 시 새 network/run이 필요하다. |
| P02 | 최우선 | 20cm 부유 reset + 41cm EE 이동 + 117° EE 회전 | 초기 낙하·착지와 큰 팔 재배치를 동시에 요구한다. 실제 발바닥 clearance에서 root z를 계산하고 현재 FK에서 EE trajectory를 시작해야 한다. |
| P03 | 최우선 | 초기 종료 목표 x=base+1m | AS2 중앙 장착 PiPER의 정상 stance reach와 맞지 않는다. 초기/랜덤 command 모두 reachability 검증이 필요하다. |
| P04 | 최우선 | root height target=0.55m | AS2 정상 stance와 충돌. 단위는 같아도 동일한 물리 task가 아니다. |
| P05 | 높음 | arm orientation의 roll만 0으로 바꾸고 `arm_induced_pitch=0.38` 유지 | Z1 기준 heuristic을 PiPER TCP frame에 정확히 변환한 것이 아니다. 초기 117° 차이가 그 증거다. FK 기반 기준 orientation과 연속 보간이 필요하다. |
| P06 | 높음 | mass +0~15kg, COM x/y/z ±150mm | nominal 전신 22.73kg에 최대 +66% 질량. raw base mass 8.8kg·작은 몸통 크기 대비 매우 큰 교란이다. 현재 arm mass는 4.816kg. |
| P07 | 높음 | 400/40 arm position drive + URDF effort 100Nm | PiPER 실제 모터 특성 검증 없이 원본 drive를 사용한다. 강한 IK 팔이 착지 중 몸통을 흔들 수 있다. 100Nm는 모델 값이며 실측 하드웨어 한계가 아니다. |
| P08 | 높음 | 종료 원인 기록이 없음 | roll/pitch 0.8rad, world z<0.1, timeout 중 어떤 것이 0.4초 reset을 만드는지 분리할 수 없다. 로그만으로 원인을 단정하면 안 된다. |

## 4. EE 목표 범위·좌표계·시간 조건

현재 center는 `(base_x,base_y,0)+yaw_rotate(0.3,0,0.7)`다. AS2 arm base offset은 `(-0.02,0,0.0820)`인데 goal center는 여전히 전방 장착 B1 기준이다.

- radius 0.4~0.95m, pitch -1.2566~1.0472rad, yaw ±1.2rad, orientation delta 각 축 ±0.5rad.
- yaw=0일 때 최대 x=base_x+1.25m. raw z 범위는 -0.2035~1.5227m다. 이 값은 **필터 전 범위**이며 실제 모든 goal이 그대로 사용된다는 뜻은 아니다.
- underground limit=-0.7은 center-relative z 검사이므로 center z=.7일 때 정상적으로 필터링되면 world z>=0을 요구한다. 그러나 지면 높이를 샘플링하지 않으며 10회 재시도 후 실패한 후보를 반드시 제거하는 보장도 없다.
- 초기 goal은 collision/reachability 검사를 거치지 않는다.
- 기존 collision check는 center-relative TCP 경로의 10개 점과 AABB만 확인한다. AS2 머리·장착판·팔 링크·손가락·관절 한계·orientation·Jacobian singularity를 확인하지 않는다.
- center z=0.7은 코드상 local terrain height를 더하지 않는다. 주석의 'terrain relative'와 실제 world-z 처리가 다르다. base height reward와 z termination도 지면 clearance가 아닌 world z를 쓴다.
- goal center의 XY는 base를 따라 움직인다. 단순히 앞으로 걸으면 현재 상대적으로 먼 goal도 함께 앞으로 이동하므로, 물체 쪽으로 접근해서 이 거리를 줄이는 task와 다르다.
- trajectory 1~3초 + hold 0.5~2초인데 AS2 episode가 약 0.4초다. 많은 episode가 초기 trajectory만 수행하다 reset되며 일반 random goal까지 가지 못한다.
- C 자세는 이전에 leg hip=0/calf=-1.6과 다른 PD로 정적 평가했다. 현재 B1 leg hip=±.2/calf=-1.5, position drive=400/40에서의 안전성은 그 결과로 대체할 수 없다.
- observation의 arm offset은 base yaw만 회전하고 실제 mount는 full base rotation을 따른다. 큰 pitch에서 goal-relative arm frame이 어긋난다. 20° tilt에서 offset 회전 차이가 수 cm 수준일 수 있다.
- EE orientation command는 IK에는 들어가지만 observation orientation 3개는 `0*...`로 채워진다. 랜덤 orientation 요구를 정책이 직접 관찰하지 못한다. EE orientation reward 역시 scale=0/None으로 비활성이다.

## 5. 제어/PPO의 상속된 문제와 task 정의

### 5.1 action in-place 변경과 PPO likelihood 불일치: 코드 경로로 확인

`PPO.act()`는 `self.transition.actions`를 반환하고 그 action에 대한 old log-probability를 저장한다. runner는 이를 clone하지 않고 `env.step(actions)`에 전달한다. env 첫 줄 `actions[:,12:]=0`이 같은 tensor의 팔 값을 지운다. 이후 storage에는 변경된 action이 들어가지만 old log-probability는 변경 전 값이다.

이는 PPO의 new/old likelihood가 같은 action을 평가해야 한다는 조건을 깨뜨린다. 특히 팔 6개 부분과 shared features의 학습을 오염시킬 수 있다. 코드상 데이터 흐름의 불일치는 확인됐지만 현재 수렴 저하에서의 기여도는 별도 실험이 필요하다. 원본 코드에서도 같은 경로이며 AS2만의 오류는 아니다.

### 5.2 실제 action 권한

- 정책 인터페이스 18차원이지만 arm 6개는 실행 전에 마스킹된다. 실제 정책이 다리 12개와 팔 6개를 모두 직접 구동하는 WBC는 아니다.
- arm은 매 step DLS IK→PhysX position drive. 그리퍼 2개는 별도 open target이다.
- IK는 position/rotation 오차를 쌓아 매 policy step joint target에 더한다. joint-limit clamp는 추가했지만 속도·가속도·torque-aware planning, self-collision avoidance, unreachable-goal fallback은 없다.
- 실제 implicit arm drive torque를 `self.torques`에 측정해 넣지 않는다. `_compute_torques()`의 arm 6개는 0이라 torque/work 로그가 팔의 실제 노력을 보여주지 못한다. arm energy reward를 나중에 켜도 먼저 계측을 고쳐야 한다.
- arm motor-strength randomization 값은 explicit torque 경로에 곱한 뒤 팔 torque가 0이 되므로 implicit position drive의 강도 변화로 이어지지 않는다.
- action_delay=3이지만 실제 선택은 초기 최신 frame, 10,000 iteration 이후 1-step 이전 frame이다. 설정의 3-step 지연이 그대로 적용되지 않는다. -1로 껐을 때의 buffer 경로도 재검토해야 한다.
- torque supervision/adaptive gain은 현재 OFF다. runner의 `default_dof_pos[-7:-2]`는 AS2에서 5개만 선택하므로 해당 기능을 켤 때 6-DOF 매핑부터 수정해야 한다.

### 5.3 현재 학습으로 만들 수 있는 행동의 범위

- `vy command=0`이 고정되어 있어 영상의 전방/좌우/후방 다방향 보행 전체를 학습하는 분포가 아니다. 처음 5,000 iteration은 vx>=0, 이후 후진이 포함된다.
- 명시적인 torso pitch/height command가 없다. `--pitch_control` 플래그는 설정에 들어가지만 현재 ManipLoco에서 사용하는 경로가 없다. EE reach 보상을 통해 간접적으로 몸통 자세를 배울 뿐이다.
- gripper는 열린 상태이며 object-conditioned grasp policy나 dragging force training이 아니다. 별도 box는 reset 때 로봇 x보다 2m 앞에 둔다. camera image도 low-level observation에 없다. proprioceptive low-level 자체는 타당한 설계일 수 있으나 high-level perception/grasp controller가 별도로 필요하다.
- 영상의 virtual base servo와 soft-grasp attachment는 실제 학습 환경에 없다. 영상 성공과 학습된 grasp/drag 가능성을 연결해서 해석하면 안 된다.
- command/terrain curriculum 관련 설정 중 실제 구현이 연결되지 않은 항목이 있다. ManipLoco reset에서 일반 command curriculum update는 호출하지 않는다. non-timeout reset에서는 새 command를 항상 뽑지 않고 기존 command를 유지한다.

## 6. Reward·센서·지형·종료 감사

| 항목 | 현재 동작 / 문제 |
|---|---|
| contact shaping OFF | AS2 현재 직접 원인. B1 runtime ON과 불일치. |
| contact shaping 부호 | 현재 복제 코드에서 두 함수는 mismatch에 대해 음수값을 반환한다. scale도 -2라 곱하면 mismatch가 **양의 보상**이다. 예: desired swing 중 큰 force가 있으면 raw<0, weighted>0. flag만 켜기 전에 의도한 부호를 확인해야 한다. 이 함수/scale은 원본에서 상속됐다. 과거 source snapshot은 symlink여서 당시 전체 코드 불변성까지 보증하지는 않는다. |
| walking_dof | `exp(-.05*sum(abs(q-q_default)))`에 양의 가중치. 보행 성공 지표가 아니라 걷기 명령 중 nominal pose 유지 보상이다. 속도 추종이 실패해도 얻을 수 있다. |
| tracking_lin_vel_max | 전진 명령 대비 signed speed ratio이며 반대 방향은 음수가 될 수 있다. 정지 명령에는 별도의 정지 보상이 섞인다. 평균만으로 이동 구간 추종률을 알 수 없다. |
| EE reward | exp(-2*L1 position error), scale=.8. 0.6이라는 W&B episode 값은 성공률 60%나 거리 0.6m가 아니다. |
| episode logging | cumulative reward/metric을 고정 max_episode_length_s=10으로 나눈다. 생존 시간이 달라지면 모든 episode reward의 분포가 달라진다. per-step 평균·시간·명령 구간별 통계가 필요하다. |
| 음수 누적과 조기 종료 | only_positive=False이고 많은 패널티가 있다. episode가 짧아져 총 음수 누적이 줄어도 reward가 개선될 수 있다. 실제 early-termination 전략을 학습했는지는 별도 진단해야 한다. |
| base_height | 0.55m world-z target이며 terrain-relative가 아니다. AS2 정상 stance와 맞지 않는다. |
| feet_height | 앞발 두 개 world-z의 norm을 0.3과 비교한다. 각 발의 local ground clearance나 swing 높이가 아니다. 지형 높이·AS2 크기에 민감하다. |
| feet_air_time | 기본은 앞발만 평가하고 0.5초 기준이다. 원하는 duty factor와 AS2 gait frequency에서 적합한지 재검증 필요. |
| feet_jerk | `hasattr(self,'last_contact_forces')`를 reward container에서 검사하지만 저장은 `self.env`에 한다. 현재 경로에서 계속 0이 될 수 있는 상속 코드 오류다. |
| contact force penalty | episode>2초에서만 활성. 0.4초 reset이 대부분이면 착지 충격을 이 reward로 제대로 평가하지 못한다. |
| force sensor contact 판정 | 6D wrench norm(힘+토크)을 1.5와 비교한다. N과 Nm를 혼합하며 발 센서 위치 변화에 따라 판정이 달라질 수 있다. 실제 normal force/contact와 calibration 필요. |
| 팔 노력 | torque/work reward는 implicit arm drive의 실제 torque를 포함하지 못한다. |
| rough terrain | roughness 0~0.1m, curriculum=False라 초기부터 전체 difficulty를 샘플한다. AS2 leg reach 대비 큰 장애물이다. |
| collision mesh 근사 | Delatin max_error=.1m라 지형 단순화 오차 허용이 AS2 foot clearance와 비슷하거나 클 수 있다. 높이 맵과 실제 mesh 간격을 검증해야 한다. |
| 지형 관측 | measure_heights=True지만 현재 66/71차원 proprio 벡터에는 height map이 없다. 현재 학습은 사실상 proprioception/history 기반이다. |
| z termination | world z<.1; 지형 기준 몸통 clearance가 아니다. |
| tilt termination | roll/pitch 각각 ±.8rad(약46°). 원래 원하는 tilt와 실패 tilt를 구분하는 command-aware 판단이 아니다. |
| 몸통 접촉 termination | terminate_after_contacts_on=[]라 즉시 종료 조건이 아니다. contact penalty만 있다. |
| 원인 분해 계측 | termination cause, joint target saturation/actual drive torque, individual loaded feet, slip, TCP error/goal reachability, terrain-relative clearance 통계가 없다. |

## 7. 자산·모델링: 맞는 부분과 아직 미확정인 부분

확인된 정상 연결: 20 DOF 순서, 18 action 인터페이스, gripper 2개 분리, FL/FR/RL/RR 발 body·센서 4개, UMI nominal TCP body 유지, free-base Jacobian의 마지막 gripper 2개 제외한 arm 6개 선택. 기존 reindex는 native FL/FR/RL/RR와 policy FR/FL/RR/RL 사이에서 유지된다. 이번 증거로는 이 매핑이 원인이라고 볼 이유가 없다.

미확정: 실제 AS2 구조 frame과 plate 체결 하중 경로, 나사/CAD 패턴, PiPER 실제 effort/속도/전류·gain·latency, 카메라/케이블 질량·간섭, UMI proxy collision 및 nominal TCP와 실제 접촉점의 관계. URDF PiPER effort=100Nm 일괄값은 실제 각 관절 하드웨어 사양으로 승인할 수 없다. 카메라가 빠진 CoM/inertia도 실제 조립과 다를 수 있다.

AS2 base primitive와 팔 mesh의 MuJoCo soft-contact overlap은 이전 시각화에서도 관찰됐다. Isaac Gym의 convex collision과 fixed-joint collapse 후 형상이 같다고 가정하면 안 된다. neutral/goal trajectory 전 구간의 실제 Isaac contact pairs와 거리·힘을 따로 검사해야 한다.

## 8. 권고 순서 — 이번에는 수행하지 않음

1. 실제 B1 argv/runtime config를 기준으로 비교 프로토콜을 재정의한다. gait ON이면 관측 크기가 799가 되므로 현재 744차원 checkpoint를 같은 실험으로 무조건 resume하지 않는다.
2. reset 자세의 foot-ground clearance를 0 근처로 맞추고, arm FK에서 EE goal을 시작한다. 초기 goal은 position/orientation 연속성을 갖게 한다.
3. AS2 stance/tilt 조건별 TCP reachability map과 collision-free IK로 command 분포를 만든다. terrain-relative height와 PiPER TCP frame을 일관되게 정의한다.
4. 팔/다리 실제 actuation 범위와 dynamic coupling을 검증하고 randomization을 nominal AS2에서 점진적으로 늘린다.
5. shaping 부호와 PPO action aliasing 등 상속 코드 문제를 별도 변경으로 검증한다. reward 튜닝과 implementation bug fix를 섞지 않는다.
6. neutral stand → 작은 velocity command → arm goal 이동 → 전방 tilt/reach → 복합 command 순으로 검증한다. 평지 zero-action과 randomization 제거 진단은 원인 분리를 위한 별도 실행으로 한다.
7. 원인별 reset·물리 단위 raw metric·실제 arm torque·가시성·reachability를 기록하고, online W&B에 post-CLI runtime config·asset/source hash를 불변 snapshot으로 남긴다.

현재 run은 단순히 B1 reward 분포와 다른 것이 아니라, B1 runtime과 gait 입력부터 다르고 AS2 물리 조건과 초기 task가 일치하지 않는 실험이다. 앞서 “원본과 동일하게 준비됐다”는 판단은 실행 가능성 검사에 치우쳤으며 위 검토를 대신하지 못했다. 이 보고서는 수정·재시작의 근거이고, 현재 실패 원인의 기여도를 모두 확정한 보고서는 아니다.
