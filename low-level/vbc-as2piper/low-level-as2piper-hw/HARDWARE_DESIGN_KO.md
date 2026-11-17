# AS2+PiPER 하드웨어 기반 보행 학습 설계

현재 기본 task의 감사·복구 기준은 `VBC_PORT_AUDIT_KO.md`다. 아래의 초기 DR/action 제한과 plane 설정은 과거 진단 조건이며 현재 학습 설정 전체를 나타내지 않는다.

이 문서는 하드웨어 초기 설계/20회 진단 기록이다. 이후 실제로 보행을 학습한 변경과500회 checkpoint 평가는 `GAIT_FIX_KO.md`를 우선한다. 현재 `run_hw.sh` 기본 task도 그 보행 수정이 포함된 WBC task다.

## 목적과 보존 범위

기존 B1 수치를 질량이 다른 AS2에 그대로 적용한 것은 하드웨어 포팅으로 충분하지 않았다. 새 `low-level-as2piper-hw` 복제본에서 다리 제어·탐색 범위·DR·물리 기준을 계산했다. 기존 `vbc-legacy`, `low-level-as2piper`, `low-level-as2piper-forward`의 학습 소스/URDF/checkpoint는 수정하지 않는다.

## 측정 기반

통합 URDF/MJCF의 명목 질량은 B1+Z1 약57.987kg, AS2+PiPER+UMI+plate 약22.728kg다. 제조사 카탈로그에서 본체 질량만 가져온 비율이 아니다. 명목 stance의 base 높이는 .516164m와 .324407m다. 기존 B1 URDF의 `0.09>>` 문자열은 별도의 기존 workspace-study 복사본에서만 정규화돼 있다.

명목 CoM/관성, 비음수 수직 발 하중, 발 Jacobian으로 정지 지지 토크를 계산했다. AS2 발당 약51–60N, calf 최대 약9.04Nm, B1 calf 최대 약34.86Nm였다. 이는 평지 네 발 수직 하중에 대한 정역학 계산이며, 주행 충격·마찰·모터 연속출력 인증이 아니다.

AS2 leg effort/velocity는 원본 URDF의 hip/thigh60Nm·24rad/s, calf90Nm·16rad/s다. 실제 연속 토크나 전기적 열 한계로 해석하지 않는다. PiPER URDF의 일괄100Nm effort는 실측 모터 한계가 아니므로 실제 하드웨어 승인값이라고 주장하지 않는다. 팔 drive400/40과 finger1800/35는 이번 고정몸통 추종 검사를 통과한 기존 시뮬레이션 제어를 유지한다.

## 단순 비례 대신 사용한 식

질량비 `a=M_AS2/M_B1=.391945`, 길이비 `l=L_AS2/L_B1=.628496`으로 두었다. L은 현재 자세의 기하학적 stance 높이이며 두 로봇이 완전히 닮았다고 가정하지 않는다.

동일 중력에서 시간 단위 `T=sqrt(L/g)`를 사용한 차원 분석은 다음 시작점을 준다.

```text
time ratio       = sqrt(l)             = .792777
velocity ratio   = sqrt(l)
frequency ratio  = 1/sqrt(l)
torque ratio     = a*l                 = .246336
angular Kp ratio = a*l
angular Kd ratio = a*l^(3/2)           = .195290
```

동적 유사성은 설계 출발점이지 다른 기구의 성능 보장이 아니다. 관련 원리의 참고 문헌은
[Dynamic similarity and scaling for the design of dynamical legged robots](https://ieeexplore.ieee.org/document/7354189/)다.
위 수치는 본 프로젝트 질량·길이에서 계산한 것이며 논문의 AS2 추천 gain이 아니다.

Kp를 이 비율로만 줄이면 약19.7이 되어 calf 하중에서 큰 정적 처짐을 허용하게 된다. 따라서 관절별로 다음 하한을 함께 적용했다.

```text
Kp_j = max(
    80*a*l,
    max_static_support_torque_j / .12rad,
    I_effective_j * (2*pi*2*gait_frequency)^2
)
Kd_j = 2*.7*sqrt(Kp_j*I_effective_j)
```

.12rad 처짐 허용, 감쇠비.7, gait 주파수의2배 swing 응답은 명시적인 초기 설계 선택이다. 실험에서 검증할 항목이며 모터 식별로 얻은 값은 아니다. I_effective는 명목 관절 질량행렬 대각값에 원본 AS2 MuJoCo preset의 armature를 더했다. 결합된 전신 관성과 감속기 마찰을 완전히 모델링한 것은 아니다.

원본 preset: `/data/team/users/suhwan/robot-sources/unitree_ros/robots/as2_description/as2.xml`.
hip/thigh armature .026, calf .038kg·m²를 Isaac Gym의 실제 DOF armature에 넣고 다시 검사했다. 원본에 있는 simulation preset과 실제 모터 식별값은 구분한다. armature를 반영하기 전 탐색 결과는 `audit_hw/without_armature`와 `audit_hw/ppo20`에 보존한다.

## 현재 적용값

| 항목 | 기존 B1 기반 | 현재 AS2 HW |
|---|---|---|
| hip/thigh/calf Kp |80/80/80|59.467/53.404/75.327|
| hip/thigh/calf Kd |2/2/2|2.626/2.358/2.628|
| leg armature |기존 asset 기본0|.026/.026/.038|
| action scale |.4/.45/.45|.2018/.2247/.2390|
| 초기 leg noise std |.8/1/1|.3/.3/.3|
| 기본 trot 위상 주파수 |2Hz|2.523Hz|
| 최대 전진/후진 envelope |±.8m/s|±.634m/s|
| yaw envelope |±1rad/s|±1rad/s 유지|
| 추가 body mass |0–15kg|0–2.273kg|
| body CoM perturbation |각축±150mm|x±27.4mm, y/z±21.4mm|
| leg reset noise |±20%|±5%|
| 초기 root velocity perturbation |±.1|±.02|
| friction DR |.3–3.0|.6–1.2 초기 범위|
| leg strength DR |.7–1.3|.9–1.1 초기 범위|
| max push velocity |.5m/s|.396m/s|

각도 action은 질량 단위가 아니므로 질량비로 곱하지 않았다. 관절별 `min(B1_scale, .2*URDF_effort/Kp, .30rad)`로 목표 offset의 토크 예산을 제한한다. 초기 std와 결합한 명목 목표각 표준편차는 약3.5–4.1°다. 정책 평균이 학습되면 실제 목표각은 이 noise 표준편차보다 커질 수 있다.

질량 DR은 선형 질량 스케일과 전체 명목 질량10% 중 작은 값으로 제한했다. CoM 범위는 전체 CoM 변화가 발 지지영역 반폭의5% 이하가 되도록 base-link 질량8.8kg로 환산했다. friction/motor-strength 범위 축소는 초기 학습용 선택이며 재질·모터 실측 범위라고 주장하지 않는다.

처음1000 iteration 동안 command envelope를25→100%로 늘린다. 원본의5000 iteration 전 전진 우선 규칙은 유지하고, stand 명령을 명시적으로20% 포함한다. 변경한 학습 과정이므로 "환경 전부 B1과 동일"하다고 설명하지 않는다.

## 물리 기준 reward와 gait 부호 수정

reward 가중치는 원본 그대로다. 하드웨어 단위가 있는 다음 기준값을 조정했다.

- Base height target: 기존 forward의 AS2 기하학 기준 .324407m 유지.
- Contact force threshold: 원본40N은 AS2 네 발 정지 하중보다도 작다. 두 발 지지에서1.5배 동하중을 허용하는 `1.5*M*g/2 = 167.22N` 사용.
- 기존 feet-height 함수는 앞발 두 개의 world-z norm을 비교한다. 현재 평지에서 발 반경.028m와 stance 높이10%의 swing clearance를 기준으로 norm 목표.08548m를 사용한다. 이것은 발 하나를85mm 들라는 뜻이 아니다. 해당 함수의 rough-terrain 적용은 별도 수정/검증이 필요하다.
- Contact force²/velocity² exponential sigma는 질량비²/길이비를 반영한다.

현재 코드의 contact-shaped force/velocity 함수는 음수 raw를 반환하고 scale도 음수여서 위반을 양의 보상으로 바꾼다. 논문 표2의 비음수 위반량×음수 가중치와 다르다. HW task의 함수 wrapper에서만 raw 부호를 반전해 **위반이0 또는 음의 보상**이 되게 했다. 원본 reward 파일/가중치와 다른 항의 수식은 유지했다. 이것은 하드웨어 스케일링과 별개인 gait reward 버그 수정이다. 검증 스크립트는 두 항의 weighted reward가 양수가 아닌지 검사한다.

## EE 후보 공급 보완

독립 구면 위치/방향 proposal과 구면 lerp는 유지한다. 추가 pose 검사에서 실패하면 방향delta도 재추출한다. IK12회 미수렴 후보에는52회 추가 풀이를 적용한다. 모든 후보가 실패하면 이전 목표를 유지한다. 초기 start는 FK와 맞춘 채 end만 `[.46,.25,0]`로 두어 처음부터 작은 이동을 경험하게 한다.

64→32환경 등 서로 다른 조건의 짧은 제어 검사는 수렴 성능 비교가 아니다. 고정 몸통32환경15초 검사에서 목표 요청149건 중145건 수락, fallback4건이었다. random EE 위치 평균1.74cm/p95 2.92cm, 방향 평균4.07°/p95 7.52°였고 arm contact>1N은 관측되지 않았다. 이 검사는 다리 보행을 인증하지 않는다. 팔 구동 gain은 동일하므로 leg armature 변경과 분리한 검사다.

## B1 gait과 WildLMa 확인

[VBC 공식 코드](https://github.com/Ericonaldo/visual_wholebody)는 low-level에서 보행과 EE/속도 추종을 학습한 뒤 high-level에서 그 정책을 사용하도록 안내한다. 현재 로컬 B1 train도 resume=false로 시작하며 사전 학습된 Unitree gait를 로드하지 않는다. gait clock/접촉 reward는 기준을 줄 뿐 관절/발 궤적을 실행하지 않는다. 별도 보행-only 선행 학습 단계를 거쳤다는 설명은 확인하지 못했다.

[VBC 논문 §3.1, Appendix C.1](https://arxiv.org/html/2403.16967v5)은 RL로 속도/EE 추종을 학습하고 phase reward로 trot을 유도한다고 설명한다. 해당 논문과 현재 공개 코드의 세부 숫자·수식이 모두 동일하지는 않다.

[WildLMa 논문 §III-A](https://arxiv.org/html/2411.15131v2)는 사전 학습된 VBC 저수준 controller 위에 VR 인터페이스를 연결한다. 오른손은 EE pose, pinch는 gripper, 왼손 가상 조이스틱은 base 선속도/각속도를 제공한다. 따라서 base velocity 입력을 제거한 방법이 아니다. 왼손 입력이 중립이면0 명령을 줄 수 있지만 이는 입력 채널 제거와 다르다. planar 속도0은 몸통 높이·pitch/roll까지 고정한다는 뜻도 아니다.

공식 프로젝트에서 WildLMa 저수준 재학습 config 전체를 확인하지 못했다. 논문이 pretrained VBC 사용을 명시하므로 그 재사용은 확인할 수 있지만 "모든 학습 하이퍼파라미터/가중치가 VBC와 완전히 동일"하거나 "velocity command 없는 정책으로 재학습"했다고 단정할 근거는 없다.

## 검증 범위

현재 평지 task는 정상적인 locomotion을 학습할 출발점을 검증하는 단계다. 명목 서기 통과, 많은 목표 수락, 낮은 EE 오차만으로 이미 정상 보행한다고 부르지 않는다. 실제 policy의 선속도/yaw 추종과 발 접촉을 `evaluate_hw.py`로 따로 측정한다. rough terrain, 지면 물체에 대한 world-fixed EE 추종, 외력 dragging, 카메라/케이블 질량, 실제 하드웨어에서의 안정성은 추가 검증 대상이다.

확장 checkpoint는 history Adam·카운터·RNG도 저장하지만 PhysX contact/episode 상태는 재시작한다. 원본7000 모델을799차원 새 모델에 그대로 이어 붙이는 방식이 아니다. stdout/stderr 파일 기록과 `nohup setsid`로 장시간 프로세스를 VS Code 터미널에서 분리한다.

## 최종 적용값 검증 결과

최종 armature/gain 조건에서32환경×10초, EE는 초기 목표에 고정한 제어 검사:

| 조건 | reset | upright 상태 비율 | 첫7초 reset | 최대 적용 leg torque |
|---|---:|---:|---:|---:|
| 명목 평지·zero action |0|100%|0|15.29Nm|
| 축소 DR·zero action |3|99.27%|0|34.85Nm|
| 축소 DR·std.3 Gaussian action |3|99.41%|0|44.08Nm|

DR 검사에는8초 주기의 push가 포함돼 있다. 넘어짐은 그 이후 구간에 관측됐으며 zero action controller가 외란을 회복하는 보행 정책이라는 뜻은 아니다. 명목 검사에서 네 발 모두5N 이상 하중을 지속했고 평균 body 높이는.3035m였다. 최대 torque는 실제로 명시 적용한 다리 PD torque다. 팔 DOF force tensor는 total force이며 별도 실제 motor torque라고 보고하지 않는다.

최종 설정 PPO 진단:1024환경×24step×20iteration=491520 transitions, W&B disabled,207.99초.

- Sampling 요청2212건 중2209건 수락, fallback3건(.136%).
- 기록512환경 상태245760개 중 random command 적용44.96%, upright99.85%.
- Upright·random command·실제 reset후.5초 이후110151개 상태에서 위치 평균2.01cm/p95 3.60cm, 방향 평균4.65°/p95 8.41°.
- 전체1024환경에서 roll/pitch 종료36건, timeout981건. 초기 episode counter를 무작위로 시작하므로 timeout을 모두 실제10초 생존으로 해석하지 않는다. 상태 오차 집계의 age는 done에서 실제 경과 시간을 재구성했다.
- 이는 여러 설정을 함께 바꾼 짧은 진단이며 gain 하나의 효과나 수렴 보행 성능을 분리 입증하지 않는다.

20회 checkpoint를 사용한 명목 평지·history encoder·deterministic action 평가(32환경10초): 모든 명령에서 넘어짐0이었지만, 전진.15m/s 명령의 실제 평균은 약−.001m/s, yaw.25rad/s 명령의 실제 평균도 거의0이었다. **현재 정책은 서기를 유지하지만 아직 보행을 학습하지 못했다.** 이 결과를 정상 gait 성공이라고 부르지 않는다. 장시간 학습 후 같은 평가로 속도 추종·발 접촉을 다시 측정해야 한다.

확장 checkpoint의 finite weight와 history/PPO Adam 저장을 확인했다.2→3회 재개 smoke에서 history Adam의 exp_avg가 그대로 유지되고 다음 iteration counter가3이 되는 것도 검사했다. 이 smoke는 최종 armature 적용 이전 실행이며 checkpoint 저장/복원 경로 검사다.

실제 PhysX 상태의 초기/후기4초 영상은 `audit_hw/ppo_armature20/ee_initial_replay.mp4`, `ee_late_replay.mp4`다. MuJoCo는 상태 재생 렌더링에만 사용했고 물리를 재계산하지 않았다. 평면은 이번 실제 학습 지형과 같다.

45000회 online run을 시작하려 했으나 현재 Linux 사용자suhwan의 W&B 인증이 없어 `api_key not configured (no-tty)`로 초기화 중 종료했다. **장시간 학습은 시작되지 않았다.** 기존 로그인된vbc 환경에서 README의 launch 명령을 실행해야 한다. 인증 파일을 읽거나 복사하지 않았다. launch script는 즉시 종료를 확인해 실패한 PID를 정상 실행처럼 보고하지 않도록 보완했다.
