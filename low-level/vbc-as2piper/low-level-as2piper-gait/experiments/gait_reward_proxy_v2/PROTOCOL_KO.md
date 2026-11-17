# VBC 동일 물리 환경: 보상 입력 및 높이 기준 비교

## 공통 조건

6144환경, seed7, episode10초, sim dt0.005s/control dt0.02s, rough-flat100% trimesh, 전체 DR를 유지한다. 사용자 확인에 따라 AS2 하드웨어 비례값(추가 질량0~5.879kg, CoM 각 축±9.427cm, push 최대0.3964m/s)을 사용한다. 마찰0.3~3.0, 모터0.7~1.3, 8초 push, 원래 reset/command schedule/random EE/PPO/network/799관측/18action/20DOF는 공통이다. AS2의2Hz 위상 입력도 공통이며 B1 원본 기본파일의 gait flag와 차이가 있음을 명시한다.

센서 설정, 관측값 생성 코드, 물리 계산 주기, terrain, 제어기는 변경하지 않는다. 진단 중 추가했던 센서/물리 샘플링 hook은 production base에서 제거했다. `manip_loco.py`는 보존한 AS2 VBC 포트와 바이트 단위로 같다.

## 측정 검증과 채택하지 않은 변경

- 실제 엔진 질량22.727724kg: URDF 합과 일치.
- 평지 정지 시 raw net-contact Fz 합/중량≈1.0. 같은 센서의 rough 평가 평균은≈0.30이었다. SDK는 triangle mesh 접촉력 보고가 누락될 수 있음을 명시한다.
- 공식 예제와 같은 sensor flag를 적용해도 AS2 발 센서 합은 중량의 약4.6%였다. calf/hip으로 옮겨도 ground reaction force를 얻지 못했다. 이 설정을 실제 GRF로 채택하지 않았다.
- 실제 충돌 mesh와 발 구의 최소거리<2mm를 기준으로4환경·5초 동작을 사후 검사했다. rough에서 기존 관측용 접촉 bit는 precision1.000/recall0.868/agreement0.915였다. raw contact force bit는 agreement0.590, 새4물리샘플+2제어프레임 filter는0.839였다. 따라서 관측과 센서를 바꾸지 않았다. 이 기준은 기하학적 근접 접촉이며 실제 정상반력 N의 측정이 아니다.
- 비대칭 heightmap의 mesh 좌표 변환은 정렬됨을 확인했다. 이전 P95 4.35cm 수치는 접지 발 바닥과 보수적인3이웃 grid 추정값의 차이이며, 메시 오차만을 뜻하지 않는다.

참고: [SDK 문제와 권장 사항](https://github.com/leggedrobotics/legged_gym#known-issues), 설치된 SDK의 `docs/programming/terrain.html`, `forcesensors.html`.

## 비교군: 처음부터 학습

| Preset | force shaping 입력 | 속도/yaw 보상 | 발/몸통 높이 보상 기준 |
|---|---|---|---|
| 과거 stable 대조군 | raw net contact | Gaussian, weight2/1 | 보수적 grid |
| stable_proxy | 기존 관측용 접촉 bit | Gaussian, weight2/1 | 보수적 grid |
| progress_gain2_proxy | 기존 관측용 접촉 bit | Gaussian+진행도, weight4/2 | 보수적 grid |
| stable_proxy_ground | 기존 관측용 접촉 bit | Gaussian, weight2/1 | 마지막 실제 접촉 발 높이 |

기존 force 항은 하중이 걸리면 거의 포화되는 지시 함수다. 새 항은
`mean((1-desired_contact)*existing_contact_bit)`에 기존 weight -2를 적용한다.
그 bit는 기존 관측에 이미 쓰이며 별도의 관측이나 GRF 크기를 만들어 넣지 않는다.

stable_proxy_ground는 접촉 bit가 참일 때 발 중심 높이를 기록한다. swing 목표는
마지막 지지 발 중심에서 최대4cm 위이며 Gaussian은 대칭이다. 몸통 지면 기준은
네 발의 마지막 지지 높이 평균에서 발 반지름을 뺀 값의 EMA0.2다. reset 시
기존 환경 원점으로 cache만 초기화하며 물리 초기 상태를 바꾸지 않는다.

stable_proxy 대 과거stable은 force shaping 입력 효과를, progress_gain2_proxy 대
stable_proxy는 형태+추종 weight 조합을, stable_proxy_ground 대 stable_proxy는
발/몸통 높이 보상 기준의 조합 효과를 비교한다. 조합 내부 모든 개별항의
인과 효과까지 분리했다고 주장하지 않는다.

## 학습과 평가

- 초기 시작은 전부 scratch, 동일 seed/6144환경, 최대600회다. 외부 run의 사전 학습 checkpoint는 사용하지 않았다.
- 세 실험이 2026-10-30 07:12:17 UTC에 동시 중단되었다. 정확한 종료 원인은 미확정이다. 이후 자체 checkpoint에서 resume1으로 재개했다. stable_proxy 및 progress_gain2_proxy는201회, ground는101회부터 재개했다. optimizer/history는 복원하지만 물리 환경·episode·지지 cache는 복원하지 않으므로 중단 없는 학습과의 동일성 및 재개 시점 차이는 보장하지 않는다. 상세는 resume_plan.json에 기록했다.
- periodic model100은101회 완료, model200은201회 완료, model400은401회 완료다. 최종model600은600회 완료다.
- 동일한 실제 완료 update끼리 비교한다. 평가 seed20261102, rough+전체DR, random EE,64환경×20초(명령당16환경).
- 속도 오차는 처음2초 제외. 넘어짐은 전체20초에서 집계. 같은 terrain/DR 추출 hash도 비교한다.
- 발 접촉력 raw 값은 SDK 제한을 표시한다. 실제 이동, 기울기, 넘어짐과 실제 mesh 기반 발 움직임을 함께 확인한다.
- 별도 holdout 검증을 통과하면 완료된 optimizer update를 저장한 뒤 자체 실험만 협조적으로 종료한다. 기존 온라인 학습/B1은 중지하지 않는다.

## Baseline 판정

정지 vx MAE<0.08/yaw MAE<0.15, 전진 vx MAE<0.10 및 평균오차<0.10,
회전 vx MAE<0.10/yaw MAE<0.25, 전진+회전 vx MAE<0.10/yaw MAE<0.25,
넘어짐 없는 환경>=95%, 각 명령에서 roll/pitch<0.35rad 비율>=99%를 확인한다.
전체 무넘어짐 여부는 별도 표시한다. body contact tensor도 같은 SDK 제한을
받으므로 그 값만으로 몸통 무접촉을 보증하지 않는다. 실제 발 lift/slip과
영상으로 보행을 확인하고, 서 있기만 하는 모델을 성공으로 분류하지 않는다.

## 추가 진단과 재현

- `matched_environment.json`: 실제 실행 snapshot을 비교했으며, 세 후보 사이의 보상 외 config 차이는 없다. PPO·asset·terrain·환경 원점 해시가 같다.
- `current_config_validation.json`: 현재 코드가 실행 당시 preset의 값을 유지함을 확인했다. constructor의744/66차원은 simulator 초기화 때 gait 5값을 현재·history11개에 추가하여799/71로 변한다. terrain의200개 sub-terrain도 생성 시 파생된다. 이 차이를 보상 설정 변화나 zero padding으로 해석하지 않는다.
- `geometric_gait.json`: 실제 triangle mesh 및 움직이는 상자와 발 구의 최소거리로 접촉·들림을 검사한다. 이는 정상반력 N이나 관통 깊이 검증이 아니다. 첫2초/종료/큰 기울기 프레임을 제외한 통계이며 전체 넘어짐 집계와 별개다.
- 201회 stable_proxy는 발 접촉 변화가2Hz지만 전진이0.012m/s다. 발 디딤과 명령 추종은 분리해서 판단한다.
- 진행도+가중치 강화 후보201회의 전진은0.164m/s, 회전은0.117rad/s, 전체 넘어짐39회다. 안정적 보행으로 채택하지 않았다.
- `progress_gain2_proxy_ground`는 높이 기준 하나만 바꾸는 후속 비교를 위한 **준비된 preset**이다. 아직 실행하거나 검증한 후보로 분류하지 않는다.
- 후속 평가에서는 옆 방향 속도와 실제 발/상자 속도를 추가 기록한다. 기존 sample column0~81은 유지하며 이후 컬럼을 추가한다. 이는 평가 기록만의 변경이며 학습 환경·보상·관측에는 영향을 주지 않는다.

재개된 실험의 상태/평가표는 `watch_proxy_study.py`가 최대4시간 동안1분 간격으로 갱신한다. 학습 프로세스를 시작하거나 중지하지 않는다.

```bash
cd /data/team/users/suhwan/vbc-as2piper/low-level-as2piper-gait
python report_proxy_v2.py
cat experiments/gait_reward_proxy_v2/RESULTS_KO.md
```

영상은 저장된 PhysX의 root/joint state를 실제 지형 메시 위에서 재생한다. PD 데모가 아니며 물리 시간을 느리게 재생하지 않는다. 표시된4개 환경은64개 전체를 대표하지 않으므로 영상만으로 통과 여부를 결정하지 않는다.

## 2026-10-30 08:46 UTC: yaw 가중치 단독 비교

- `stable_proxy_resume1`은101/201회에서 명령 이동이 거의 없어335회 완료 후 협조 종료했다. model335와 Adam/history를 저장했고, 이미 저장된 model300의301회 평가는 계속한다.600회를 완료한 실험으로 표시하지 않는다.
- `progress_gain2_proxy_yaw4`는 기존 강화 후보의201회 checkpoint에서 분기했다. 비교 대상 `progress_gain2_proxy_resume1`도 동일 checkpoint에서 재개했으므로, 공통 정책·Adam/history·seed7에서 yaw 추종 weight만2→4로 바꾸는 비교다. planar weight4, grid 높이 기준 등 나머지는 같다. 각 분기에서 물리 환경은 재초기화하며 bitwise 재현을 주장하지 않는다.
- 이 실험은 scratch가 아니다. 이후 production을 처음부터 시작할지 결정하기 위한 보상 비교이며, 기존 온라인 run이나520회 transfer를 사용하지 않았다.
-301회 강화안은 전진0.193m/s, yaw0.225rad/s, 전체 넘어짐47회였다. lateral MAE는 약0.10~0.12m/s로 확인되어, 이후 통과 판정에는 각 명령 lateral MAE<0.10m/s도 포함한다. 이전 데이터는 lateral 기록이 없으므로 검증 완료로 표시하지 않는다. 이 기준 추가 이전에도 모든 full-environment 후보가 탈락했다.
- 높이 기준만 바꾸는 `progress_gain2_proxy_ground`는 여전히 준비 상태이며 실행하지 않았다.
- 기록의 완료 update는 runner 학습 iteration이다. 원래 VBC와 같이20회마다 history encoder(DAGGER)를 갱신하고, 나머지 iteration에서 PPO를 갱신한다. 이 스케줄을 변경하지 않았다.

## 2026-10-30 09:04 UTC: rough swing 보상 그룹 비교

`stable_proxy_ground_resume1`은 model300(301회 완료)을 저장한 뒤305회 완료 상태를 별도 저장하고 협조 종료했다.301회 평가는 계속한다. 장시간에도 학습이 불가능하다고 단정한 것이 아니라, 단기 비교가 끝난 후보의 자원을 후속 원인 비교로 전환했다.

`progress_gain2_proxy_ground_floor`는 동일한 강화 정책201회 checkpoint에서 분기한다. planar/yaw weight4/2 및 progress 혼합은 기본 강화안과 같다. 변경은 다음 높이 보상 그룹이다.

1. 발·몸통의 지면 높이를 기존 접촉 bit의 마지막 실제 접촉 높이 기준으로 바꾼다.
2. rough terrain에서 swing은4cm를 최소 목표로 하여 더 높은 swing의 초과 높이 감점을 없앤다. stance에는 지면 기준0의 대칭 목표를 유지한다. 물리적인 joint/collision 한계를 없앤 것이 아니다.
3. plane에서는 기존4cm 대칭 Gaussian을 그대로 사용한다.

이는 높이 기준과 swing 손실의 **조합 효과**다. 서로의 개별 효과까지 분리했다고 주장하지 않는다. 기록된 궤적에서 높이 감점의 변화를 계산한 자료는 `counterfactual_swing_height.json`이며, 보상이 높아졌다는 사실만으로 실제 보행 개선을 주장하지 않는다.

`check_study_config.py`로 기존 세 preset의 값이 바뀌지 않음, yaw 분기는 weight 하나만 변경됨, stance의 떠 있는 발은 계속 감점됨, rough의 높은 swing은 허용됨, plane 보상은 유지됨을 확인했다. full-environment 학습 및 별도 seed 검증을 통과하기 전 공개 기본 task로 채택하지 않는다.

### 동일성의 정확한 의미

같은 config/seed/URDF/PPO와 실제 terrain·환경 원점·정적 robot DR(질량·CoM·마찰·모터) 해시를 비교한다. 이것이 모든 reset·EE·push의 난수 경로까지 같다는 뜻은 아니다. 정책에 따라 종료 시점과 RNG 소비가 달라지고 GPU 물리는 bitwise 재현을 보장하지 않는다. 단일 학습 seed의 실험 결과이며 모든 하드웨어 조건에서의 수렴을 증명하지 않는다.

### history/teacher 비교

`teacher_vs_history_301.json`은 동일한301회 정책과 terrain/정적 DR에서 첫10초를 비교했다. 회전 평균 yaw는 history0.205,teacher0.210rad/s, yaw 표준편차는 둘 다 약0.474rad/s였다. 현재 흔들림을 history 적응 지연만으로 설명하는 근거는 없었다. 이 검사는 원인 진단이며 독립 holdout 검증을 대신하지 않는다.

## Fresh v3: 작은 명령 연속성과 stance 접촉 누락 비교

### 기존 진행도 보상의 문제

401회 강화 정책을 평지·명목 하드웨어에서 별도 평가했다. 전진0.25m/s에0.437m/s, 제자리 회전에 전진0.435m/s로 응답했다. 이 진단은 full-environment 학습의 환경을 바꾼 것이 아니며, 원래 환경의 실패를 평지 성공으로 대체하지 않는다. 실제로 이 평지 진단도 명령 추종에 실패했다.

추가했던 일정 비율의 진행도 항은 작은 command로 속도를 나눈 뒤 포화시키므로, 작은 명령에서도 빠른 이동이 보상의 상당 부분을 얻는다. planar/yaw의0.001 부근 fallback 경계에서 raw 보상이 각각 약0.491/0.435 급변했다. 목표점이 전역 최대라는 이전 검사는 통과했지만, 작은 명령의 연속성과 실제 명령 일반화 검사가 부족했다.

수정은 `mix=0.5*min(command 크기/sigma,1)`이다. planar는 command norm, yaw는 절댓값을 사용한다.0에서 Gaussian으로 연결하고, 작은 명령의 진행도 기여를 줄인다. 큰 명령의 탐색 유인은 유지하며 command range, sampler, clips, 물리·PPO·관측·reset을 바꾸지 않는다. [실제 reward 함수의 수치 검사](smooth_tracking_probe.json)는 양·음 명령, 목표점 최대값,0 근처 연속성과 기존 경계 급변 제거를 확인했다. 새 급변은 약0.000006 이하였다.

### 실제 실행한 fresh pair

| Preset | Run | 시작 | 차이 |
|---|---|---|---|
| progress_gain2_proxy_smooth_yaw4 | gait_reward_smooth_yaw4_fresh_s7_v3 | checkpoint 없이0 | smooth progress + planar/yaw weight4/4, 기존 proxy swing-contact cost |
| progress_gain2_proxy_smooth_yaw4_contactmatch | gait_reward_smooth_yaw4_contactmatch_fresh_s7_v3 | checkpoint 없이0 | 동일 설정에서 missing stance contact cost만 추가 |

두 run은6144환경,seed7,최대600회다.799관측/18action/20DOF, 원래 rough terrain·전체 DR·reset·EE·command schedule·PPO·2Hz 위상을 그대로 사용한다. 높이 보상은 두 run 모두 grid/대칭4cm다. 이전 ground_floor 그룹은301회 넘어짐72회로 control47회보다 나빴으므로 채택하지 않았다. 준비된 smooth_ground_floor preset은 실행하지 않았다.

기존 presence cost는 `mean((1-desired)*presence)`다. contactmatch는 여기에 `mean(desired*(1-presence))`를 더한다. 일치하는 대각선2발은0, 필요한 stance 발 하나가 빠지면0.25, 전부 떠 있으면0.5, moving 중 네 발이 모두 붙으면0.5, 반대 위상으로 접촉하면1이다. 기존 weight-2와 metric의 음수 부호는 유지한다. 정지에서는 네 발 접촉 시0이다. 이는 정상반력 N을 복원하는 항이 아니다.

기존 bit는 이전 정책에서 높은 precision을 보였지만, 새 reward로 정책이 달라진 뒤에도 실제 메시 접촉과 일치하는지 재확인한다. bit 기반 reward/위상 일치만 좋아졌다고 성공으로 판정하지 않는다.

[초기화 비교](matched_fresh_v3.json): 두 run의 checkpoint 인자는None,initial iteration0이다. model0은 첫 history-only 갱신 뒤 저장되므로 history 외33개 텐서를 비교했으며 기존 scratch control과 정확히 같았다. 새 pair의 config 차이는 preset 이름과 symmetric_contact flag뿐이다. PPO/asset/terrain/환경 원점도 같다. 물리 초기 DOF hash는 GPU 비결정성으로 달라질 수 있다.

실행 시점 source52개는 각 runtime snapshot의 SHA256과 일치하는 버전으로 `source_snapshots/`에 보관했다. 이전 기본안503회, yaw4 분기433회, ground_floor 분기332회 상태를 저장하고 협조 종료했다. 보존된301/401 등 milestone 평가는 계속하며 이들을600회 완료 실험으로 표시하지 않는다.

### 평가 기록 보완

- 평지 진단의 첫 저장 시도는 `height_samples=None` 처리가 빠져 실패했다. guard 수정 후 같은401회 체크포인트로 재실행한 `progress401_nominal_diagnostic/policy_evaluation.json`을 사용한다. 첫 실패 로그는 보존했다. 학습과 기존 rough 평가는 영향이 없었다.
- 이전 geometry stride5는2Hz/50Hz의25-step 주기에서5개 위상만 표본화한다. 조건별 접촉 비율을 full-phase 값으로 해석하지 않는다. 원본은 `geometry_stride5_archive/`에 보관하고 stride4로 다시 검사한다. 선택 후보도 stride4로 실제 발 들림·접촉·접선 미끄러짐을 검사한다.
- actual triangle/box와 발 구의 근접은 정상반력 N이나 static support의 보증이 아니다. root 자세·실제 이동·종료·영상과 함께 판단한다.

## Fresh v4: 발 높이 기준만 분리

201회 실제 geometry에서 D/E 모두 앞발P95 약7~10cm, 뒷발은약1~2cm였고 contactmatch만으로 위상별 발 들림의 차이가 줄지 않았다. 보수적3이웃 min height grid는 base 높이를 보수적으로 추정할 때 쓰는 근사이며, 실제 접촉 발 높이보다 낮을 수 있다. 이를 발의4cm swing 기준으로 그대로 쓰면 일부 지형에서 실제 발 들림 목표가 작아지거나 지면 아래에 놓일 수 있다.

`progress_gain2_proxy_smooth_yaw4_footground` / `gait_reward_smooth_yaw4_footground_fresh_s7_v4`를 checkpoint없이0에서 시작했다. D와의 차이는 **foot_height_reference=proxy** 하나다. 기존 접촉 bit가 참일 때의 발 중심Z를 보관하고 그 기준에서Z방향4cm swing을 평가한다. 센서 bit는 기존 관측값이며 정상반력 N을 뜻하지 않는다.

이 실험은 앞서 실패한 ground_floor 그룹과 다르다. **몸통 높이 기준/target은D의grid 방식 그대로이며,4cm 대칭 높이 손실도 그대로**다. stance contactmatch를 추가하지 않는다. `_foot_ground_height(mode=...)`로 발 기준과 몸통 기준을 분리했으며, CPU contract에서 발 목표의 물리적 의미가 복원되고 동일 상태의 몸통 높이 reward는 정확히 같은지 확인했다. plane의 보상도 유지한다.

물리 reset·URDF·PD·관측·PPO·command·EE·terrain·DR는 바꾸지 않는다. 온라인 메시/IK 유효성 검사나 triangle raycast를 추가하지 않는다. read-only 발 높이 cache를 기존 control 주기에서 vectorized 갱신하는 reward 입력 변경이다. 사용 소스는 launch 전에 `source_snapshots/gait_reward_smooth_yaw4_footground_fresh_s7_v4/PRELAUNCH_SHA256.json`과 함께 보관했다.
