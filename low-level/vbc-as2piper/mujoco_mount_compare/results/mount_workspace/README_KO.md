# AS2 PiPER 장착 전진·작업공간·바닥 도달 비교

2026-10-28. 현재 실행 중인 학습 및 B1 baseline은 수정하지 않았다. 분석용 URDF/MJCF 복사본만 생성했다.

## 학습 로그 해석

`training_log_snapshot.txt`는 iteration 2079/45000의 완성된 stdout 블록이다. 당시 mean episode length=151.25 step=**3.025초**, dones=0.01, surrogate=0.1248, collection=3.842초, learning=0.313초, total FPS=35,485였다. 초기 약 20 step보다 늘었지만 최대 10초보다 짧다. dones는 소수 둘째 자리로 출력되므로 정확한 1.000%라는 뜻은 아니다.

| 지표 | 이 저장소 구현에서의 의미 | 해석 주의점 |
|---|---|---|
| Train/dones | 매 제어 step에서 done인 환경 수/6144를 최근 100 step 평균 | 실패율/성공률이 아니다. timeout도 포함한다. 0.01이면 대략 step마다 61개 환경 종료. 정상상태에서는 평균 episode 길이의 역수와 가까워지지만 서로 다른 평균창과 반올림 때문에 정확히 일치하지 않는다. |
| Train/mean_episode_length | 최근 완료된 100 episode 길이의 평균, 단위 control step | dt=0.02초. 최대 episode 10초. 학습 시작 직후 randomized episode counters 때문에 일부 초기 timeout은 별도 해석해야 한다. |
| Perf/total_fps | 6144환경 × rollout 24step ÷ (collection+learning wall time) | 전체 병렬 transition 처리량. 단일 로봇 제어 주파수나 렌더 FPS가 아니다. 로그 시점의 출력/저장 비용을 모두 포함한 end-to-end 처리량도 아니다. |
| Perf/collection time | 한 iteration의 rollout 24step 수집 wall time | policy inference, env.step/물리/관측/보상, buffer 저장, 일부 CPU 통계·동기화 포함. 순수 physics 시간만이 아니다. 가상시간으로는 각 환경 0.48초를 수집한다. |
| Loss/surrogate | PPO의 clipped policy objective를 최소화하도록 부호를 바꾼 loss | `mean(max(-A*r,-A*clip(r,1-eps,1+eps)))`. 보행/팔 advantage가 혼합된다. 양수·음수·0 근접 자체가 성공 또는 수렴의 판정 기준은 아니다. |
| Episode_metric/* | 종료된 환경에서 `sum(raw metric)/고정 최대 episode 초(10)`를 계산한 후 runner에서 다시 평균 | 실제 episode 길이로 나눈 per-step 평균이 아니다. episode가 길어지기만 해도 값이 커질 수 있다. reward weight 적용 전 metric이며 물리적 의미는 각 함수마다 다르다. |
| Episode_rew/* | 종료 episode의 가중 reward 합/10초 | 총 `Train/mean_reward`와 집계/스케일이 다르다. 다리 총 reward는 환경에서 추가로 /100 처리된다. |

`metric_alive`는 raw=1이므로 대략 N/10이다. 따라서 같은 집계로 계산한 `metric_X/metric_alive`는 step 평균을 가늠하는 데 유용하지만, runner가 step별 종료 집단을 같은 비중으로 평균하므로 엄밀한 전체 transition 평균은 아니다. 로그의 151.25 step과 `10*metric_alive`가 다른 이유도 집계창/가중치 차이다.

구체적으로 **metric_tracking_ee_world는 EE L1 위치 오차(m)**이고 reward는 `exp(-2*L1/sigma)`다. snapshot의 `4.9914/13.7236 ≈ 0.364 m`는 대략적인 step 평균 L1 오차로, 성공률이 아니다. `metric_base_height`는 높이 자체이며 `4.9455/13.7236 ≈ 0.360 m`. `metric_tracking_lin_vel_max`는 전후진 명령 방향 속도를 명령 크기로 나눈 점수(위쪽 1로 제한, 반대방향 속도면 음수)이며, 작은 x 명령에서는 `exp(-abs(vx))`이다. snapshot의 `3.1898/13.7236≈0.232`는 대략적인 평균 점수다. `walking_dof`는 기본 다리 자세와의 편차에 대한 지수 점수로, 걷기 성공률이 아니다.

현재 종료 조건: |roll| 또는 |pitch|>0.8rad(45.8°), base world z<0.1m, timeout, 지정 termination link 접촉. 현재 AS2 termination link 목록은 비어 있다. **EE 추종 실패 직접 reset 조건은 주석 처리되어 있다.** 따라서 “EE를 못 따라가서 매번 직접 reset”은 틀린 설명이다. 팔 목표/착지/제어 불일치가 자세를 무너뜨려 간접 reset을 만들 가능성은 있으나 원인별 카운터가 없어 비율은 확정하지 못한다. 현재 run을 변경하지 않았다.

소스: `low-level-as2piper/rsl_rl_local/rsl_rl/runners/on_policy_runner.py:118–233`, `.../algorithms/ppo.py:198–215`, `low-level-as2piper/legged_gym/envs/manip_loco/manip_loco.py:170–212,259–279,329–337`, `.../rewards/maniploco_rewards.py:17–20,180–185`.

## 좌표계

원본 B1은 arm mount offset `[0.3,0,0.09]`를 이용하지만 목표 구면 중심은 **body XY + yaw 회전한 `[0.3,0,0.7]`**이다. z=0.7은 terrain 기준의 고정 world 높이이며 실제 arm base 높이가 아니다. 관측에서 arm offset도 yaw만 적용한다. 그러므로 기존 B1 목표 생성기를 그대로 “실제 arm-base 6D 좌표계”라고 부를 수 없다.

AS2에서도 실제 arm base를 명시적 기준으로 사용해야 한다. 이번 샘플은 다음 두 좌표로 함께 저장했다.

```
p_world = T_world_body * T_body_arm * p_arm
p_arm = inverse(T_world_arm) * p_world
p_head = inverse(T_world_head) * p_world
```

`arm_frame_proposal.json`에 모델별 body→arm 변환, 초기 실제 TCP의 arm-frame 좌표, 샘플 범위를 저장했다. `*_workspace.npz`의 `arm`/`head`가 변환된 실제 샘플이다. 샘플 min/max 상자를 그대로 명령 범위로 삼으면 내부의 도달 불가능/충돌 영역이 포함되므로, joint-space sampling 또는 IK/충돌 필터를 통과한 목표를 사용해야 한다. PiPER와 Z1의 tool-axis도 다르다(여기서는 PiPER TCP +z, Z1 +x).

전신 tilt 동안 바닥 물체를 고정해서 잡으려면, 목표를 생성할 때 arm 기준→world로 변환한 뒤 world 목표를 유지해야 한다. 매 step마다 같은 local target을 움직이는 body에 다시 붙이면 물체 대신 몸통을 따라가는 목표가 된다. yaw 정렬/중력 정렬 명령을 원한다면 실제 arm frame과 별개의 frame으로 명시해야 한다. 현재 학습의 목표 생성기는 변경하지 않았으며 JSON은 적용 전 제안이다.

head는 CAD 센서 frame이 자료에 없어 **원본 trunk/base visual mesh 최전방 x 단면의 중심**으로 정의했다. AS2 `[0.334705,0,0.124055]`, B1 `[0.489388,0,-0.001875]` (각 body 기준 m). 비교용 nose proxy이며 실측 head frame이 아니다. 모든 head-relative workspace 축은 standing body 축과 평행하다.

## 장착 위치 선정

기존 판은 x=[-90,+50]mm, 140×120×6mm이다. PiPER base mesh의 전방 끝은 arm base x+40mm다. 판 가장자리 여유 **5mm를 설계 가정**으로 두면 arm base는 최대 **x=+5mm**, 현재 -20mm에서 **25mm 전진**할 수 있다. 기존 판에 x=+50/+100/+150mm로 암만 옮기는 것은 베이스가 판 밖으로 나가므로 실제 장착 후보로 채택하지 않았다. 볼트 홀의 위치/가장자리 강도까지 검증한 한계는 아니다.

더 큰 이동을 위해 별도 **프로파일+연장판** 후보를 만들었다. 기존 140mm 프로파일은 유지하고 판을 **189×120×6mm**로 늘려 전방으로 49mm cantilever를 만든다. 올라오는 AS2 전방 외장과 최소 5mm 여유를 요구한 1mm 격자 탐색에서 arm base **x=+54mm**를 선택했다. 현재에서 **74mm 전진**, 외장 간격 **5.227mm**, 판 질량 추가 **95.256g**이다. 가짜 지지대를 추가하지 않았다. 이 후보는 기존 판 두 variant의 공정한 높이/질량 비교와 분리해야 한다.

**추천:** 판을 유지하는 low-level 기구 후보는 `plate_only_forward`(x=+5mm). 작업공간 확장을 위한 CAD 후보는 `profiles_cantilever`(x=+54mm). 이것은 정해진 판/프로파일 높이·5mm 여유 조건 아래의 최대 전진 후보이며, 모든 기구 설계 중 전역 최적점이나 제작 승인안은 아니다. 더 낮은 front-specific bracket/판 재설계는 이번 탐색 밖이다.

## CoM·회전관성·전방 작업공간

모두 grounded neutral legs, AS2 arm C=[0,1.1,-0.55,0,0,0]에서 비교. B1은 원본 arm 자세 사용. 카메라/케이블/파지 물체 질량은 제외. 관성은 **전체 CoM 주위**, standing body 축 기준. body 원점 주위 관성이나 동적 관성 응답과 구별한다.

| 모델 | arm x mm | 질량 kg | CoM x / z mm | Iroll / Ipitch / Iyaw kg·m² | 앞발 총 정적 하중 비율 | head 앞 최대 샘플 TCP x mm |
|---|---:|---:|---:|---:|---:|---:|
| plate current | -20 | 22.728 | 13.05 / 381.97 | .7501 / 1.2284 / .8218 | 52.84% | 441.5 |
| plate forward | +5 | 22.728 | 18.35 / 381.97 | .7501 / 1.2344 / .8278 | 54.09% | 466.5 |
| profiles current | -20 | 22.868 | 12.85 / 386.59 | .7990 / 1.2775 / .8224 | 52.79% | 441.5 |
| profiles forward | +5 | 22.868 | 18.12 / 386.59 | .7990 / 1.2836 / .8285 | 54.03% | 466.5 |
| profiles extended | +54 | 22.963 | 28.63 / 386.74 | .7992 / 1.3096 / .8544 | 56.51% | 515.5 |
| B1 Z1 reference | +300 | 57.987 | 24.40 / 520.83 | 2.3674 / 6.5342 / 6.0756 | 57.24% | 651.8 |

전진은 pitch/yaw 관성을 줄이지 않는다. plate 25mm 전진은 CoM +5.30mm, Ipitch +0.49%, Iyaw +0.73%. 프로파일 자체는 plate 대비 CoM +4.62mm, Iroll +6.52%, Ipitch +4.00%. 연장판 후보는 기존 profiles 대비 CoM 전진 +15.77mm, Ipitch +2.51%, Iyaw +3.90%를 지불한다. 낮은 관성만 우선하면 plate가 유리하고, 가까운 전방 바닥 접근은 전진 배치가 유리하다.

16000 joint-uniform samples/model, 같은 normalized random seed, 관절 한계 안쪽 0.015rad 여유, 자기 충돌 proxy 관통>0.5mm 제거, TCP z>=20mm. orientation은 workspace cloud에서 제약하지 않았다. head 앞 30mm voxel 점유량은 plate current .051624→forward .058077m³(+12.5%), profiles .052542→.058779m³(+11.9%), 연장판 .075573m³, B1 .104085m³. 이는 **표본 voxel 점유량**이며 정확한 연속 workspace 부피/도달 확률/파지 가능한 부피가 아니다. 각 점의 orientation·토크·손목 카메라 가시성을 모두 보장하지 않는다.

## 전방 바닥 도달

앞다리를 접고 후방 hip sagittal pivot 주위로 몸통을 기울인다. 네 발 중심을 world에 고정하고 각 다리 IK 오차<=3mm, joint limits를 적용했다. 목표는 **standing head 앞** +100/+200/+300mm, 바닥 z=0이다. 몸통 tilt와 함께 head도 움직이지만 물체 목표는 고정이다. UMI 실제 TPU mesh와 B1 실제 finger mesh의 distal surface 평균점을 사용했다. 단순 link6 원점 도달 검사가 아니다.

팔 IK 위치 오차<15mm, tool-down 정렬 31.8° 이내, finger mesh 최저 z>=-3mm, collision proxy 관통<=0.5mm, 전체 CoM 지지다각형 margin>30mm. 각 각도에서 IK seed 3개, 각도 0–40°를 2.5° 단위로 탐색했다. 표의 각도는 **발견한 해의 최소 격자 각도**이며 전역 최솟값/해 부재의 증명은 아니다.

| 모델 | head+100mm | head+200mm | head+300mm | +100mm 해의 최대 arm 중력 토크 Nm |
|---|---:|---:|---:|---:|
| plate current | 7.5° | 7.5° | 미발견 | 10.34 |
| plate forward | 5° | 5° | 30° | 10.06 |
| profiles current | 7.5° | 7.5° | 22.5° | 10.29 |
| profiles forward | 5° | 7.5° | 17.5° | 10.02 |
| profiles extended | 0° | 2.5° | 10° | 9.43 |
| B1 Z1 | 10° | 10° | 10° | 9.76 |

이 표는 물체 인식·파지 시연에 쓸 수 있는 **기구학 후보**다. B1보다 AS2가 +100mm에서 각도가 작다는 사실만으로 더 안정적이라는 결론을 내릴 수 없다. B1은 로봇 높이/전방 길이/다리/arm 초기 자세가 다르고 +200/+300mm에서도 비교적 작은 각도로 접근했다. AS2는 현재 중앙 배치에서 앞쪽 먼 바닥 접근이 특히 불리하다. 프로파일 높이는 근거리에서는 불리할 수 있지만 특정 자세에서 머리/관절 제약을 피하여 먼 거리 접근을 돕기도 한다.

추가로 기존 MuJoCo 데모의 **가정된 PiPER 9Nm 제한**을 모든 arm joint에 적용하면 +100mm에서 plate 37.5→27.5°, profiles 35→25°, 연장판 10°로 변했다. +200/+300mm는 조건을 통과한 해를 발견하지 못했다. B1은 80% URDF effort 기준에서 각 10°였다. 이 비교는 토크 한계가 로봇별로 다르므로 동등한 안전등급 비교가 아니다. PiPER URDF effort=100Nm와 데모 9Nm 모두 실제 연속 토크 검증값이 아니므로 어느 한쪽을 제작 승인 근거로 삼으면 안 된다. 실제 연속/peak 토크, 기어/열 한계, 동작 속도, 물체 질량을 확인해야 한다.

정적 토크는 MuJoCo gravity bias와 네 발 비음수 수직 하중 평형으로 계산했다. 마찰원뿔·접촉 힘의 유일해·관절 탄성·가속/충격·dragging 외력은 검증하지 않았다. 초기 9Nm 조건을 통과하는 깊은 tilt는 학습 termination 45.8°에 가깝고 동작 여유가 작다. 정적 지지 margin만으로 이를 안전하다고 부를 수 없다.

## 로드·접합·물리 검사

AS2 다섯 모델 nq=27,nv=26,nu=20 유지. B1 복사본 nq=26,nv=25,nu=19. UMI 관련 link/joint XML이 기존 mesh-seated V7과 동일한지 검사했다. plate↔PiPER z gap≈0, 기존 지지 footprint 유지, 연장판 앞부분 AS2 외장 관통 없음. `interface_validation.json`은 triangle clipping에 의한 최소/샘플 최대 gap, 지지 근접 면적 비율을 별도 기록한다. 모델의 fixed joint가 실제 구조 나사 결합의 검증을 대신하지 않는다.

`standing_dynamics.json`/CSV: MuJoCo 중력·접촉 3초, actuated-joint PD+중력 feedforward, base에 가짜 힘 없음. AS2 다섯 모델 최대 pitch 0.851/0.921/0.853/0.921/1.131°, 최대 roll<0.026°, 마지막 1초에서 네 발의 최소 normal force가 모두 0보다 컸다. B1 복사본은 동일한 단순 PD 틀에서 최대 pitch10.27°로 더 크게 처졌다. 이 controller는 B1 정책의 성능을 재현하거나 로봇별로 동등하게 튜닝한 제어기가 아니므로 이 결과로 AS2가 B1보다 안정적이라 주장하지 않는다. ground-reach 자세와 전환 trajectory는 이번에 동적 검증하지 않았다.

## 그림과 재실행

- `workspace_3d.png`, `workspace_side.png`, `workspace_top.png`: head-relative 샘플 작업공간.
- `mount_tradeoff.png`: 같은 C 자세의 CoM/관성.
- `ground_reach.png`: 기구학 및 추가 토크 제한의 도달 각도.
- `robot_pose_comparison.jpg`: 실제 메시 standing와 head+100mm 최소 기구학 tilt. 행별 카메라 거리는 B1 크기에 맞춰 다르다. 동역학 프레임이 아니다.
- `results.json`, `*_workspace.npz`, `*_reach_trials.json`: 좌표·질량·관성·자세·하중 수치.
- `arm_frame_proposal.json`: 학습 적용 전 좌표계 제안.

```bash
cd /data/team/users/suhwan/vbc-as2piper
umask 0002
export OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
export PYTHONPATH="$PWD/mujoco_mount_compare/.deps:$PWD/mujoco_mount_compare"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1
python mujoco_mount_compare/mount_workspace_study.py
python mujoco_mount_compare/check_workspace_dynamics.py
python mujoco_mount_compare/plot_mount_workspace_study.py
```

원본 V7/학습 URDF/config/reward/PPO/run script는 변경하지 않았다. B1 malformed `0.09>>`는 개발 worktree의 새 `b1z1_workspace_study.urdf` 복사본에서만 정리했다. PiPER DAE 한 파일의 mm 단위를 재해석하는 대신 기존 검증 변환기처럼 대응하는 공식 metre STL을 사용한다. `/data/team/users/suhwan/vbc-legacy`의 코드·학습·환경·프로세스는 변경/종료하지 않았다.

미확정 사항: AS2 상부 외장은 실제 구조 프레임/볼트 기준면으로 확인되지 않음, PiPER bolt pattern/edge distance, cantilever 강도/진동/체결 하중, 실제 모터 제한, 카메라·케이블·물체 질량/간섭, 실제 손목 카메라 시야, 물체 파지와 dragging 하중. 이 자료는 다음 CAD/제어 설계의 수치 비교 자료다.
