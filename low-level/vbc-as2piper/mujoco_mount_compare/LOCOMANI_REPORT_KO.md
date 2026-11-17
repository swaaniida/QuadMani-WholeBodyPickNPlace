# AS2 + PiPER + UMI locomotion/manipulation 준비 자세와 동작 데모

## 결과의 범위

이 결과는 `mesh_seated`의 두 장착 후보를 그대로 사용해 팔 준비 자세, 앞다리 기반 전방 tilt, 손목 카메라 후보 시야, 물체 접근·회수·drag 동작을 비교한 것이다. 장착부 before/after 비교는 반복하지 않았다.

현재 저장소에는 학습된 AS2+PiPER 보행 정책이나 접촉 force-control grasp controller가 없다. 따라서 walking 영상은 **관절 stepping + 제한된 가상 base velocity servo**, grasp는 **손가락 닫힘 이후 최대 22N의 compliant Cartesian attachment**로 모사했다. 영상은 동작 설계와 카메라 구도를 검토하기 위한 것이며 actuator-only 보행, 실제 파지 성공률, 하드웨어 안전성을 증명하지 않는다. CSV에는 가상 외력과 grasp proxy의 힘·오차를 모두 기록한다.

## 연구에서 가져온 원칙

- [QuadWBG](https://arxiv.org/html/2411.06782)는 single arm-mounted camera, base pose reachability, 안정성을 함께 다루며 locomotion으로 팔의 유효 작업공간을 맞춘 뒤 정밀 grasp를 수행한다. 이 구조에서 카메라 관찰, base 접근, arm motion을 독립적으로 보지 않았다.
- [B1+Z1 deployment framework](https://arxiv.org/html/2512.18938)는 coordinated whole-body control이 floating-base arm보다 reach를 늘릴 수 있고, Isaac Gym과 MuJoCo 접촉 차이가 정책 행동에 영향을 준다고 보고한다. 그래서 이번 MuJoCo scripted gait 수치를 학습 정책 성능으로 해석하지 않는다.
- [Nonlinear MPC for non-negligible arm dynamics](https://arxiv.org/html/2507.22042)는 사족보행 몸체에 비해 팔의 질량·관성이 무시할 수 없을 때 arm full dynamics를 locomotion model과 결합한다. AS2 중앙 장착에서도 팔을 길게 전방으로 내민 travel pose를 피하는 근거다.
- [Omni-directional 6D EE tracking](https://arxiv.org/html/2412.03012)은 locomotion, manipulation, 두 상태 사이 transition을 하나의 multi-stage 문제로 다룬다. 이번 pick 시나리오도 몸통 tilt와 팔 하강을 동시에 보간한다.

논문에서 특정 로봇의 관절각을 복사하지 않았다. AS2는 PiPER가 몸통 중앙에 있고, 전방 머리 위 장착 B1/Spot 계열과 베이스–팔 위치 관계가 다르다. 아래 값은 현재 AS2+PiPER URDF와 MuJoCo 질량·관성·관절 한계에서 직접 평가했다.

## 선택한 자세

| 용도 | PiPER q1..q6 (rad) | 설계 의도 |
|---|---|---|
| locomotion/manipulation-ready | `[0, 0.76, -0.89, 0, 0.45, 0]` | 좌우 대칭, 손을 몸통 가까이, 카메라가 그리퍼와 전방 바닥을 함께 관찰 |
| floor grasp 접근 | `[0, 2.3356, -0.9586, 0, -0.8569, 0]` | 앞다리 tilt와 함께 TCP를 바닥 물체로 내림 |
| lifted drag | grasp에서 travel 방향으로 22% 보간 | 물체는 바닥에 둔 채 손목/팔 하중 자세만 조금 회복 |

5초 중력 settle 후 선택한 travel pose는 기존 `[0,1.2,-1.4,0,0.2,0]`보다 다음과 같이 바뀐다.

| Variant | 기존 TCP x/z | 선택 TCP x/z | 기존/선택 전신 CoM x |
|---|---:|---:|---:|
| profiles_plate | 0.473/0.469m | 0.298/0.380m | 27.7/7.9mm |
| plate_only | 0.473/0.449m | 0.299/0.359m | 28.0/8.2mm |

TCP와 CoM 값은 AS2 `base_link` 기준이다. 선택 자세는 손을 약 174mm 몸통 쪽으로 당기고 89mm 낮추며, 전신 CoM의 전방 편향을 약 20mm 줄인다. 팔 자체 CoM은 base 기준 x≈-22mm다. q1/q4/q6을 0으로 두어 중앙 장착의 좌우 대칭을 유지한다.

이 선택에는 trade-off가 있다. TCP position Jacobian singular values는 약 `[0.654, 0.352, 0.199]`, product는 `0.0457`로 기존 자세의 `0.0882`보다 작다. 즉 최대 manipulability posture가 아니라 locomotion 중 compactness와 manipulation 시작 준비를 우선한 자세다. 최솟값 0.199로 이 모델에서 즉시 singular한 자세는 아니다.

## 전방 tilt와 카메라

전방 tilt는 네 다리를 모두 같은 값으로 접지 않는다. 앞다리만 standing에서 다음만큼 변경한다.

```text
front thigh: +0.35 rad
front calf:  -0.70 rad
rear legs:   standing nominal 유지
```

중력 settle 결과 몸통 pitch는 약 20.2~20.3°이고 네 발 하중을 유지한다. 팔 하강은 이 tilt와 동시에 진행한다. tilt walking/pick에서 기록된 최소 CoM support margin은 약 69~100mm다. 이는 현재 평면·마찰·scripted servo 조건의 결과다.

카메라는 gripper base 기준 `(0, -72, -40)mm`에 놓고, 그리퍼 +Z 축에서 바닥 방향으로 20° 기울인 후보다. 수직 FOV는 70°로 가정했다. 16:9 영상에서 standing과 전환 동작 동안 물체 중심과 TCP가 계산상 수평/수직 FOV에 포함됐고, 실제 wrist inset에서도 두 대상이 보이는지 확인했다.

카메라의 실제 CAD, 질량, 렌즈 intrinsic/distortion, near range, cable bend radius는 미정이다. 카메라 body는 동역학 질량에 넣지 않았다. 실제 부품 선택 후 같은 최적화를 다시 해야 한다.

## 생성한 시나리오

각 시나리오는 `profiles_plate`, `plate_only` 두 구성에 대해 생성한다.

1. `standing`: 선택한 locomotion-ready arm pose.
2. `tilt_in_place`: 정지 상태에서 앞다리만 굽혀 전방 tilt.
3. `stand_walk_omni`: upright에서 forward, left, backward, right, yaw-left 순서.
4. `tilt_walk_omni`: 약 20° tilt를 유지하며 같은 방향 명령.
5. `pick_recover`: standing에서 tilt와 팔 하강을 동시에 수행하고, 손가락을 닫아 물체를 잡은 뒤 standing/travel pose로 복귀.
6. `pick_drag`: grasp 자세와 tilt를 유지하며 물체를 바닥에서 여러 방향으로 drag.
7. `pick_lift_drag`: grasp 후 arm만 22% 들어 하중 자세를 완화하고 물체는 바닥 접촉을 유지한 채 drag.

모든 영상은 640×360, 12fps이며 오른쪽 위에 320×180 wrist-camera inset과 현재 방향 명령을 표시한다. 외부 카메라는 움직이는 base를 따라간다.

## 검증 결과와 한계

- 14개 시뮬레이션에서 MuJoCo warning은 0이었다.
- 네 발 normal force가 모두 1N 이상인 표본 비율은 99.7~99.9%였다.
- roll 최대는 약 2.31°, pitch 최대는 약 22.3°였다.
- upright walking의 x/y 이동 범위는 약 0.10/0.07m, tilted walking은 약 0.16/0.07m다. 이는 가상 base servo가 만든 경로 범위다.
- wrist camera의 물체 중심/TCP 수직 FOV 포함률은 기록 표본에서 모두 100%였다. 중심점 FOV 검사는 전체 물체 segmentation 품질이나 motion blur를 보증하지 않는다.
- compliant grasp의 최대 TCP–object 중심 오차는 약 44mm다. 물체 반경 35mm를 포함한 거친 grasp proxy이고 finger contact grasp가 아니다.
- `pick_recover`는 물체를 약 0.61~0.63m까지 들어 올렸다. drag 계열은 물체의 바닥 접촉을 유지했다.
- 접근 중 AS2의 단순 base collision sphere와 PiPER link4/link5 사이에 최대 약 0.58mm의 transient soft-contact penetration이 있었다. 1mm 이하이지만 실제 CAD 간섭 없음으로 승인할 수 없다. 원본 AS2 collision primitive가 visual mesh보다 거칠다는 점도 함께 고려해야 한다.
- `pick_recover`에서 piper joint2의 9Nm command limit 포화가 약 10~11% 발생했다. 동작은 완료됐지만 실제 하드웨어용 복귀 궤적은 torque-aware trajectory optimization이 필요하다.
- 지면은 평면, 물체는 180g sphere, grasp proxy 한계는 22N이다. 다른 물체 질량·마찰·형상, 카메라 질량, 케이블, drag 반력은 아직 검증하지 않았다.

세부 수치는 `results/locomani/summary.json`, 자세 비교는 `pose_evaluation.json`, 각 프레임 로그는 개별 CSV에 있다.

## 재실행

```bash
cd /data/team/users/suhwan/vbc-as2piper
bash mujoco_mount_compare/run_locomani_suite.sh
```

학습된 18-action WBC를 연결할 때에는 scripted stepping과 virtual base servo를 제거하고, 같은 arm/tilt/EE trajectory를 command로 사용해야 한다. 그 다음 camera mass randomization, object pose distribution, arm payload, external drag wrench, foot friction을 학습·평가 조건에 포함한다.
