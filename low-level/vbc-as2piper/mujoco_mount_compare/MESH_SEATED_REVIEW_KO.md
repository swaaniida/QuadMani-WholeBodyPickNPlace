# AS2 + PiPER + UMI V7 장착부 감사

이번 결과는 **메시에서 떠 있던 장착부를 바로잡은 설계 후보**다. AS2의 실제 하중 지지 프레임·체결면은 제공된 단일 몸통 STL만으로 식별할 수 없으므로, 하드웨어 장착 구조가 검증되었다고 해석하면 안 된다. 가짜 받침대나 볼트 형상을 추가하지 않았다.

## 원인과 좌표 체인

- 원본 `as2piper_umi_actuated_v7.urdf`: `base_link → as2_to_piper (0,0,0.150), R=I → piper_base_link`. 장착판/프로파일 링크가 없다.
- 기존 MuJoCo 생성기: 이 조인트를 건너뛰고, 자체 설정 `x=0.050, z=0.135`에서 판/프로파일을 추가한다. URDF와 MJCF의 장착 변환이 서로 달랐다.
- AS2 visual origin은 `(0,0,0), rpy=(0,0,0)`, mesh scale은 생략되어 1이다. PiPER base도 같고 DAE 단위는 meter=1이다. MuJoCo는 대응하는 원본 STL을 사용한다.
- 몸통 중앙 `(x,y)=(0,0)`의 수직선과 **삼각형 면**의 최상부 교점은 `z=0.076003201m`다. 중앙에 정점이 없다고 상부 면이 없는 것은 아니다. 정점만 골라 만든 중앙 AABB는 이 큰 삼각형들을 놓쳤다.
- 기존 135mm 기준은 중앙면보다 **58.996799mm 위**, 원본 PiPER 150mm 오프셋은 중앙면보다 **73.996799mm 위**다. global max Z=134.8mm는 중앙 장착면이 아니다.
- 기존 판 전체 footprint에서 직접 장착판의 최소 간격은 **28.610mm**, 프로파일 바닥의 최소 간격은 좌우 **20.120/20.163mm**다. 이는 중앙 간격과 별개의 값이다. 전방 돌출부 때문에 위치별 간격이 크게 변한다.
- `x≈0.06m`부터 상부가 솟아오른다. 기존 150mm 판을 x=+0.05m에 둔 채 높이만 76mm로 낮추면 앞쪽 형상을 관통한다.

새 구성은 동일한 몸통 평탄 영역을 사용하는 후보로, 장착 중심 x=-0.020m, y=0m, 판 140×120×6mm, 프로파일 140×20×20mm(중심 y=±35mm)를 사용한다. 이 치수는 제조사 승인 치수가 아니다. PiPER base STL footprint는 판 안에 들어간다.

```text
world → free AS2 base_link
      → as2_assumed_mount_interface: xyz=(-0.020, 0, 0.076028478218), R=I
         → al_profile_left/right (profiles_plate만): y=±0.035, z=0.010
         → mount_plate: z=profile_height + 0.003
            → piper_base_link: z=0.003, R=I
               → 원본 piper_joint1..6 → flange → gripper base
                  → 원본 prismatic 손가락 2개 → PLA/TPU visual/collision
                  → piper_umi_nominal_tcp: gripper base의 z=0.183851719307
```

장착 기준 높이는 판 footprint `x=[-0.09,0.05], y=[-0.06,0.06]`에 걸리는 원본 삼각형을 XY 사각형으로 잘라 얻은 국소 최상부로 계산한다. 중앙 교점보다 25.277μm 높은 미세 형상까지 포함해 관통을 피했다. 로봇 전체의 최대 Z를 사용하지 않았다. 양쪽 프로파일과 판은 같은 고정 기준면에 연결되고, 프로파일 상면과 판 하면은 일치한다. 실제 나사/프레임 체결은 모델링하지 않았다.

## 접합 검사

| 검사 | profiles_plate | plate_only |
|---|---:|---:|
| PiPER base의 AS2 기준 z | 102.028478mm | 82.028478mm |
| AS2 메시↔좌우 프로파일 하면 간격 | 0.016454~0.028404mm | 해당 없음 |
| 프로파일 상면↔판 하면 | 0mm | 해당 없음 |
| AS2 메시↔판 하면 최소 간격 | 20mm (프로파일 공간) | 0mm |
| 판 footprint 내 몸통 메시 관통 | 0mm | 0mm |
| 판 상면↔PiPER STL 하면 | 수치 오차 수준의 0mm | 수치 오차 수준의 0mm |
| 판 질량 | 0.27216kg | 0.27216kg |
| 프로파일 추가 질량 | 0.14000kg | 0kg |
| 총 로봇 질량 | 22.867724kg | 22.727724kg |

직접 장착판 footprint의 19×19 수직선 표본 중 98.892%가 몸통 상부와 1mm 이내다. 곡면 가장자리에는 최대 2.108mm 틈이 남는다. 따라서 전체 판 하면이 완전한 평면 접촉을 한다는 주장은 하지 않는다. 프로파일의 각 footprint 표본은 100%가 1mm 이내다. 관통량은 표본이 아닌 **사각형으로 잘린 모든 원본 삼각형의 극값**으로 계산했다. 이 결과는 삼각형 표면 검사이며 볼트 체결/강도 검사와 다르다.

PiPER의 판 접합은 STL 하면 z≈0, URDF visual origin, 실제 컴파일된 MJCF body 변환을 각각 검사했다. 모든 질량 있는 body의 주관성모멘트 양수와 삼각 부등식을 검사했다. 프로파일 관성은 후보 질량과 외곽 box를 이용한 근사이며 실제 중공 단면 CAD 관성이 아니다.

## 보존과 모델 범위

- 원본 V7 actuated/open URDF, UMI 메시, 기존 MuJoCo 생성기·실행기·설정·두 XML을 보존했다. `audit/before_sha256.json`과 자동 검사에서 SHA-256 일치를 확인한다. 변경 전 파일은 `audit/before/`에 보관한다.
- 새 URDF는 `as2piper_umi_v7_mesh_seated_profiles_plate.urdf`, `as2piper_umi_v7_mesh_seated_plate_only.urdf`다. 원본의 모든 link와 기존 장착 조인트 외 모든 joint가 바이트 직렬화 기준으로 동일함을 검사한다.
- 새 MJCF는 `models/mesh_seated/`에만 생성한다. 기존 XML을 덮어쓰지 않는다. 기존 생성기를 실행해도 새 모델은 바뀌지 않는다.
- 두 모델 모두 free base 포함 nq=27, nv=26, nu=20. nbody는 34/32이며, 질량 없는 명시적 interface body가 하나 추가됐다. 실제 액추에이터 순서·qpos 주소·WBC 18개 조인트 목록은 `geometry_report.json`에 저장한다.
- 그리퍼는 V7의 20DOF 중 마지막 2 prismatic joint로 유지한다. 별도 PD가 +0.05/-0.05m 열림을 제어한다. WBC 정책 코드와 18-action 설계는 변경하지 않았다.
- V7 기존 검사 통과: 열림 외폭 134.5mm, 높이 중심 차이 약 0, 닫힘 안쪽 AABB 여유 2.061mm. 추가 Isaac Gym 로딩/학습은 실행하지 않았다.
- 두 후보 사이에는 프로파일 존재·높이·질량/관성과 이에 따른 PiPER 높이만 다르다. PiPER orientation, 제어 조건, 초기 자세는 동일하다. 팔 초기값 `[0,1.2,-1.4,0,0.2,0]`을 유지했다. 검증된 보행 stow 자세라는 뜻은 아니다.

## 렌더링·동역학

`results/mesh_seated/compare_{front,side,top,mount_close}.png`는 변경 전후·두 후보를 동일한 카메라로 비교한다. 개별 PNG도 함께 저장했다. 접합 검증 후 정면·측면·상면·장착부 확대를 생성했다.

stand/crouch/arm은 각 5초의 중력·지면 접촉·PD 데모다. 학습된 보행, 제어기 성능 비교, 하드웨어 안전성 인증이 아니다. 두 구성에 같은 gains/targets와 2ms time step을 사용했다. 각 CSV에는 CoM, base 위치/자세, 발별 지면 normal force와 접촉 수, 액추에이터별 command 및 joint qpos, nominal TCP와 **실제로 변환한 TPU 메시 최저점**, 자기접촉 수/최소 거리, MuJoCo warning 수가 기록된다. TPU 최저점은 지면 도달성의 필요한 검사이며 접촉면 방향·하중을 갖춘 grasp/dragging 가능성을 보증하지 않는다.

수치 결과는 `results/mesh_seated/dynamics_table.md`, 세부 결과는 `dynamics_summary.json`, 영상은 `{variant}_{stand,crouch,arm}.mp4`다. 영상은 OSMesa 소프트웨어 렌더링이며 인코더 정렬 때문에 640×368로 저장된다. 수치 로그는 렌더링과 동일한 결정적 입력으로 최종 버전에서 재실행했다.

## 미해결된 실제 기구 조건

1. **AS2 상부 메시가 실제 하중 지지 프레임인지 외장 커버인지 확인되지 않았다.** 현 후보는 상부 메시 면에 강체 고정한 가정이며, 제조 가능한 프레임→장착물의 하중 경로가 검증된 것은 아니다. 외장 커버를 구조 프레임으로 재명명하거나 보이지 않는 가짜 지지를 추가하지 않았다.
2. 실제 AS2 프레임/레일 높이, 홀 위치/간격, 나사 규격·체결 깊이, 커버 제거/관통 필요 여부의 CAD 또는 현물 측정이 필요하다. 실제 프레임면이 다르면 interface와 판을 다시 설계해야 한다.
3. PiPER 해당 기종의 베이스 체결 홀·볼트 패턴 및 배선/커넥터 간섭을 확인해야 한다. 70×70mm M5×4 참고값은 이번에 확정하지 않았다.
4. 후보 판/프로파일에는 실제 볼트 구멍, 볼트 질량, 중공 단면, 체결부 유연성/강도 모델이 없다. URDF fixed joints는 강체 체결의 가정이며 볼트 강도를 검사하지 않는다.
5. 기존 로봇 collision은 AS2 단순 primitive 및 PiPER/UMI mesh다. MuJoCo mesh collision은 convex hull 근사다. 가까운 고정 body끼리의 접촉이 생략되는 특성 때문에 자기접촉 0만으로 조립 무관통을 판단하지 않고, 장착부는 원본 삼각형을 따로 검사했다.
6. 현재 팔 자세의 crouch는 TPU를 지면에 닿게 하지 않는다. ground reach 필수 조건은 **미충족**이다. 별도의 충돌·한계·균형을 고려한 reach 자세/제어 실험이 필요하다. walking, EE tracking, dragging 외력은 이번 stand/crouch/arm 데모에서 검증하지 않았다.

## 재실행과 diff

현재 기본 Python에 MuJoCo가 없어 프로젝트 내부 `.deps/`에만 MuJoCo 3.1.6과 의존성을 한 번 설치했다. 다른 Python 환경, 시스템 패키지, B1+Z1 학습을 변경하지 않았다.

```bash
cd /data/team/users/suhwan/vbc-as2piper
bash mujoco_mount_compare/run_mesh_seated_checks.sh
```

핵심 코드: `build_verified_mounts.py`(새 URDF 및 직접 변환), `verify_mounts.py`(삼각형 접합/보존/관성/MJCF/렌더 검사), `run_verified_demo.py`(별도 PD 및 상세 계측), `summarize_verified.py`(6개 결과 요약).

초기 Git 상태에서 AS2 자산·tools·MuJoCo 디렉터리 전체가 untracked였다. 따라서 일반 `git diff --stat`에는 이 작업이 나타나지 않는다. `audit/changes.patch`에 원본 대비 두 URDF/두 MJCF/설정 차이와 새 코드의 전체 diff를 저장했고, `audit/change_stat.txt`에 해당 patch의 파일별 통계를 저장한다. 기존 B1+Z1 경로에 대한 쓰기·설치·프로세스 제어는 수행하지 않았다.

## 최종 동역학 수치

| Variant | Mode | Pitch RMS (deg) | Four feet >1N after 1s | TPU min / final z (m) | Self contacts max |
|---|---|---:|---:|---:|---:|
| profiles_plate | stand | 0.188 | 100.0% | 0.693 / 0.752 | 0 |
| profiles_plate | crouch | 0.271 | 100.0% | 0.670 / 0.671 | 0 |
| profiles_plate | arm | 0.172 | 100.0% | 0.693 / 0.748 | 0 |
| plate_only | stand | 0.206 | 100.0% | 0.674 / 0.732 | 0 |
| plate_only | crouch | 0.289 | 100.0% | 0.650 / 0.651 | 0 |
| plate_only | arm | 0.189 | 100.0% | 0.674 / 0.728 | 0 |

6개 실행에서 1초 이후 네 발이 모두 1N 이상 지지한 표본 비율은 100%, MuJoCo warning과 보고된 자기접촉은 0이었다. 액추에이터 명령 포화는 없었다. arm 실행에서 열린 그리퍼의 소프트 한계를 최대 0.0403mm 초과했으며, 다리/팔 회전 관절 한계 초과는 없었다. 이는 해당 5초 PD 데모 범위의 결과다.

영상 6개는 모두 디코딩 확인했고 각각 126프레임, 24fps, 640×368이다. 기존 정수 렌더 stride 때문에 영상 길이는 5.25초이며 시뮬레이션 시간은 5초다. 로봇 전체가 프레임 안에 보이는 것을 마지막 프레임 모음에서도 확인했다.

B1+Z1 저장소의 작업 전후 tracked status가 동일함을 확인했다. 개발 저장소의 일반 `git diff --stat`는 빈 결과이며, 관련 디렉터리가 처음부터 untracked였기 때문이다.
