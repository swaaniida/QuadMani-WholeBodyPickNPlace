# AGENTS.md — AS2 + PiPER + UMI 프로젝트 작업 지침

> **대상:** AstraGo 서버의 `/data/team/users/suhwan/vbc-as2piper`에서 실행하는 Codex CLI.
> **현재 상태:** AS2 + PiPER + UMI 그리퍼의 20-DOF 모델은 로딩되지만, **AS2 상판–장착판–PiPER 베이스의 물리적 접합 위치가 검증되지 않았고, 현재 MuJoCo 영상에서 장착판이 몸통 위에 떠 있는 문제**가 있다. 단순 카메라/팔 자세 문제가 아니다.
> 이 파일은 이전 채팅에서 확인한 작업 맥락을 전달한다. **파일 내용·경로·실제 형상은 서버에서 직접 검사한 결과를 우선**하며, 아래의 수치 중 '후보/가정'으로 명시된 값은 확정 치수가 아니다.

## 1. 최우선 목표

1. **AS2–PiPER 장착 구조를 URDF부터 올바르게 수정**한다. AS2의 실제 상부 장착면과 PiPER 베이스/어댑터의 체결 기준을 확인한다.
2. 다음 **두 가지 실제로 연결된 기구 구조**를 만들고 MuJoCo에서 각각 검증한다.
   - **A: profiles_plate** — AS2 구조 프레임 → 좌우 알루미늄 프로파일 → 어댑터 플레이트 → PiPER.
   - **B: plate_only** — AS2의 하중 지지 구조에 어댑터 플레이트 직접 장착 → PiPER.
3. 시각적으로만 붙이는 것이 아니라 **접합 간격·메시 관통·질량·관성·충돌·실제 지지 경로**를 검사한다. 접합면 치수가 불확실하면 가정과 검증 필요 항목을 명시한다.
4. 두 구성에서 동일한 조건으로 **standing / crouch / arm motion**을 실행하고, 로봇 전체가 보이는 MP4 및 접합부 확대 사진을 생성한다.
5. 후속 실험: walking 시 안정성, crouch 시 UMI 그리퍼의 바닥 도달성, EE tracking, dragging 외력. **지금 제공된 데모 PD/scripted motion은 학습된 보행 정책이 아니다.**

### 가장 중요한 교훈

과거 수정에서 초기 팔 관절각과 카메라를 고치는 데 시간을 썼지만, **원본 통합 URDF의 잘못된 장착 오프셋과 MuJoCo 생성기의 임의 상판 높이**를 먼저 고치지 않아 장착판이 뜬 문제가 남았다. **이 오류를 포즈/렌더링으로 가리지 말 것.**

## 2. 절대로 건드리지 말아야 할 것

- **기존 Baseline:** `/data/team/users/suhwan/vbc-legacy` — B1+Z1 PPO 및 W&B 실험용. **수정/삭제/리셋/종료/재설치 금지.** 읽기 전용 참고만 허용.
- **개발 worktree:** `/data/team/users/suhwan/vbc-as2piper` (Git branch `feat/as2-piper-dev`). **AS2+PiPER 수정은 오직 이 디렉터리에서.** `.git` 파일이 `vbc-legacy` 쪽 Git 메타데이터를 참조하는 linked worktree임을 인지할 것.
- 공유 AstraGo 워크로드, 다른 사용자 프로세스, 기존 GPU 학습, W&B 실행을 중지하지 말 것. `pkill`, `killall`, `git reset --hard`, `git clean -fdx`, 대량 `rm -rf`, root/시스템 설정 변경은 명시적 허가 전에는 실행하지 말 것.
- `/data/team`은 공유 NFS. **인증 파일/토큰/비밀번호/SSH 개인키/`auth.json` 읽거나 출력·커밋하지 말 것.**
- 개인 Codex 홈 `/data/team/users/suhwan/codex-home` (Linux 사용자 `suhwan`, UID 1101, 접근 제한)은 비공개. SSH 공용 사용자 `vbc`와 분리되어 있다. **이미 로그인 및 프로젝트 쓰기 권한 문제는 해결됨; 다시 사용자/권한 설정하지 말 것.**
- Codex가 이 개발 worktree 안에 만드는 파일은 공용 `vbc` 그룹에서 읽고 쓸 수 있어야 한다. 디렉터리는 `2775`, 일반 파일은 `664`, 실행 스크립트는 `775`를 사용하고, 생성 스크립트는 `umask 0002`로 실행한다. 작업 후 `suhwan:vbc` 소유 산출물에 그룹 읽기/디렉터리 진입 권한이 빠지지 않았는지 확인한다. 이 공유 정책은 비공개 `/data/team/users/suhwan/codex-home`에는 적용하지 않는다.
- 원본 UMI V7 그리퍼 형상을 임의 회전·수정하지 말 것. 필요하면 복사본에서 작업하고 변경 전/후 기하학 검사를 수행할 것.
- 코드를 수정하기 전에 `git status --short` 및 변경 대상 확인. 유의미한 변경은 diff를 제공하고, 기존 성공 버전의 복구 수단을 확보한다.

## 3. 현재 개발 환경과 저장소

- 컨테이너: Ubuntu 20.04 기반, 기본 Python 3.8. MuJoCo **3.1.6** 로딩 성공. 독립적인 Python 환경은 사용자별로 다를 수 있으므로 확인할 것.
- GPU: NVIDIA A100 MIG 3g.40GB — CUDA 물리/학습에 사용. 이 MIG에서 EGL 그래픽 렌더링은 동작하지 않았음. **MuJoCo 오프스크린 영상은 OSMesa 소프트웨어 렌더링으로 성공**했음.
- 렌더링 예:
  ```bash
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 \
  python mujoco_mount_compare/run_mount_demo.py \
    --variant plate_only --mode arm --seconds 5 --video
  ```
  기존 `mujoco.Renderer(...,height=720,width=1280)`는 MJCF 기본 프레임버퍼 640×480을 넘어서 에러가 났다. **640×360으로 변경**하거나 XML `<visual><global offwidth="1280" offheight="720"/></visual>` 설정 필요.
- Root project:
  `/data/team/users/suhwan/vbc-as2piper`
- 주요 리소스(실제 파일 확인 필수):
  ```text
  low-level/resources/robots/as2piper/
    urdf/as2piper.urdf
    urdf/as2piper_umi_actuated_v7.urdf      # 보존 대상: UMI 구동 모델
    urdf/as2piper_umi_open_pose_v7.urdf     # 보존 대상: 시각화용 열림 상태
    meshes/as2/
    meshes/piper/
    meshes/umi/umi_pla.stl
    meshes/umi/umi_tpu.stl
  tools/
    build_as2piper.py
    finish_umi_gripper_v6.py
    finish_umi_gripper_v7.py
    check_umi_gripper_v7.py
    umi_v7_fit_report.json
  mujoco_mount_compare/
    build_mount_scenes.py
    run_mount_demo.py
    compare_results.py
    mount_config.json
    models/profiles_plate.xml
    models/plate_only.xml
    results/
  ```
  파일명/버전은 현장 `find`, `git status`로 재검증. 이전 테스트용 V1~V6 중간 URDF가 정리·아카이브되었을 수 있음.
- 소스 CAD/URDF 자료:
  ```text
  /data/team/users/suhwan/robot-sources/unitree_ros/robots/as2_description/
  /data/team/users/suhwan/robot-sources/agx_arm_urdf/
  ```
- Isaac Gym의 라이브러리는 구형 환경에서 `vbc-legacy/third_party/isaacgym`로 설치/참조되었으므로 새 사용자 `suhwan` 환경에서는 import 가능 여부부터 확인. 설치/업그레이드 과정에서 Baseline 손상 금지.

## 4. AS2 + PiPER + UMI 모델에서 검증된 사실

### 4.1 Kinematics / DOFs

- AS2 다리 12 DOF + PiPER arm 6 DOF + 그리퍼 finger prismatic joint 2 DOF = **20 actuated joints**.
- 정책의 **WBC action 차원은 18(다리12+암6)**을 유지. 그리퍼 2개는 정책 출력에 포함하지 않고 별도의 위치 제어로 **열림 유지**.
- Isaac Gym에서 V7 계열 20 DOF 로딩이 된 이력. 이전 출력의 DOF 순서는 AS2 0~11 / PiPER 12~17 / `piper_gripper_joint1` 18 / `piper_gripper_joint2` 19. 실제 최종 URDF와 MuJoCo joint 주소로 검증할 것.
- PiPER 기존 그리퍼 베이스의 **좌우 전체 폭 142.5 mm**. 이것은 EE 축 방향 길이나 플랜지-손가락 관절 거리로 해석하면 안 됨.
- V7은 열린 상태에서 두 손가락 **바깥 폭 134.5 mm**를 목표(베이스 양 끝에서 각각 **4 mm 안쪽**), 손가락 높이 대칭 정렬, 닫힘 시 접촉면 중첩 방지(~2 mm 여유)로 보정 완료. 실제 금속/플라스틱 볼트 결합 치수는 별도 실측 필요.
- UMI 구성: **PLA ㄷ자 부품의 막힌 면이 PiPER 슬라이딩 레일 접합면에 맞닿고, TPU 부품이 ㄷ자 홈에 들어간다.** 양 손가락의 넓고 평평한 안쪽 접촉면이 서로를 바라보는 방향. V7 조립 결과 화면은 사용자에게 승인받았으므로 회전 방향을 뒤집지 말 것.
- 그리퍼 관절이 **움직일 수 있어야 한다**. 과거 정적 preview용으로 DOF 18로 만든 모델을 동역학용으로 사용하지 말 것.
- 구형 테스트에서는 gripper open target이 `(+0.05m, -0.05m)`이었으나 **V7 최종 한계/offset에 맞는 열림 목표**는 URDF/제어 코드에서 다시 확인할 것.

### 4.2 기존 AS2 본체 메시 측정값(원본 STL의 정점 범위; CAD 체결면 아님)

- AS2 base mesh AABB (`base_link.STL`):
  - X: `[-0.1750, 0.3347] m`
  - Y: `[-0.1135, 0.1135] m`
  - Z: `[-0.0633, 0.1348] m`
  - XY 중앙 부근 정점 Z: `[-0.0633, -0.0520] m` — 상판 높이를 의미하지 않을 수 있음.
- PiPER base mesh AABB (`base_link.stl`):
  - X: `[-0.0585, 0.0400] m`
  - Y: `[-0.0400, 0.0400] m`
  - Z: `[0.0000, 0.0860] m`
- **메시의 global `max(Z)=0.1348m`를 로봇 중앙의 실제 체결면이라고 가정하지 말 것.** 원본 URDF의 `<visual><origin>` 및 메시 좌표계/단위를 반영하고, 상판·커버·내부 하중 지지 프레임을 구별할 것.
- PiPER 베이스 체결 홀은 매뉴얼 참고값 `70×70 mm`, M5×4라고 이전 조사에서 기록됐으나 **실제 해당 기종 CAD/하드웨어에서 다시 확인 필요**.

### 4.3 현재 장착 모델의 문제(해결되지 않음)

- 초기 통합 URDF는 `base_link -> piper_base_link`에 `as2_to_piper` fixed joint를 배치했고, 과거 생성 스크립트의 임시 오프셋은 `(0, 0, 0.15)m`이었다. **V7에 현재 반영된 값은 직접 확인**.
- MuJoCo 생성기는 AS2 plate 장착 기준을 **검증 없이 `estimated_as2_mount_surface_z_m = 0.135`m**로 설정한 버전이 있다. 영상에서 판이 AS2 상판 위에 **떠 있는 현상**을 사용자가 발견했다. 이 문제는 물리 장착 위치 자체가 틀린 것.
- MuJoCo 생성 모델 로그:
  ```text
  profiles_plate: PiPER z=0.1610m, mount mass=0.6145kg, motors=20, mesh=28
  plate_only:     PiPER z=0.1410m, mount mass=0.3645kg, motors=20, mesh=28
  ```
  초기 설계 가정은 **프로파일 높이 20mm, plate 두께 6mm, 구조물 질량 차이 0.25kg**. **실측 치수가 아닌 설계 후보**.
- 실제 AS2 상부 레일 홀 간격, 레일 장착 높이, 체결 구조, 판 지지 위치는 아직 확정 안 됨. 허위의 정밀 결합이라고 주장하지 말 것.
- PiPER 암의 초기 관절값 `[0,1.2,-1.4,0,0.2,0]`로 높이 솟은 모습이 렌더링된 바 있음. 대안 `[0,0.15,-0.10,0,0,0]`을 과거 임시 stow 후보로 검토했으나 **최종 합당한 초기 보행 자세로 인증된 것은 아님**. 기구학/자기충돌/관절 한계에 맞춰 검증할 것.

## 5. 현재 MuJoCo 진행 및 검증 사실

- `python mujoco_mount_compare/build_mount_scenes.py`로 두 MJCF가 생성됨.
- MuJoCo `3.1.6`에서 두 모델 로딩 성공:
  ```text
  profiles_plate: nq=27, nv=26, nu=20, nbody=33
  plate_only:     nq=27, nv=26, nu=20, nbody=31
  ```
- Standing/Crouch/Arm-motion PD 데모 CSV 생성 성공. 과거 결과:

  | Mode | Variant | CoM mean z (m) | Pitch RMS (deg) | Mean sum abs joint torque (Nm) |
  |---|---|---:|---:|---:|
  | stand | profiles_plate | 0.38363 | 1.24728 | 74.39454 |
  | stand | plate_only | 0.37885 | 1.22152 | 73.57012 |
  | crouch | profiles_plate | 0.33021 | 1.42601 | 78.23654 |
  | crouch | plate_only | 0.32551 | 1.39451 | 77.39692 |
  | arm | profiles_plate | 0.38331 | 1.23791 | 75.26222 |
  | arm | plate_only | 0.37854 | 1.21175 | 74.42891 |

  **경고:** 이 수치는 **잘못되었을 가능성이 있는 장착 형상을 사용한 PD 데모**에서 나왔으므로, 실제 하드웨어 안정성/locomotion 성능 또는 장착 최적화 근거로 사용하면 안 됨.
- 이전 기술적 오류와 수정 기록:
  - PiPER visual DAE는 MuJoCo에서 직접 처리 안 됨 → 대응되는 `.stl` 메시 선택 처리.
  - MJCF `<inertial>`에 `quat`+`fullinertia` 동시 사용 오류 → 관성 텐서를 `I_body = R I_URDF R^T`로 회전한 후 `fullinertia`만 넣는 방식으로 수정.
  - `run_mount_demo.py` 함수 안 `import mujoco.viewer`가 지역변수 `mujoco`를 가려 `UnboundLocalError` → `from mujoco import viewer`로 수정.
  - EGL: A100 MIG에서 렌더 불가. OSMesa가 설치되어 **360×640 RGB 프레임 렌더 성공**. CPU MP4 영상 생성 가능.
- 모델 로드/영상 성공은 **맞는 조립을 보증하지 않는다**. 기구/메시 측 검증이 별도로 필수.

## 6. 정확하게 고치는 작업 순서 (순서 변경하지 말 것)

### A. 읽기 전용 진단

1. `git status --short` 확인 후 개발 저장소만 조사.
2. `as2piper_umi_actuated_v7.urdf`에서 `as2_to_piper`, 링크/메시 `<origin>`, PiPER base의 orientation, arm joint axis, gripper TCP 위치를 추출.
3. `build_mount_scenes.py`, `mount_config.json`, 생성된 두 `.xml`을 읽고 변환 행렬 체인을 기술:
   ```text
   world → floating AS2 base_link → AS2 mounting interface
      → profile(s), if any → plate → PiPER base_link
      → PiPER arm links → UMI gripper / TCP
   ```
4. STL/DAE의 원점·단위·visual origin을 확인. 실제 상부 판이 있는 위치의 삼각형 형상과 레일 표면을 **국소 XY ROI**에서 판별. AABB의 단일 최대 Z 이용 금지.
5. 두 모델의 mount plate와 AS2 실제 구조 사이 **수직 거리·표면 간격·관통 여부**를 수치화. plate↔PiPER bottom도 별도 측정.

### B. URDF 및 장착 구조 수정

6. 실제 물리 체결면이 자료로 확인되지 않았다면 **'가정된 interface'와 필요한 현물 치수 목록을 명확히 기록**. 상부 외장 커버와 구조 프레임 구분. 체결 나사 구멍 간격을 추측으로 확정하지 말 것.
7. 접합면을 정의하고 로봇 장착 기구를 **명시적 링크/고정조인트, visual, collision, mass/inertia**로 모델링. 단일 `as2_to_piper` 임시 오프셋이 과거 형상을 다시 띄우지 않도록 변환 생성 체인을 일관화.
8. 두 variant의 유일한 차이는 **프로파일의 존재/추가 높이·질량**이 되도록 하고, 두 버전의 PiPER orientation 및 다른 로봇 설정은 동일하게 유지.
9. **원본 V7을 그대로 보존한 새 결과 파일명 사용**. 원본 생성기/검증 스크립트가 필요하다면 별도 브랜치/복사본에서 변경.
10. MuJoCo 변환과 초기 관절각/카메라는 **접합면 검증 후** 수정. 베이스 고정 조인트를 시각적으로 만족시키기 위한 가짜 메시나 가짜 지지 접촉을 넣지 말 것.

### C. 자동 검증·시각화

11. MuJoCo `MjModel.from_xml_path` 로딩, `nq/nv/nu`, 링크/관절 순서 검사.
12. 접합부 기하학에 대해 기계적 기준을 분리해 검증:
    - 의도된 AS2 structural support↔plate 접촉/볼트 기준면 간격 **≈0**, 미세 허용 오차는 명시 (예: ≤1mm 목표).
    - plate↔PiPER base 접합 간격 **≈0**.
    - 명백한 메시 관통/공간 부유 없음; 프로파일 버전은 실제 양쪽 레일과 판 사이 지지 경로 있음.
    - UMI V7 손가락은 앞서 승인된 조립/방향/높이/너비를 유지.
13. 고정/중립 자세에서 **정면·측면·위쪽·장착부 확대** 3D 프레임을 각각 출력하고, 두 버전의 동일 카메라 비교 이미지를 제공.
14. 그 다음 중력·접촉 기반 stand / crouch / arm 데모 MP4를 출력. 떨어지거나 비정상적인 joint pose면 원인별 수치 로그를 출력.
15. 각 variant에서 base CoM, 발별 지면 접촉과 normal force, 로봇 자세, actuator torque, TCP 지면 높이(특히 crouch) 기록. `contact_samples` 수치만으로 네 발 안정 접촉이라고 판단하지 말 것.
16. 산출물은 **diff, 재실행 명령, 변경 전후 정면/측면 스크린샷, 실제 간격·관통 측정표, 불확실성 목록**을 포함.

## 7. 장착 위치 및 학습 목적

- AS2 `base_link` 기준 XY 중심을 초기 후보로 두고, 전방 장착 후보 `x=0`, `+0.05`, `+0.10m`의 reach vs stability를 **장착 접합이 정상화된 뒤** 비교할 계획. 좌우는 `y≈0` 우선 후보.
- **Locomotion:** 팔을 몸통 가까이 접은 자세가 중력/관성 관점에서 유리할 수 있으나, 접힌 자세의 충돌/모터 한계 검사 필요.
- **Crouch ground reach:** 안정적으로 몸통을 숙였을 때 UMI 실제 접촉점이 지면에 닿을 수 있어야 한다는 **필수 제약조건**. 단순한 `piper_link6` 위치가 아니라 실제 손가락/TCP 좌표를 사용.
- **Dragging:** TCP에 수평/수직 반력을 넣어 전복 모멘트/지면 접촉력 측정.
- 하드웨어 제작 이전에 AS2 실제 상부 프레임/레일 홀 패턴, PiPER 70×70mm 홀 패턴, 장착 하중 경로, 구조 부품 질량과 나사 결합을 검증. 추정 치수는 CAD 설계 후보일 뿐.

## 8. Codex 작업 방식 및 보고 형식

- **실행 가능한 확인·수정까지 진행**하되, 중요한 미확정 물리 치수는 임의로 '검증됨' 처리하지 말 것.
- 진단 결과가 이전 안내와 충돌하면 **현장 파일, 실제 STL, MuJoCo 실행 결과**를 우선하고 원인을 설명.
- 처음에는 변경 계획과 위험 파일 범위를 짧게 출력. 그 뒤 파일 수정/테스트/렌더링을 진행.
- 새 패키지 다운로드를 반복하지 말고 이미 원격 저장소에 있는 코드·메시를 직접 고칠 것.
- **기존 Baseline 손상 없음을 최종 확인**하고, `git diff --stat`, 변경 경로, 검증 명령, 남은 물리적 불확실성을 요약.
- 주 사용 언어: **한국어**. 코드/로그/커밋 메시에는 명확한 영어 식별자를 사용해도 됨.

### 시작할 때 권장하는 읽기 전용 명령

```bash
cd /data/team/users/suhwan/vbc-as2piper
pwd
whoami
git status --short
find low-level/resources/robots/as2piper/urdf -maxdepth 1 -type f -print
sed -n '1,260p' mujoco_mount_compare/build_mount_scenes.py
cat mujoco_mount_compare/mount_config.json
```

### Codex에게 처음 요청할 내용

> `AGENTS.md`를 읽고 현재 V7 URDF와 MuJoCo 두 variant의 변환 체인을 먼저 감사해 줘. **AS2와 장착판이 떨어져 있는 원인을 실제 mesh/URDF 좌표로 계산**하고, 접합부를 바르게 수정해 줘. 시작부터 팔 pose/카메라만 고치지 말 것. V7 그리퍼 20 DOF와 WBC 18-action 설계 유지, `/data/team/users/suhwan/vbc-legacy` 수정 금지. 수정된 두 모델을 4시점 렌더링하고 간격/관통·MuJoCo 로드/동역학 테스트 결과로 검증해 줘.
