# AS2 + PiPER + UMI V7 — MuJoCo 장착 비교 (설계 후보)

이 폴더는 기존 `vbc-as2piper` 작업 트리 안에 넣는다. **`vbc-legacy`/기존 B1Z1 학습을 수정하지 않는다.** V7 모델의 다리 12개, PiPER 6개, 그리퍼 2개 = 20개의 구동관절에 6자유도 부유 베이스를 추가해 MuJoCo에서 중력과 접촉이 있는 시뮬레이션을 수행한다.

## 모델 비교

- `profiles_plate.xml`: AS2 상부 기준면 → **20mm 두 개의 알루미늄 프로파일** → 6mm 알루미늄 플레이트 → PiPER
- `plate_only.xml`: AS2 상부 기준면 → **6mm 알루미늄 플레이트** → PiPER
- 공통 값: 베이스 기준 전방 X=+50mm, Y=0mm, AS2 장착 기준면 Z=135mm(임시), 플레이트 150×150×6mm, 밀도 2700kg/m³.
- 프로파일: 250mm 길이 × 20×20mm 단면 두 개, Y=±54mm, 질량 **0.50kg/m** 가정(속이 빈 압출재).
- 총 어댑터 질량: 플레이트 **0.3645kg**, 프로파일 **0.2500kg** 추가. PiPER 설치 높이 `plate_only=0.141m`, `profiles_plate=0.161m` (`base_link` 기준). 두 형상 모두 실제 AS2의 홀 패턴/장착 강도를 검증한 설계가 **아님**.

## 0. 사전 파일 확인

아래 파일이 이미 존재해야 한다.

```
vbc-as2piper/
  low-level/resources/robots/as2piper/
    urdf/as2piper_umi_actuated_v7.urdf
    meshes/as2/... 
    meshes/piper/...
    meshes/umi/umi_pla.stl, umi_tpu.stl
  mujoco_mount_compare/  ← 이 ZIP의 폴더를 여기로
```

원본 V7 URDF와 실제 메시 파일을 그대로 읽으며, 모든 MJCF 모델은 `mujoco_mount_compare/models/` 아래에 생성된다. 원본 URDF나 이미 완료한 WBC 코드는 수정하지 않는다.

## 1. AstraGo — 독립 MuJoCo 환경 설치 (Python 3.8)

기존 PyTorch/Isaac Gym 환경과 충돌하지 않도록 별도 venv를 생성한다.

```bash
cd /data/team/users/suhwan/vbc-as2piper
python3 -m venv /data/team/users/suhwan/venv-mujoco38
source /data/team/users/suhwan/venv-mujoco38/bin/activate
python -m pip install --upgrade pip
python -m pip install 'mujoco==3.1.6' 'numpy<1.25' 'imageio' 'imageio-ffmpeg'
python -c 'import mujoco; print(mujoco.__version__)'
```

Python 3.8에서는 `mujoco==3.1.6`을 사용한다. Python 3.9+의 더 최신 MuJoCo를 동일하게 사용할 경우 물리 엔진 결과가 조금 다를 수 있으니 비교는 동일 버전으로 진행한다.

## 2. 실제 로봇 URDF → 두 개의 MJCF 생성

```bash
cd /data/team/users/suhwan/vbc-as2piper
python mujoco_mount_compare/build_mount_scenes.py
ls -lh mujoco_mount_compare/models/
```

생성 후 `models/profiles_plate.xml` 및 `models/plate_only.xml`을 MuJoCo에서 로딩한다. 빌더는 **20 모터 + 부유 베이스**, 관절 이름 및 STL 리소스 경로를 유지한다. PiPER의 DAE 시각화 메시 대신 짝이 되는 공식 STL을 사용하는 경우 색상/재질은 완전 동일하지 않을 수 있다.

## 3. AstraGo SSH 서버 — GUI 없는 물리 시뮬레이션

아래 명령은 GPU를 요구하지 않는다.

```bash
python mujoco_mount_compare/run_mount_demo.py --variant profiles_plate --mode crouch --seconds 8
python mujoco_mount_compare/run_mount_demo.py --variant plate_only --mode crouch --seconds 8
python mujoco_mount_compare/compare_results.py --mode crouch
```

모드 종류:

- `stand`: 중력/지면접촉 + 서기 자세 유지용 PD (기본)
- `crouch`: 두 환경에서 같은 다리 굽힘 자세 명령으로 낮추기
- `arm`: 같은 팔 관절 움직임 추종
- `step`: 다리 관절 교번 사인 명령 **데모일 뿐 안정적인 보행 정책 아님**

실행 결과는 `mujoco_mount_compare/results/{variant}_{mode}.csv`로 저장한다. `compare_results.py`는 CoM 높이, 몸통 Roll/Pitch RMS, 높이 최솟값, 액추에이터 명령량 등을 비교한다. **보행이 보장되지는 않는다.** 실제 비교용 WBC/Locomotion policy를 연결하기 전 동역학 및 접촉 확인 실험이다.

## 4. 움직이는 영상 만들기 (원격/headless, 선택)

서버에서 EGL이 지원된다면:

```bash
MUJOCO_GL=egl python mujoco_mount_compare/run_mount_demo.py --variant profiles_plate --mode arm --seconds 8 --video
MUJOCO_GL=egl python mujoco_mount_compare/run_mount_demo.py --variant plate_only --mode arm --seconds 8 --video
```

`results/`에 MP4 파일이 생성된다. **NVIDIA MIG, EGL 라이브러리, 디스플레이 설정에 따라 영상 렌더링이 실패할 수 있다.** 이 경우 OpenGL 소프트웨어 렌더링이 설치되어 있다면 `MUJOCO_GL=osmesa`도 시도할 수 있지만, AstraGo 환경에서 동작이 보장되지는 않는다. 영상 렌더링 실패는 물리 시뮬레이션 실패와 구별한다.

## 5. Windows/로컬 WSL — 직접 움직이는 GUI 보기

MuJoCo Viewer는 데스크톱 창이 필요하므로, 일반적인 AstraGo Remote-SSH 터미널에서 `--gui`를 실행해도 Windows 화면이 나타나지 않는다. 로컬 Windows/WSL에서 **프로젝트 구조와 실제 메시 파일까지 복사**해 같은 위치에 두고 Python/MuJoCo를 설치한 뒤 실행한다.

```bash
python mujoco_mount_compare/build_mount_scenes.py
python mujoco_mount_compare/run_mount_demo.py --variant profiles_plate --mode arm --seconds 10 --gui
# 다른 터미널/실행에서
python mujoco_mount_compare/run_mount_demo.py --variant plate_only --mode arm --seconds 10 --gui
```

이미 두 XML을 생성했고 같은 디렉터리 구조로 프로젝트를 복사했다면 1번째 빌드 명령은 생략 가능하다. 단, XML 안의 메시 경로가 유효해야 한다. 로컬에서는 설치된 Python 버전에 맞는 MuJoCo를 사용한다.

## 6. 정확도 / 다음 검증

설계 후보로 확인해야 할 것:

1. AS2 몸통 커버가 아닌 **실제 하중 지지 프레임**의 장착면 높이와 홀 패턴.
2. 프로파일/플레이트 볼트 규격과 강성, 실제 장착 하중 및 모멘트.
3. 실제 사용 부품의 질량·관성, 20개 관절 기어 감속비·토크 한계·마찰.
4. PiPER·AS2의 충돌 형상. MuJoCo는 Mesh 충돌에 볼록껍질 근사를 적용하므로 손가락/팔 자기충돌 평가 시 주의.
5. **Crouch에서 TCP가 바닥에 도달 가능한지**는 별도의 IK/접촉 조건 검증이 필요. 단순 `crouch` 모드는 그 조건을 자동으로 만족시키는 정책이 아님.
6. MuJoCo에서 검증된 안정적 보행 및 동일한 WBC 정책 연결 전에는 `step` 모드만으로 Locomotion 성능을 주장할 수 없음.

하드웨어 후보 치수는 `mount_config.json`만 변경하면 재생성할 수 있다. 같은 관절/제어 입력으로 두 장착 높이를 비교하도록 구성했다.
