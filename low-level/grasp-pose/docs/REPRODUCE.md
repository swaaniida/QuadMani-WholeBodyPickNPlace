# 처음부터 복원

경로는 clone 위치로 바꿀 수 있다. 코드 대부분은 프로젝트 상대경로다. OBB PowerShell 원격 경로와 rig generator의 기본 외부 로봇 경로는 현재 astrago 환경 예제이므로 다른 서버에서는 직접 수정/인자를 지정한다.

## Linux / A100

필수: Git, curl, NVIDIA driver, CUDA11.7 toolkit(Tracking CUDA extension 빌드), C++ compiler/toolchain. nvcc 경로 기본은 `/usr/local/cuda-11.7`. 설치는 별도 환경만 만들며 기존 서비스를 재시작하지 않는다. 기존 CUDA/driver를 자동 변경하지 않는다.

```bash
cd grasp-pose/graspgenx-realsense
bash scripts/install_server.sh
bash scripts/install_tracking.sh
bash scripts/install_client_tools.sh
# 실제 공식 RGB-D sample의 LFS 파일 복원:
.venv/bin/python scripts/fetch_sample_lfs.py
```

uv0.13.7/Python3.10을 프로젝트 내부에 설치한다. `configs/*freeze.txt`는 당시 환경 증거이며 완전한 cross-platform lockfile이 아니다. 추적 GPU toolkit/아키텍처는 install_tracking.sh의 실제 빌드 환경에 맞춰야 한다. 현재 SM80/A100 기준이다. Google Drive 모델 다운로드 제한이나 upstream 접근 제한은 별도 해결이 필요하다.

**이미 실행 중이면 새 서버를 중복으로 띄우지 않는다.**

```bash
bash scripts/start_server.sh
# 별도 terminal:
bash scripts/start_tracking_server.sh
# localhost health (inference 아님):
.venv-client/bin/python local_client/grasp_client.py
```

기본127.0.0.1:5556/5557. 부팅 서비스는 자동 등록하지 않는다.

## Windows / D435i

이 Git 저장소를 Windows에도 clone했다면 `graspgenx-realsense/local_client`로 이동한다. 별도 ZIP이 필요하면 Linux에서 `python scripts/package_windows.py`로 생성한다. ZIP은 GitHub 소스에 포함하지 않는다.

```powershell
Set-Location "$env:USERPROFILE\grasp-pose\graspgenx-realsense\local_client"
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-temporal.txt
```

기존 설치 폴더가 `graspgenx-realsense\local_client`라면 그 위치를 그대로 사용한다. 새 PowerShell에서 tunnel을 유지한다:

```powershell
ssh -p 2222 -i "$env:USERPROFILE\.ssh\id_ed25519" -N -o ExitOnForwardFailure=yes -L 127.0.0.1:15557:127.0.0.1:5557 suhwan@astrago
```

실행 terminal:

```powershell
.\.venv\Scripts\python.exe main_temporal.py --port 15557 --fps 15 --auto-grasp --duration 6
# 클릭: 초기 mask/추적/기록, G: 수동 grasp, R: reset/new take, S: save, Q: 종료
# 최신 유효 비교 녹화 재생:
.\.venv\Scripts\python.exe view_temporal.py
# 특정 녹화:
.\.venv\Scripts\python.exe view_temporal.py "outputs\tracking_temporal\auto_sessions\SESSION"
# 카메라+거치대+기존UMI 확인:
.\.venv\Scripts\python.exe view_wrist_rig.py --rig-only
```

카메라 단독 점검/실측 calibration은 다른 camera 앱을 닫고:

```powershell
.\.venv\Scripts\python.exe export_realsense_calibration.py --serial YOUR_SERIAL
```

기록은 로컬 `outputs*/` 아래 생성되며 Git에서 제외한다. RGB-D/Mask/CSV/NPZ를 함께 백업한다. depth filtering은 near>0, far>near; band=0은 물체 mask 전체 depth 범위 유지다.

## 검사와 실제 inference 구분

```bash
# GPU/USB/service 호출 없는 회귀 검사; upstream이 없으면 공식 OBB helper 1개 skip:
.venv-odometry/bin/python -m pytest -q tests
# 실제 GPU 공식 샘플 검증 (사용자가 필요할 때 실행):
.venv/bin/python scripts/sample_inference.py --runs 10
# 실행 중인 두 서버와 실제 모델 요청을 수행하는 통합 검사:
.venv-client/bin/python scripts/tracking_integration_smoke.py
```

전체 테스트는 NumPy/Viser/Open3D/OpenCV/pyrealsense2 의존성이 필요하다. Linux pyrealsense2 import는 libusb가 필요할 수 있다. Open3D 계산도 CPU 자원을 사용한다. 이번 Git 정리에서는 모델 inference나 서비스 재시작을 하지 않는다.

## optional geometry rebuild / profiling

기본 viewer는 이미 포함된 rig만으로 동작한다. 재생성은 공식 camera source와 외부 robot URDF/meshes 필요:

```bash
bash scripts/fetch_camera_source.sh
.venv-odometry/bin/python scripts/build_wrist_camera_rig.py --robot-urdf /PATH/TO/as2piper_umi_open_pose_v7.urdf --clocking-deg 180 --camera-roll-deg 0
```

Tracking model profile 재실행 소스는 `scripts/profile_tracking_models.py`. `outputs/model_profile/manual/`에 저장하며 별도 Tracking replica로 GPU 메모리를 추가 사용한다. 모델/데이터/두 서버가 준비된 때만 `.venv-tracking/bin/python scripts/profile_tracking_models.py` 실행. 기존 측정 원본은 `docs/evidence/`에 별도 보존한다. 학습/제어는 하지 않는다.

## 라이선스·외부 파일

`THIRD_PARTY_NOTICES.md`와 `docs/DEPENDENCIES.json` 참조. 모델은 공식 revision에서 다운로드하며 Git에 올리지 않는다. 포함된 CAD 중 사용자 제공 UMI/거치대 및 로봇 원본은 권리자가 정한 배포 조건을 따른다. 이 정리로 새 blanket license를 임의 부여하지 않았다.
