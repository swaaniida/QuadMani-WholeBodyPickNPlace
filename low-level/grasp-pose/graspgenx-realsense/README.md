> Historical milestone README. For current setup and selection behavior see [reproduction](../docs/REPRODUCE.md) and [current state](../docs/CURRENT_STATE.md). Older weights/status statements below describe their original milestone.

# GraspGenX–RealSense 독립 PoC

## Tracking SAM 통합

기존 기능에 **공식 SAM ViT-H + AOTT VOS** 서비스를 추가했다. 클릭 또는 텍스트로 초기화하고 연속 추적한 mask를 공식 GraspGenX `infer_scene_depth()`로 전달한다. 기존 수동 ROI 모드와 GraspGenX 서버는 유지한다. [실행 및 Windows 절차](local_client/TRACKING_README.md), [실제 검증 보고](TRACKING_REPORT.md), [통합 요청 사양](TRACKING_TASK.md).

서버는 서로 다른 환경의 독립 프로세스이며 localhost에만 바인딩한다.

```bash
cd /data/team/users/suhwan/grasp-pose/graspgenx-realsense
# 기존 GraspGenX 서버(5556)가 꺼져 있을 때만 시작
bash scripts/start_server.sh
# 별도 터미널: Tracking SAM 서비스(5557)
bash scripts/start_tracking_server.sh
```

현재 보고 시점에는 두 서버 모두 실행 중이다. 중복 시작하지 않는다. Tracking 로그/PID는 `outputs/tracking/server.log`, `server.pid`이다. 새 환경 재설치가 필요하면 `bash scripts/install_tracking.sh`를 사용한다. GraspGenX의 `.venv`와 Tracking SAM의 `.venv-tracking`을 섞지 않는다.

공식 demo 및 통합 재검증:

```bash
HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch" .venv-tracking/bin/python scripts/tracking_official_demo.py
.venv-client/bin/python scripts/tracking_integration_smoke.py --text-test
.venv-client/bin/python scripts/tracking_worker_smoke.py
.venv-client/bin/python scripts/measure_tracking_gpu.py
.venv/bin/python -m pytest -q tests
.venv-client/bin/python scripts/package_windows.py
```

공식 DAVIS 동영상은 실제 연속 추적을 검증했다. RGB-D 통합은 공식 녹화 데이터 한 장의 **static replay**이며 live D435i 실험과 구분한다. Windows USB/GUI/SSH 성능 및 실제 움직이는 RGB-D의 최종 하드웨어 검증은 아직 필요하다.

요청 사양은 [TASK.md](TASK.md), 실제 실행 결과는 [REPORT.md](REPORT.md), Windows 절차는 [local_client/README.md](local_client/README.md)에 저장했다. 프로젝트 전체와 Python runtime/환경은 `/data/team/users/suhwan/grasp-pose/graspgenx-realsense` 안에 있다. 기존 상위 README와 로봇 프로젝트는 유지했다. 이전 경로에 호환용 심볼릭 링크를 남기지 않았다.

프로젝트 경로:

```text
/data/team/users/suhwan/grasp-pose/graspgenx-realsense
├── TASK.md, README.md, REPORT.md
├── upstream/GraspGenX/         # 공식 pinned source, checkpoints, franka_umi assets
├── .venv/                     # 독립 Python 3.10 / torch 2.5.1+cu121
├── .venv-client/              # 별도 CPU client 검증 환경; torch 미설치
├── .runtime/python/           # 프로젝트 내부 Python 3.10 runtime
├── server/serve.py             # 공식 서버 재사용 + synchronized metrics
├── local_client/              # Windows 복사용 코드, 공식 CAD 및 sweep params
├── scripts/                   # 설치, 공식 sample, ZMQ, RGB-D replay, ZIP
├── configs/                   # franka_umi sweep, 환경 freeze
├── tests/                     # metric geometry, depth filtering, pose, timeout
├── docs/DECISIONS.md
├── outputs/{rgb,depth,pointcloud,grasps,benchmarks}/
└── dist/graspgenx-realsense-windows.zip
```

## 서버 실행

```bash
cd /data/team/users/suhwan/grasp-pose/graspgenx-realsense
bash scripts/start_server.sh
```

현재 127.0.0.1:5556에서 이미 서버가 실행 중이면 중복 실행하지 않는다. 로그와 PID는 `outputs/benchmarks/server.log`, `server.pid`에 있다. 확인:

```bash
.venv/bin/python local_client/grasp_client.py
ss -ltnp '( sport = :5556 )'
```

종료 시 PID 파일의 프로세스가 이 프로젝트 서버인지 `ps -fp`로 확인한 후 해당 PID에만 SIGTERM을 보낸다. GPU reset, 전체 Python 프로세스 종료, 시스템 driver 변경은 하지 않는다. 상주 서버는 자동 부팅 서비스로 등록하지 않았다.

공식 CLI 자체도 실제 실행/health/추론/loopback 바인딩을 검증했다. 추가 metrics 없이 공식 CLI만 실행하려면 기존 5556 서버를 먼저 종료하거나 별도 포트를 쓴다.

```bash
.venv/bin/python upstream/GraspGenX/client-server/graspgenx_server.py \
  --config upstream/GraspGenX/ext/graspgenx_checkpoints/release \
  --assets_dir upstream/GraspGenX/ext/gripper_descriptions/gripper_descriptions/assets \
  --default_gripper franka_umi --host 127.0.0.1 --port 5556
```

## Windows로 복사

PowerShell에서:

```powershell
scp suhwan@astrago:/data/team/users/suhwan/grasp-pose/graspgenx-realsense/dist/graspgenx-realsense-windows.zip .
Expand-Archive .\graspgenx-realsense-windows.zip -DestinationPath .\graspgenx-realsense
cd .\graspgenx-realsense\local_client
```

이후 [Windows 설치/터널/실행 절차](local_client/README.md)를 따른다. 실제 SSH 포트가 별도면 기존 SSH config를 쓰거나 scp에 `-P`, ssh에 `-p`를 추가한다.

## 재현 가능한 서버 검증

```bash
# 독립 환경 재설치가 필요한 경우 (현재는 설치 완료)
bash scripts/install_server.sh
.venv/bin/python scripts/sample_inference.py --runs 10
# 5556 서버가 실행 중인 상태에서:
.venv/bin/python scripts/network_benchmark.py --runs 10
# 공식 recorded RGB-D 샘플을 LFS SHA-256 검증 후 가져와 replay:
.venv/bin/python scripts/fetch_sample_lfs.py
.venv/bin/python scripts/rgbd_replay_smoke.py
.venv-client/bin/python scripts/check_scene_depth.py
# 별도 5557 포트에서 공식 CLI 자체를 검증하고 그 프로세스만 종료:
.venv/bin/python scripts/check_official_cli.py
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/package_windows.py
```

기존 서버 driver 535를 유지하기 위해 CUDA 13/TensorRT fast-inference 설치는 하지 않았다. 시스템 nvcc 11.7과 별개로 torch cu121 runtime을 사용한다. checkpoint/asset 다운로드는 시스템 git-lfs 없이 공식 Hugging Face snapshot으로 수행했다. upstream 실험용 sample `.npy` LFS 파일은 별도 hydration script로 확인한다. 모든 sample directory가 hydration된 것은 아니다.

## 결과와 범위

공식 sample point cloud의 실제 A100 추론, ZMQ 세 가지 경로, 녹화 RGB-D → ROI → cloud → 원격 추론 → 실제 CAD/score scene → overlay/파일 저장을 검증했다. **Windows 실제 USB 카메라 획득, 하드웨어 재연결, 브라우저/GUI 조작 및 SSH 구간 RTT는 아직 검증하지 않았다.** 목업 추론은 없다. 모델을 학습시키거나 로봇에 명령을 보내지 않는다.

`infer_scene_depth()`의 공식 wire 인터페이스도 client에 제공하지만 기본 UI는 segmented point cloud 경로이다. Scene-depth는 rectified pinhole K와 정수 instance mask를 요구한다. 해당 선택 인터페이스는 현재 하드웨어 완료 기준에 포함하지 않았다.

참고: [공식 GraspGenX](https://github.com/NVlabs/GraspGenX), [standalone server](https://github.com/NVlabs/GraspGenX/tree/main/client-server). 설치 commit 및 실제 측정값은 REPORT.md에 명시한다.

Grasp 선택은 클라이언트에서 `S=0.7C+0.1R+0.1P+0.1Y`로 재정렬합니다. 최종 S#1과 기존 C#1 비교, 좌표축 정의 및 Windows 업데이트는 [선택 기준 안내](local_client/GRASP_SELECTION.md)를 참고하세요. 모델/추적 서버 재시작은 필요 없습니다.

Tracking 중 자동 Grasp 추론 및 최종 Score Top-1 기록, Frame ID 기준 RGB-D/Mask 아카이브, pose 변화량/FPS 측정과 종료 후 재생: [자동 기록 사용법](local_client/AUTO_GRASP.md).
