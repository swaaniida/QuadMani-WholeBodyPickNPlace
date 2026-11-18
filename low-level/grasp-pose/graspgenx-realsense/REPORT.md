# 실행 결과 보고 — 2026-11-11 UTC

## 1. 구현 및 완료 상태

요청 문서를 [TASK.md](TASK.md)에 저장하고 실제 설치/실행을 수행했다. 권한 부여 후 프로젝트 전체를 `/data/team/users/suhwan/grasp-pose/graspgenx-realsense`로 이전했다. Python runtime은 프로젝트 내부 `.runtime/python`, 환경은 `.venv` 및 `.venv-client`에 있다. 이전 경로에 호환용 심볼릭 링크를 남기지 않았고, 사용자의 최신 지시에 따라 서버를 새 경로에서 재시작했다. 다른 로봇 프로젝트와 Python 환경은 변경하지 않았다.

| 항목 | 결과 및 근거 |
|---|---|
| 공식 sample PC → 실제 A100 inference | 완료. 10,703점, diffusion 200 후보, top 100. 두 planner 각각 warmup 1회 + 측정 10회 |
| SE(3)/confidence 검증 | 완료. shape, finite, homogeneous row, RᵀR≈I, det(R)≈1, score [0,1] 검사 |
| ZMQ health / loopback 바인딩 | 완료. 127.0.0.1:5556. [health.json](outputs/benchmarks/health.json) |
| grasp-pose로 전체 이전 후 검증 | 새 Python runtime/venv 및 새 경로의 서버에서 health OK, 실제 GraspMoE 100개 반환, 테스트 8개 통과. [relocation.json](outputs/benchmarks/relocation.json) |
| 공식 CLI 자체 실행 | 완료. 별도 127.0.0.1:5557에서 health와 100개 grasp 반환 확인 후 해당 프로세스 종료 |
| 원격 inference 경로 | name/diffusion, params/graspmoe, params/diffusion 모두 실제 모델 검증 |
| 서버 오류 처리 | unknown action, 잘못된 cloud shape, sweep 누락 시 오류 반환; 이후 health 복구 |
| Windows client 코드 | RGB-D acquisition/alignment, ROI/depth mask/cloud, 비동기 inference, 저장, overlay/Viser 구현 |
| RGB-D pipeline 검증 | 공식 녹화 데이터 replay에서 실행. 1,219점 → 100개 grasp. Windows USB 촬영과 구분 |
| 3D/overlay | 실제 공식 CAD 10개, axes, score, cloud scene 생성 및 HTTP 200 확인. 저장 overlay 이미지를 직접 확인 |
| CPU-only client | torch 미설치 별도 환경에서 client/UI 모듈 import, 실제 health, scene-depth inference 확인 |
| 선택형 scene-depth API | 공식 RGB-D/K/instance mask로 실제 A100 250개 grasp 반환 확인 |
| 자동 테스트 | geometry/filter/overlay/invalid SE(3)/timeout recovery 8개 통과 |
| Windows ZIP의 독립성 | 임시 폴더에 압축 해제 후 20개 파일 SHA-256/CRC, torch 없는 import, 실제 서버 health, 실제 저장 결과 100개 및 CAD 10개/Viser HTTP 검증 통과 |
| Windows USB 장치/GUI/재연결/SSH RTT | **미검증**. 원격 Linux에서 접근 불가; Windows 실행 절차 제공 |

**전체 하드웨어 완료 기준은 아직 충족하지 않았다.** Windows 카메라 획득 및 실제 노트북→SSH→서버→시각화 검증이 남아 있다. 가짜 추론이나 생성한 pose를 실측 결과로 사용하지 않았다. 단위 테스트의 수학적 fixture는 실제 inference와 구분되어 있다.

## 2. 설치 버전 및 서버 환경

- Host `astrago`, user `suhwan`.
- 실제 GPU: **NVIDIA A100-SXM4-80GB, MIG 3g.40gb**, 가용 인스턴스 메모리 **40192MiB**, SM 42개. 전체 A100의 40GB 전용 카드와 구성이 다르다.
- NVIDIA driver **535.161.08**, nvidia-smi CUDA compatibility 표시는 12.2, 시스템 nvcc는 **11.7.99**.
- 독립 Python **3.10.22**, PyTorch **2.5.1+cu121**, NumPy **1.26.4**. torch CUDA runtime **12.1**을 사용하며 시스템 driver/toolkit을 변경하지 않았다.
- GraspGenX commit **`44eb1d77d6d27bc97ee361e3afb717a2115e30a9`** (실행 시 공식 HEAD).
- upstream 배포 metadata 버전 **1.0.0**; 패키지 `__version__`은 **0.1.0**으로 서로 다르므로 commit을 식별 기준으로 사용한다.
- 디스크 조사 시 프로젝트 NFS 약 23TiB 가용, / overlay 약 7.3TiB 가용. 초기 Python 3.8 환경은 그대로 유지했다.
- 자세한 환경: [environment.txt](outputs/benchmarks/environment.txt), [server-freeze.txt](configs/server-freeze.txt).

## 3. Pretrained checkpoint

공식 [adithyamurali/GraspGenXModel](https://huggingface.co/adithyamurali/GraspGenXModel), revision **`7c834043c11a11417e31d6d5ea9355801e40a2c1`**:

- Generator: `release/gen/epoch_736.pth`.
- Discriminator: `release/dis/epoch_1056.pth`.
- 두 object encoder는 **ptv3vanilla**, gripper conditioning은 **sweep_volume_v2**.
- Grasp representation **r3_so3**, released DDPM **20 steps**, eager PyTorch **fp32**.
- 모델 로딩은 프로세스당 한 번이고 모든 sampler가 공유한다. 초기 직접 실험에서 모델 로딩 약 **23,992ms**, 첫 GraspMoE 호출 약 **4,944ms**였으며 warm latency와 분리했다.
- 시스템에 git-lfs가 없어 공식 HF snapshot download를 이용했다. 새 학습/finetuning/TensorRT 컴파일은 수행하지 않았다.

## 4. Gripper configuration

공식 [gripper_descriptions](https://huggingface.co/datasets/adithyamurali/gripper_descriptions), revision **`19a03c00d19aeaf052d0f6801f0041982d676e8a`**, **franka_umi**:

- `sweep_volume.extents = [0.08, 0.022, 0.1]` m.
- `sweep_volume.offset = [0, 0, 0.14]` m.
- Half-open extents `[0.04, 0.022, 0.1]`, offset `[0, 0, 0.14]` m.
- Fingertip depth **0.177m**.
- 실제 공식 `vis_mesh.obj`를 클라이언트에 동봉하고 정적 열린 자세로 표시한다. CAD 로딩 실패 시 공식 sweep 기반 wireframe이라는 사실을 UI에 표시한다.
- 반환 행렬은 **T_camera_gripper_base**이며 로봇 TCP/EE와 같다고 가정하지 않는다. 카메라 optical frame X 오른쪽, Y 아래, Z 전방, meter.

## 5. A100 서버 실행 명령

```bash
cd /data/team/users/suhwan/grasp-pose/graspgenx-realsense
bash scripts/start_server.sh
```

보고 시점에는 이 서버가 **127.0.0.1:5556**에서 실행 중이다. `outputs/benchmarks/server.pid`와 `server.log`로 확인한다. 이 래퍼는 공식 `GraspGenXZMQServer`를 재사용하고 전체 handler의 CUDA 동기화 시간과 allocator peak만 추가한다. 공식 모델, checkpoint loading, planner, ZMQ protocol을 다시 작성하지 않았다. 공식 CLI를 그대로 실행하는 명령도 [README.md](README.md)에 있다. 기본 CLI의 0.0.0.0 바인딩을 사용하지 않는다. 자동 부팅 서비스는 등록하지 않았다.

## 6. Windows SSH 터널

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:5556:127.0.0.1:5556 suhwan@astrago
```

기존 SSH alias/실제 포트를 유지한다. 별도 포트가 필요하면 `-p 실제SSH포트`를 추가한다. 5556을 외부 인터넷에 공개하지 않는다.

## 7. Windows 설치 및 실행

```powershell
scp suhwan@astrago:/data/team/users/suhwan/grasp-pose/graspgenx-realsense/dist/graspgenx-realsense-windows.zip .
Expand-Archive .\graspgenx-realsense-windows.zip -DestinationPath .\graspgenx-realsense
cd .\graspgenx-realsense\local_client
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe grasp_client.py --timeout-ms 5000
.\.venv\Scripts\python.exe main.py
```

S → ROI 선택 → depth 트랙바 조정 → V로 cloud 확인 → G 추론 → V로 3D 결과 → R 초기화 → Q 종료. Top 10, best score, camera axes, 각 score와 행렬/회전, 실제 CAD 및 RGB projection을 제공한다. UI/설정/출력/하드웨어 검증 세부 절차는 [local_client/README.md](local_client/README.md)에 있다.

## 8. 실제 추론 결과

공식 `assets/sample_data/object_pc/1740787815_545011.json`의 **pc만** 사용해 새 inference를 수행했다. sample에 저장된 grasp 정답은 사용하지 않았다. 입력 10,703점, diffusion 200 후보, top 100 반환. GraspMoE는 50개 추가 analytic OBB 후보와 diffusion을 discriminator로 점수화한다.

| 직접 공식 planner (warm 10회) | Mean ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| GraspMoE | 921.75 | 924.82 | 954.21 |
| Diffusion | 929.43 | 917.57 | 999.36 |

원시 기록: [sample.csv](outputs/benchmarks/sample.csv), [sample_summary.json](outputs/benchmarks/sample_summary.json). GraspMoE 마지막 측정의 score 범위 0.6104–0.9410, diffusion 0.5817–0.9051. 난수 샘플링에 따라 score는 달라진다.

| 실제 ZMQ 경로 (warm 10회, localhost) | Server handler mean ms | RTT mean ms | RTT P50 ms | RTT P95 ms |
|---|---:|---:|---:|---:|
| name / diffusion | 399.38 | 400.46 | 387.83 | 474.30 |
| params / graspmoe | 924.26 | 925.39 | 920.31 | 963.32 |
| params / diffusion | 957.53 | 958.64 | 974.29 | 995.46 |

**Name API와 planner API는 전처리 범위가 다르다.** params/direct planner에는 outlier filtering이 포함되므로 이 표를 모델 자체 속도의 공정한 planner 비교로 해석하면 안 된다. 위 RTT는 **서버 내 localhost**이며 Windows SSH 네트워크 성능이 아니다. 상세 기록: [network.csv](outputs/benchmarks/network.csv), [network_summary.json](outputs/benchmarks/network_summary.json).

공식 recorded RGB-D `real_world/00` replay는 실제 데이터의 obj_1 bounding ROI를 시작점으로 사용하고, 실제 mask는 구현한 ROI/depth 필터로 생성했다. Camera pose를 적용하지 않고 **camera frame**에 그대로 유지했다. Cloud 1,219점, 반환 100개, best score **0.99461**. [overlay PNG](outputs/rgb/official_rgbd_replay_overlay.png), [grasp JSON](outputs/grasps/official_rgbd_replay.json), [grasp NPZ](outputs/grasps/official_rgbd_replay.npz)를 저장했다. 이 score는 실제 물리적 grasp 성공을 뜻하지 않는다.

선택형 `infer_scene_depth`도 공식 녹화 depth/K/단일 instance mask로 검증했다. 720×1280 depth, instance 1, 250개 후보 반환. [scene_depth.json](outputs/benchmarks/scene_depth.json). 기본 UI는 segmented PC 경로를 유지한다.

## 9. GPU 메모리

| 실험 | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|
| 직접 GraspMoE sample | 836.61 | 970 |
| 직접 diffusion sample | 819.26 | 1018 |
| ZMQ RGB-D replay | 784.30 | 918 |

이는 **torch CUDA allocator 측정값**이며 CUDA context/driver 등 모든 프로세스 GPU 메모리를 뜻하지 않는다. 초기 sample 실행 중 nvidia-smi MIG 전체 사용량은 **1234MiB**로 관측했고 최종 서버 상주 상태는 **1182MiB**였다. [gpu-final.txt](outputs/benchmarks/gpu-final.txt)에 snapshot을 기록했다. 다른 사용자의 작업을 종료하거나 GPU/MIG 구성을 변경하지 않았다.

## 10. 단계별 시간

다음은 공식 녹화 RGB-D replay **1회**이며 실제 카메라 latency로 보고하지 않는다.

| 구간 | ms |
|---|---:|
| Dataset 파일 로딩 | 100.47 |
| ROI/depth segmentation | 55.66 |
| Point cloud 생성 | 5.20 |
| Downsampling | 4.24 |
| 클라이언트 localhost RTT | 440.73 |
| 서버 infer_ms (RTT 내부) | 438.14 |
| CUDA 동기화 handler (RTT 내부) | 438.61 |
| Visualization scene/overlay 생성 및 HTTP 확인 | 541.22 |
| 결과 저장 | 338.05 |
| Replay 전체 elapsed | 1489.41 |
| 실제 RGB-D acquisition / alignment | **미측정** |

[rgbd_replay.json](outputs/benchmarks/rgbd_replay.json). RTT와 서버 시간은 중첩된다. Visualization은 브라우저 frame rendering 완료를 측정한 값이 아니다. 첫 Viser 시작 비용이 포함된다. Windows에서는 capture/alignment/conversion 포함 각각의 타이머를 CSV에 기록하고 매 요청 후 mean/P50/P95를 출력한다. 서로 다른 PC의 시계를 빼지 않는다. 고정 capture 이후 인간 대기는 frame age로 분리한다.

## 11. 한계 및 후속 작업

- Windows 실제 RealSense 획득/지원모드 fallback/USB 재연결/GUI/브라우저 조작은 노트북에서 실행해야 한다. 이 원격 Linux에는 libusb-1.0 공유 라이브러리가 없어 Linux pyrealsense2 import도 완료하지 않았으며, USB 테스트를 했다고 주장하지 않는다. CPU client의 나머지 모듈과 원격 protocol은 torch 없는 환경에서 검증했다.
- 실제 노트북의 SSH 구간 RTT, camera latency, end-to-end P50/P95는 아직 없다. Windows에서 동일 capture G 반복 20회 이상 및 새로운 capture 여러 장으로 측정한다.
- 수동 ROI/depth mask는 같은 깊이의 배경, 반사/투명 물체, 가려짐에서 실패할 수 있다. Mask/cloud preview로 확인하고 near/far/band/percentile을 조정한다. SAM/YOLO는 설치하지 않았다.
- SDK가 역투영할 수 없는 modified_brown_conrady color 모델은 명시적 오류를 내며 rectified stream이 필요하다. 대부분의 무왜곡/SDK 지원 모델에는 실제 calibration을 사용한다.
- GraspMoE analytic 후보는 입력 Z를 기준으로 생성된다. 카메라 frame과 중력 frame을 동일하게 간주하지 않는다. 실제 시점에서 diffusion 비교가 필요하다.
- 모델 confidence는 충돌/도달 가능성/물리적 성공의 보장이 아니다. Scene collision filtering, robot TCP calibration, 제어는 구현 범위 밖이다.
- 실행 서버는 상주 프로세스이며 자동 재부팅 복구 서비스가 아니다. 요청 중 health는 같은 REP 서버 큐에서 기다릴 수 있다.
- 그림의 CAD는 공식 franka_umi 정적 열린 mesh이며 개별 candidate의 최적 jaw width나 실제 보유 그리퍼를 검증한 것은 아니다.

산출물: [Windows ZIP](dist/graspgenx-realsense-windows.zip), [SHA256SUMS](dist/SHA256SUMS), [테스트 로그](outputs/benchmarks/tests.log), [ZIP 검증](outputs/benchmarks/package_check.json), [설계 결정](docs/DECISIONS.md). ZIP은 모델/torch 환경을 포함하지 않는다. 실제 녹화 sample 추론 결과를 재생 예제로 포함한다.
