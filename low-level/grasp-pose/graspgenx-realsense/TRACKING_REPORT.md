# Tracking SAM 통합 실행 보고

실행일: 2026-11-11 UTC. 기존 `/data/team/users/suhwan/grasp-pose/graspgenx-realsense` 프로젝트를 확장했다. 기존 수동 ROI 기능, RealSense capture, metric cloud/depth 전처리, GraspGenX 서버, Viser/CAD 및 저장 코드를 재사용했다. 로봇 제어/정책 학습은 하지 않았다.

**서버 구현 및 데이터셋 검증은 완료했다. 최종 live D435i 하드웨어 완료 기준은 아직 충족하지 않았다.** 원격 서버에서는 Windows USB에 접근할 수 없다. 실제 촬영한 연속 RGB-D와 Windows GUI/SSH latency를 [Windows 절차](local_client/TRACKING_README.md)로 확인해야 한다.

## 실제 구현

- 공식 SAM ViT-H 클릭 초기화 → 공식 AOTT VOS reference/memory propagation. 매 프레임 SAM 단독 추론으로 대체하지 않았다.
- 공식 GroundingDINO → SAM → AOTT의 optional text 초기화. `mustard bottle`을 실제 녹화 이미지에서 검증했다.
- Windows는 기존 `RealSenseCapture`를 사용해 RGB/align depth/실제 K/scale을 한 frame ID에 묶는다. JPEG RGB와 lossless uint16 depth를 Tracking service에만 한 번 보낸다.
- Tracking service는 localhost 5557, GraspGenX는 기존 localhost 5556이다. 두 환경/모델은 독립 프로세스이다.
- 서버가 mask를 bg=0/target=1로 만들고 depth 필터를 적용한다. G는 동일 프레임의 depth/K/instance mask를 **공식 infer_scene_depth()**로 전달한다.
- 단일 asynchronous grasp worker가 호출하므로 tracking/capture가 계속된다. 서버 snapshot은 최대 8개, grasp job은 1개이며 새 capture는 latest-frame 정책으로 중간 프레임을 건너뛴다.
- 세션/순서/frame ID/generation 확인, stale/비최신 snapshot 거부, reset 이후 과거 결과 폐기, mask loss latch 및 invalid-depth G 차단을 구현했다.
- 클릭/G/V/R/S/Q UI, raw tracking mask(초록)와 depth 제외 부분(빨강), ID/status/FPS/frame age/각 모델 시간 표시, cloud/top 10/score/best highlight/공식 franka_umi CAD/camera axes 및 capture-matched RGB grasp overlay를 제공한다.
- frame CSV와 grasp CSV를 분리한다. Capture/alignment/encoding/RTT/SAM/VOS/mask-depth/grasp/시각화/저장/전체 요청 시간을 기록한다.

## 설치 및 실제 API

| 구성 | 확인한 소스/버전 |
|---|---|
| Tracking SAM | [RogerQi/Tracking_SAM](https://github.com/RogerQi/Tracking_SAM), commit `d5e5e81a361dce4ddbe3264a2b8027b2b43ea3fd` |
| SAM | facebookresearch/segment-anything, commit `dca509fe793f601edb92606367a655c15ac00fdf`, `vit_h` |
| AOT submodule | `ada8a3cbf0ba6dde563a49e78e56dbbcde01d143`, `aott`, `pre_ytb_dav` |
| Correlation CUDA | `14a159ebad5399adf5db965bb5cab095c98ebc87` |
| GroundingDINO | `60d796825e1266e56f7e4e9e00e88de662b67bd3`, SwinT OGC |
| Tracking 환경 | `.venv-tracking`, Python 3.10.22, torch 1.13.1+cu117, torchvision 0.14.1+cu117, NumPy 1.26.4 |
| 기존 GraspGenX | commit `44eb1d77d6d27bc97ee361e3afb717a2115e30a9`, `.venv`, torch 2.5.1+cu121 |
| 장치 | A100-SXM4-80GB의 MIG 3g.40gb, 40192MiB, 42 SM |

공식 INSTALL.md는 Python 3.8 환경을 안내한다. 기존 프로젝트 내부 Python 3.10 runtime으로 별도 환경을 구성하고 실제 공식 demo로 호환성을 검증했다. 시스템 driver/CUDA와 기존 GraspGenX 의존성은 변경하지 않았다. CUDA 11.7/gcc 9.4로 correlation과 GroundingDINO 확장을 SM80에 맞춰 빌드했다. 상세 [tracking-freeze.txt](configs/tracking-freeze.txt), [빌드 로그](outputs/tracking/install-extensions.log).

소스에서 확인한 streaming 경로는 `main_tracker.sam_predictor.set_image()/predict()` → `vos_tracker.add_reference_frame()` → 후속 `main_tracker.propagate_one_frame()` → AOT의 `match_propogate_one_frame()/decode_current_logits()/update_memory()`이다. 원격 adapter는 GUI 클릭 좌표만 socket 입력으로 바꾸며 VOS를 재구현하지 않았다. Text 경로는 공식 `annotate_init_frame(method='dino')`를 사용한다. VBC [visual_wholebody](https://github.com/Ericonaldo/visual_wholebody) 참고 소스를 확인했지만 로봇/controller 코드는 통합하지 않았다.

Checkpoint는 공식 MODEL_PREP.md의 링크에서 받았다.

| 파일 | bytes | SHA-256 |
|---|---:|---|
| sam_vit_h_4b8939.pth | 2564550879 | `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e` |
| AOTT_PRE_YTB_DAV.pth | 23332161 | `157b12b2fe6a465f38dafb84992b47b7a0100c534328e9888224f083bc7b03d3` |
| groundingdino_swint_ogc.pth | 693997677 | `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799` |

[checksums](outputs/tracking/checkpoints.json), [MODEL_PREP.md](upstream/Tracking_SAM/MODEL_PREP.md). GraspGenX release generator/discriminator, `franka_umi` sweep/visual assets와 TCP 구분은 기존 [REPORT.md](REPORT.md)와 같다.

## 실제 검증 결과

### 1. 공식 demo.py — 실제 움직이는 동영상

원격 Linux에 GUI 디스플레이가 없어 **공식 demo.py main()을 실행하되 OpenCV 표시와 matplotlib 클릭 입력만 headless harness로 자동화**했다. 클릭은 공식 첫 annotation 안의 `(311,290)`이다. **GT mask를 AOT reference로 넣지 않았다.** 실제 SAM 결과를 reference로 사용했다.

- 실제 DAVIS bear **82프레임**, 초기 SAM 1회 + AOTT propagation **81회**.
- 최초 SAM mask의 annotation 대비 IoU **0.9700006**.
- 초기화 pipeline cold time **12,123ms** (SAM/첫 CUDA 호출/AOT reference/annotator 렌더링 포함).
- 첫 VOS 호출 제외 80개 구간: mean **54.46ms**, P50 **63.75ms**, P95 **68.06ms**.
- VOS 연산 기준 **18.36FPS**. Capture/네트워크/GUI가 포함된 Windows live FPS를 뜻하지 않는다.
- Torch allocator peak **6414.87MiB**.

[원시 CSV](outputs/tracking/official_demo/frames.csv), [summary](outputs/tracking/official_demo/summary.json), [최종 실제 추적 overlay](outputs/tracking/official_demo/last_overlay.png), [실행 로그](outputs/tracking/official-demo.log).

### 2. Tracking mask + 실제 녹화 RGB-D + 실제 GraspGenX

공식 GraspGenX `real_world/00`의 실제 RGB와 meter depth/K를 사용했다. 한 장의 RGB-D를 반복하는 **static replay**이며 움직이는 D435i RGB-D 촬영 결과가 아니다. 가짜 depth나 가짜 grasp를 만들지 않았다.

- 녹화 RGB의 mustard bottle에 실제 클릭 `(620,450)` → SAM mask → 후속 AOTT propagation.
- 추적 mask로 추출한 cloud **1179점**.
- 공식 scene-depth API → **100개** SE(3)/confidence 반환. shape/finite/SE(3)/score 범위 검증.
- Grasp 요청 프레임 ID: `official-static-rgbd-00006`; 반환/저장/overlay와 동일함을 확인.
- Grasp가 실행되는 동안 **4개** 추가 tracking 프레임 처리 (최종 재검증 기준).
- GraspGenX infer_ms **665.43ms**, synchronized handler **683.23ms**, 내부 RPC RTT **693.92ms**.
- G submit부터 3D scene/overlay 및 파일 저장까지 **1217.39ms**. localhost의 1회 replay 값이다.
- 3D scene: 실제 공식 CAD 10개 + cloud/axes/score, HTTP 200. 브라우저 수동 조작은 검증하지 않았다.
- reset 후 G 거부 확인. Optional `mustard bottle` text 초기화는 실제 mask **7113픽셀**, status tracking.

[통합 summary](outputs/tracking/integration_summary.json), [실제 overlay](outputs/tracking/rgb/integration_static_rgbd_overlay.png), [grasp NPZ](outputs/tracking/grasps/integration_static_rgbd.npz), [grasp JSON](outputs/tracking/grasps/integration_static_rgbd.json). 저장 JSON의 `source`는 official static replay라고 명시한다.

해당 최종 run의 [frame CSV](outputs/tracking/integration_frames.csv), [grasp CSV](outputs/tracking/integration_grasps.csv)도 저장했다.

### 3. 실제 Windows networking worker 검증

`TrackingWorker`를 PyTorch 없는 `.venv-client`에서 실행하고, 카메라 provider만 위 실제 녹화 데이터의 static replay로 연결했다. 실제 두 모델/서비스를 사용해 G와 추적 동시 진행, frame ID 일치, 100개 반환, reset 및 torch 미import를 검증했다. 단순 mock 서버 결과를 사용하지 않았다.

[worker summary](outputs/tracking/worker_summary.json), [frame CSV](outputs/tracking/worker_replay/tracking_frames.csv), [latency mean/P50/P95](outputs/tracking/latency_summary.json). 정적 replay/local transport의 작은 표본이며 live/SSH benchmark는 아니다. SAM 초기화와 VOS를 별도 column에 기록한다.

### 4. A100 두 모델 동시 구동

Tracking SAM/AOTT/GroundingDINO와 GraspGenX가 동시에 resident인 상태에서 실제 replay의 tracking+G를 실행했다. MIG 전체 사용량의 0.2초 간격 표본에서 최대 **8859MiB (약 8.65GiB)**를 관측했다. 총 **40192MiB** 내에서 동시 실행에 성공했다. 샘플링은 짧은 peak를 놓칠 수 있으므로 정확한 순간 최대치가 아니라 관측 최대치이다. 각 모델 allocator peak와 MIG total은 범위가 다르다.

[GPU summary](outputs/tracking/gpu_summary.json), [samples CSV](outputs/tracking/gpu_samples.csv), [nvidia-smi snapshot](outputs/tracking/gpu-final.txt).

### 5. 자동 테스트 및 기존 기능

기존 geometry/overlay/timeout 8개에 lossless depth/float recorded depth/K/shape/mask PNG/depth mask 제한/lost latch 검증을 추가해 **14개 테스트 통과**. 기존 GraspGenX 서버는 종료하거나 재설치하지 않았고 실제 health/추론을 다시 확인했다.

최종 Windows ZIP(약 3.6MiB)을 임시 폴더에 풀어 **29개 파일 SHA-256/ZIP CRC**, torch 없는 standalone import, 실제 Tracking service health, 실제 저장 grasp 100개와 공식 CAD 10개의 scene 및 HTTP 200을 확인했다. [패키지 검증](outputs/tracking/package_check.json), [테스트 로그](outputs/tracking/tests.log), [Windows ZIP](dist/graspgenx-realsense-windows.zip), [ZIP checksum](dist/SHA256SUMS).

## 서버·Windows 실행 명령

두 서버 모두 보고 시점에 실행 중이다. 중복 실행하지 않는다. 재기동이 필요할 때만 각각 실행한다.

```bash
cd /data/team/users/suhwan/grasp-pose/graspgenx-realsense
bash scripts/start_server.sh           # GraspGenX, 127.0.0.1:5556
# 별도 터미널
bash scripts/start_tracking_server.sh  # Tracking SAM, 127.0.0.1:5557
```

Windows PowerShell 터널:

```powershell
ssh -p 2222 -i "$env:USERPROFILE\.ssh\id_ed25519" -N -o ExitOnForwardFailure=yes -L 127.0.0.1:15557:127.0.0.1:5557 suhwan@astrago
```

업데이트한 ZIP을 풀고 기존 client 폴더에서:

```powershell
.\.venv\Scripts\python.exe tracking_client.py --health --port 15557
.\.venv\Scripts\python.exe main_tracking.py --port 15557
```

새 Windows 패키지/다운로드/설치/optional text 및 diffusion 비교는 [TRACKING_README.md](local_client/TRACKING_README.md)에 정리했다. Requirements는 CPU-only이며 기존 Windows 환경을 재사용할 수 있다.

## 남은 검증 및 한계

- **필수 하드웨어 완료 기준 미검증**: 실제 D435i 연속 촬영 → 움직이는 물체 추적 → 동일 RGB-D/mask의 grasp → Windows 시각화. 이 작업을 했다고 주장하지 않는다.
- Windows capture/alignment/USB 재연결/GUI 조작과 SSH 구간 latency/FPS는 아직 실측값이 없다. 코드에서 측정·저장하도록 구현했고 실제 Windows에서 검증해야 한다.
- Tracking loss 검출은 empty/area/temporal overlap 등의 보수적 heuristic이다. 배경으로의 semantic identity drift를 완전히 검출하지 못한다. Lost가 감지되면 G를 차단하고 재클릭을 요구한다.
- Text 초기화는 공식 코드의 단일 detection 제한(0개/여러 개이면 오류)을 따른다. 클릭 기반 경로가 우선이다.
- 공식 scene-depth API는 rectified pinhole K/depth를 가정하므로 nonzero distortion 계수가 남아 있으면 G를 명시적으로 거부한다. 무보정 pinhole로 잘못 추론하지 않는다. 추가 rectification은 후속 지원 항목이다.
- 동일 카메라에서 매우 빠른 움직임/큰 network delay로 프레임을 많이 건너뛰면 VOS가 놓칠 수 있다. 주기 자동 grasp update는 아직 활성화하지 않았으며 단일 bounded 요청 인터페이스를 재사용해 추가할 수 있다.
- Voxel은 표시/저장 cloud에 적용된다. Scene-depth 모델의 sampling은 공식 API가 담당한다. Band 기본값은 0으로 불필요한 물체 절단을 줄였다.
- GraspMoE의 OBB 후보는 입력 camera Z에 의존한다. 중력 정렬을 가정하지 않는다. 모델 gripper base는 실제 PiPER/UMI TCP가 아니며 confidence는 도달성/충돌/성공 보장이 아니다.
- One-way 업로드/다운로드 latency는 측정하지 않았다. RTT, service duration, encoding, payload bytes, result poll RTT를 구분한다. 서로 다른 PC clock을 빼지 않는다.

모든 기능은 기존 프로젝트 내부에 추가했고 기존 수동 ROI `main.py`도 남겨 두었다. 실행 결과/지연/메모리/하드웨어 완료 여부는 위 로그로 구분한다.
