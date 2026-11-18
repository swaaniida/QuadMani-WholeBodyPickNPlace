# Tracking SAM 모드 — Windows 실행

기존 RealSense capture, depth calibration, cloud 전처리, GraspGenX 통신, Viser/CAD 및 저장 코드를 확장했다. 기존 `main.py`의 수동 ROI 모드도 유지한다. 새 모드는 **main_tracking.py**이다. Windows에 SAM, AOT, PyTorch, CUDA를 설치하지 않는다.

## 업데이트 및 실행 (PowerShell)

```powershell
cd $env:USERPROFILE
scp -P 2222 -i "$env:USERPROFILE\.ssh\id_ed25519" suhwan@astrago:/data/team/users/suhwan/grasp-pose/graspgenx-realsense/dist/graspgenx-realsense-windows.zip .
Expand-Archive .\graspgenx-realsense-windows.zip -DestinationPath .\graspgenx-realsense -Force
cd .\graspgenx-realsense\local_client
# 처음 설치할 때만 실행. 기존 .venv가 있으면 그대로 사용한다.
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

별도 PowerShell에서 터널을 열고 유지한다. 로컬에서 5556 바인딩이 거부된 이력이 있어 **15557**을 사용한다. Windows는 이 tracking 포트만 필요하고 GraspGenX로 큰 RGB-D를 다시 보내지 않는다.

```powershell
ssh -p 2222 -i "$env:USERPROFILE\.ssh\id_ed25519" -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:15557:127.0.0.1:5557 suhwan@astrago
```

첫 번째 창에서 D435i를 연결하고 실행한다.

```powershell
.\.venv\Scripts\python.exe tracking_client.py --health --port 15557
.\.venv\Scripts\python.exe main_tracking.py --port 15557
```

Health는 Tracking SAM service가 모델을 로딩한 뒤 `status: ok`를 반환한다. 서버 시작 직후에는 로딩 완료까지 기다린다. 별도의 GraspGenX 서버도 실행 중이어야 G 요청이 성공한다.

## 조작

- **마우스 왼쪽 클릭**: `Tracking SAM: matched RGB + aligned depth` 창의 **왼쪽 RGB 영역**에서 대상 클릭. 해당 표시 프레임을 SAM으로 초기 분할하고 이후 프레임은 AOTT VOS가 추적한다. 처음 SAM은 수 초 걸릴 수 있다.
- **G**: 네트워크 worker가 처리한 최신 유효 RGB-D/mask snapshot을 고정해 서버에서 GraspGenX `infer_scene_depth()` 실행. 한 건만 허용한다. 추론 중 추적과 LIVE capture가 계속된다.
- **V**: 최근 grasp의 실제 cloud+공식 franka_umi CAD+top 10+score+camera axes를 브라우저에서 표시. Grasp가 없으면 현재 유효 tracked cloud를 표시한다.
- **R**: 추적/선택 초기화. 이후 새로 클릭한다. 과거 실행 중인 grasp 결과는 generation으로 폐기한다.
- **S**: 동일 시점 RGB, raw depth, meter depth, K/scale, raw tracking mask와 filtered mask 및 calibration/config 저장.
- **Q / Esc**: 종료.

`RealSense LIVE` 창은 카메라 원본 최신 프레임이다. Tracking 창은 **모델 응답과 같은 RGB-D 프레임**을 표시하므로 그만큼 늦을 수 있다. status, ID 1, depth retained 비율, tracking FPS, frame age, SAM/VOS/grasp 시간을 함께 표시한다. Grasp overlay는 별도의 **촬영 프레임에 고정된 창**에 표시한다. 과거 grasp를 현재 영상에 겹치지 않는다.

필터에서 **초록색은 유지된 target mask**, **빨간색은 depth 필터 때문에 제외된 부분**이다. Near/Far/Band(mm)를 조절한다. 기본 Band=0으로 SAM이 선택한 물체의 깊이 폭을 임의로 잘라내지 않는다. Voxel은 표시/저장 cloud downsampling에 적용되고, scene-depth 모델 내부 point sampling은 공식 GraspGenX가 처리한다. 물체 선택에 ROI를 다시 요구하지 않는다.

Mask가 비거나 너무 작음/갑작스러운 면적 변화/시간적 overlap 상실/카메라 재연결/해상도 변경이면 추적을 lost로 latch하여 G를 막는다. **재클릭 또는 R 후 재선택**한다. 단순 geometry heuristic이므로 identity drift를 완전히 검출하는 semantic confidence는 아니다. 이미지와 mask를 확인해야 한다. Depth가 부족한 경우에는 추적을 유지하되 G를 막는다.

## 옵션

```powershell
# 실제 GroundingDINO → SAM → AOTT 텍스트 초기화 (시작 프레임에 1회)
.\.venv\Scripts\python.exe main_tracking.py --port 15557 --text "mustard bottle"
# 같은 영상에서 diffusion 비교
.\.venv\Scripts\python.exe main_tracking.py --port 15557 --planner diffusion --output outputs\tracking_diffusion
# 전송 빈도 / threshold / 서버 반환 top-K
.\.venv\Scripts\python.exe main_tracking.py --port 15557 --fps 8 --threshold 0.5 --topk 100
```

Text 초기화는 공식 코드의 단일 bbox 제한을 따른다. 검출이 0개 또는 여러 개이면 오류를 표시하고 클릭 초기화를 사용한다. 텍스트 모델을 SAM 단일 이미지 반복으로 바꾸지 않는다. 이후 추적은 동일한 AOTT 엔진이다. 3D 표시는 가독성을 위해 top 10으로 유지한다.

## 좌표·통신·저장

- 카메라 optical frame X 오른쪽/Y 아래/Z 전방, meter. T_camera_gripper_base이며 로봇 TCP가 아니다.
- RGB JPEG(quality 90), 원본 uint16 depth의 zlib lossless 압축, 실제 scale/K, frame ID/session/sequence를 한 메시지로 전송한다. Mask는 PNG, background 0/target 1이다.
- 서버는 timestamp가 다른 mask/depth를 합치지 않는다. Grasp worker는 frame ID와 generation이 같은 snapshot을 사용하며 최신이 아니거나 오래된 snapshot은 거부한다.
- 공식 scene-depth API는 **rectified pinhole K/depth**를 가정한다. 카메라가 nonzero distortion 계수를 보고하면 G를 명시적으로 거부한다. 현재 지원 범위는 aligned frame이 무왜곡/rectified인 경우이다. 왜곡이 남은 frame을 pinhole로 잘못 해석하지 않는다.
- 기본 전송 상한은 10FPS이다. 카메라는 지원 가능한 모드(기본 640×480@30)로 계속 캡처한다. 추론/전송 속도가 낮으면 중간 frame을 건너뛰고 최신 frame만 보내며 무한 queue를 만들지 않는다. 큰 물체 이동/네트워크 지연으로 VOS가 놓치면 재초기화한다.
- `outputs/tracking/`에 tracking_frames.csv, grasp_requests.csv, captures/, rgb/, depth/, pointcloud/, grasps/를 저장한다. 기존 manual 모드 출력과 구분한다.
- frame CSV는 capture/alignment/conversion, JPEG/lossless encoding, 전송 byte 수, track RTT, service 시간, SAM 초기화/VOS, mask-depth 및 frame age를 기록한다. `RTT-service_ms`는 전송·직렬화 등의 합계 추정이며 순수 one-way network latency가 아니다.
- grasp CSV는 server inference, 서버 내부 RPC RTT, 결과 poll RTT, cloud/시각화/저장, G 요청부터 완료까지를 구분한다. 서로 다른 PC의 시계를 빼지 않는다. 서버/RTT 구간은 중첩되므로 더하지 않는다. SAM 초기화 시간은 AOT reference 준비도 포함한다. Text 초기화 시간은 DINO+SAM+reference 합계이다.
- viewer GPU rendering 완료는 측정하지 않는다. CPU에서 scene/overlay를 만든 시간을 기록한다. Confidence는 물리적 성공·충돌 안전성·로봇 도달 가능성 보장이 아니다.

## Windows에서 남은 실제 하드웨어 검증

1. D435i 모델/serial/640×480@30 또는 fallback/실제 intrinsics/scale 확인.
2. 클릭하여 초기 mask 확인 후 물체/카메라를 움직이며 연속 추적 확인.
3. G 중 LIVE와 mask가 갱신되는지, 결과 frame ID가 저장 RGB-D/mask와 일치하는지 확인.
4. V에서 cloud/CAD/grasp/score 확인 및 S 저장.
5. 물체가 화면 밖으로 나가거나 USB/터널이 끊겼을 때 G 차단과 R/재클릭 복구 확인.
6. 여러 촬영 시점에서 CSV mean/P50/P95 확인.

서버에서 실제 모델의 공식 동영상 추적과 **공식 녹화 RGB-D의 static replay 통합**을 검증했지만, Windows USB/실제 움직이는 D435i RGB-D/GUI 조작/SSH RTT를 검증한 것은 아니다. 이것이 최종 하드웨어 완료 기준이다.

Grasp 후보 선택 및 S#1/C#1 비교: [GRASP_SELECTION.md](GRASP_SELECTION.md). 모델 추론은 그대로이며 결과 후처리에 카메라 방향 선호도를 적용합니다.

자동 반복 추론, Top-1 기록, 프레임별 RGB-D/Mask 보존 및 종료 후 재생은 [AUTO_GRASP.md](AUTO_GRASP.md)를 참고하세요. `main_tracking.py --port 15557 --auto-grasp --duration 60`으로 실행합니다.
