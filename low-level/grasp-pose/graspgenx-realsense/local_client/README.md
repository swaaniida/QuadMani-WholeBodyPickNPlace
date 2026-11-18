# Windows RealSense → GraspGenX 클라이언트

**Tracking SAM 모드 추가:** 클릭 초기화 → AOTT 연속 추적 → G scene-depth grasp는 [TRACKING_README.md](TRACKING_README.md)의 `main_tracking.py`를 사용한다. 이 문서의 기존 `main.py` 수동 ROI 기능도 유지한다.

Python 3.11 x64 권장. PyTorch/CUDA/GraspGenX 설치가 필요 없다. 서버 모델은 astrago에서만 실행한다. 이 폴더 전체를 복사하거나 Windows ZIP을 풀어서 사용한다.

## 1. 설치 (PowerShell)

```powershell
cd .\local_client
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

현재 요구사항은 numpy 1.26.4를 사용하므로 Python 3.13 이상은 사용하지 않는다. RealSense를 USB 3 포트에 연결하고 카메라를 점유한 다른 프로그램을 닫는다.

## 2. SSH 터널 (별도 PowerShell 창)

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:5556:127.0.0.1:5556 suhwan@astrago
```

현재 사용 중인 SSH host alias/포트/ProxyJump 설정을 그대로 사용한다. 별도 포트가 필요하면 `ssh -p 실제SSH포트 ...`를 추가한다. 서버에서 관측한 sshd listen 포트는 2222이지만 외부 접속 포트와 동일하다고 단정하지 않는다. `astrago`가 SSH config alias라면 기존 alias를 사용한다. 터널 창을 계속 켜둔다. 5556을 방화벽에서 공개할 필요는 없다.

## 3. 연결 확인 및 실행

```powershell
.\.venv\Scripts\python.exe grasp_client.py --timeout-ms 5000
.\.venv\Scripts\python.exe main.py
```

첫 명령은 health `status: ok`, gripper `franka_umi`, ptv3vanilla 모델, DDPM 20 steps를 출력해야 한다. 기본 추론은 params 기반 공식 `infer_object`, planner `graspmoe`, diffusion 200 후보, top 100 반환이다. GraspMoE의 추가 OBB 후보도 discriminator로 scoring한다.

비교 실험은 서로 다른 출력 폴더에 저장한다.

```powershell
.\.venv\Scripts\python.exe main.py --api params --planner diffusion --output outputs_diffusion
.\.venv\Scripts\python.exe main.py --api name --planner diffusion --output outputs_name
```

공식 name-based `infer`는 planner 인자가 없고 diffusion만 실행한다. `--api name --planner graspmoe`는 조용히 다른 동작을 하지 않고 오류로 거부한다. params 방식은 동봉한 **공식 franka_umi 설정에서 추출한 값**을 사용한다.

## 4. 조작

- **S**: 현재 RGB-D 스냅샷을 고정하고 RGB에서 ROI를 드래그한다. Enter/Space로 확정, C로 취소한다. 선택 중에도 카메라 획득 스레드는 계속 동작한다.
- **G**: 선택한 고정 프레임을 다시 필터링하고 서버에 추론 요청한다. 동시에 1건만 허용하며 추가 G는 무시한다. LIVE 미리보기는 계속 갱신된다.
- **V**: 선택 cloud 또는 최근 결과를 로컬 3D 브라우저 `http://127.0.0.1:8080`에서 연다. 드래그로 회전, 휠로 확대한다. 추론 전에도 cloud 확인이 가능하다.
- **R**: 선택과 표시를 초기화한다. 이미 실행 중인 서버 계산은 취소하지 않지만 해당 결과는 폐기한다.
- **Q / Esc**: 종료. 진행 중인 네트워크 요청은 timeout까지 종료를 지연시킬 수 있다.

LIVE와 별도로 `Selected capture...` / `GRASP RESULT...` 창을 표시한다. 과거 결과를 현재 LIVE 영상에 덧그리지 않는다. 모든 선택/결과에는 serial, 연결 세션, camera frame number로 만든 capture ID가 있다. 새 물체/시점을 찍으려면 S를 다시 누른다. G만 반복하면 같은 캡처에 대한 반복 추론이다.

## 5. Depth 필터

트랙바 단위는 mm이다. 실제 저장/통신 단위는 meter이다.

- near/far: 유효 depth의 절대 범위.
- foreground percentile: ROI 유효 depth에서 기준 깊이를 고르는 percentile (기본 10).
- band: 기준 깊이보다 뒤로 허용할 깊이 폭 (기본 80mm). 0은 band 제한을 끈다.
- voxel: metric voxel 크기 (기본 3mm). 0은 voxel 필터를 끈다.

기본적으로 depth band 안의 가장 큰 연결 영역을 남긴다. 가까운 잡음/배경에 mask가 잡히면 ROI를 좁히거나 near/far, percentile, band를 조절한다. 물체와 배경 깊이가 비슷하면 이 수동 방식만으로 분리되지 않을 수 있다. ROI 안 모든 점이 실제 대상이라는 보장은 없다. 3D cloud와 mask를 먼저 확인한다. downsample 후 최소 100점, 기본 최대 12,000점이다. `--max-points`로 변경한다.

## 6. 좌표계, intrinsics, 표시

RealSense optical frame: X 오른쪽, Y 아래, Z 전방. depth는 color에 정렬한 뒤 **aligned depth의 intrinsics**로 역투영한다. RGB overlay는 color intrinsics로 투영한다. 실제 sensor depth scale을 읽고 미터로 변환한다. 왜곡이 있으면 SDK의 보정된 unit ray를 캐시하며 overlay에도 SDK 투영을 사용한다. SDK가 역투영할 수 없는 modified_brown_conrady 모델은 명시적으로 오류를 내며 rectified stream이 필요하다.

데이터를 임의 회전하거나 크기 정규화하지 않는다. Viser의 up-direction만 화면 조작을 위해 -Y로 설정한다. 결과 행렬은 `T_camera_gripper_base`이다. TCP/로봇 EE로 변환하지 않는다. CAD는 공식 franka_umi `vis_mesh.obj`의 정적 열린 자세이며, 실제 보유한 그리퍼와 일치하는지 보장하지 않는다.

3D에는 camera axes, 물체 cloud, 상위 10개의 grasp axes, confidence, 공식 sweep wireframe 및 공식 CAD가 표시된다. 최고 score는 금색/굵은 선이다. CAD가 없거나 로딩 실패하면 UI에 **CAD 없음 / wireframe only**라고 표시한다. 오른쪽 GUI에서 CAD를 끄고 행렬과 quaternion을 볼 수 있다. JSON의 quaternion 순서는 **xyzw**, Viser GUI는 **wxyz**라고 각각 표시한다.

GraspMoE OBB branch는 입력 frame의 Z축을 기준으로 후보를 만든다. 카메라 Z가 중력 방향이라는 가정은 하지 않는다. 카메라 자세에 따른 후보 편향을 diffusion과 비교해야 한다. confidence는 discriminator score이며 실제 grasp 성공 확률/충돌 회피/로봇 도달 가능성 검증이 아니다.

## 7. 저장 및 latency

`outputs/` 아래에 RGB 원본/overlay PNG, raw depth + depth_m + K + scale NPZ, mask PNG, cloud/color NPZ, grasp NPZ 및 JSON(모든 행렬, 위치, xyzw quaternion, confidence, branch, calibration, filter 설정)을 저장한다. 파일 이름은 UTC run ID이고 metadata에는 원래 camera timestamp/프레임 ID를 보존한다.

```powershell
.\.venv\Scripts\python.exe recording.py outputs\benchmarks\latency.csv
.\.venv\Scripts\python.exe view_saved.py outputs\grasps\실제결과파일.npz
.\.venv\Scripts\python.exe view_saved.py example_results\official_rgbd_replay.npz
```

마지막 예제는 **공식 녹화 RGB-D 데이터에 실제 A100 모델 추론을 실행한 저장 결과**의 재생이다. 현재 USB 카메라 실험이 아니며 재생 시 새 추론을 하지 않는다.

CSV는 acquisition(wait_for_frames), alignment, conversion/ray 준비, segmentation, cloud generation, downsampling, 클라이언트 network RTT(직렬화+전송+서버+수신), server infer, CUDA 동기화된 handler 전체 시간, visualization(브라우저 scene 생성+overlay), save, G 요청부터 파일 저장 완료까지 request_total을 분리한다. 각 단계 mean/P50/P95를 출력한다.

`request_total_ms`는 G 이후 처리 시간으로 인간의 ROI 선택 대기와 이미 완료한 capture 시간을 포함하지 않는다. `frame_age_at_request_ms` 및 `capture_to_result_ms`는 선택한 고정 프레임의 나이/인간 대기를 포함한다. acquisition/alignment는 **해당 선택 프레임에서 측정된 별도 구간**이다. RTT 안에 서버 추론이 포함되므로 둘을 더하지 않는다. 서로 다른 PC의 시계를 빼지 않는다. visualization 시간은 브라우저 GPU 렌더 완료 시간은 아니다. 공식 CLI를 직접 쓰면 synchronized handler/GPU 필드는 비어 있고 infer_ms만 제공된다.

## 8. Windows에서 남은 검증

1. 실행 시 실제 모델명/serial, 지원 depth/color mode, 활성 mode, depth scale, aligned/color intrinsics 출력 확인.
2. 기본 640×480@30 또는 지원 가능한 fallback mode에서 RGB/depth가 모두 보이는지 확인.
3. 물체와 배경이 분리된 mask, meter 단위 cloud를 V로 확인.
4. G 추론 중 LIVE 갱신/중복 요청 억제 및 frame ID 일치 확인.
5. Top 10 CAD/pose/score, 최고 후보 강조, RGB axes 및 저장 JSON/NPZ/CSV 확인.
6. 반복 G 20회 이상 수행해 평균/P50/P95 확인. S로 새 프레임도 테스트.
7. 카메라 USB 분리/재연결, 터널 단절/복구, 잘못된 필터에서 오류 복구 확인.

이 Windows USB/GUI 검증은 원격 서버에서 실행되지 않았다. 카메라 재연결은 2초 간격으로 시도하며 원래 요청한 serial이 있으면 동일 serial만 선택한다.

## 참고 소스

- [GraspGenX standalone protocol](https://github.com/NVlabs/GraspGenX/tree/44eb1d77d6d27bc97ee361e3afb717a2115e30a9/client-server)
- [RealSense alignment example](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py)
- `official_viser_helpers.py`: 공식 Viser 함수만 변경 없이 추출; SPDX/출처 보존. 모델을 복제하거나 추론을 재구현하지 않는다.
- `assets/`: 공식 franka_umi mesh/config/sweep 및 provenance. 원본 안내와 GraspGenX LICENSE는 동봉한다.
