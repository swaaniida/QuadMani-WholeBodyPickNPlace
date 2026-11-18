# Continuous background ICP + ground filtering + temporal selection

기존 `main_tracking.py`, `auto_grasp.py`, `grasp_ranking.py`, `view_session.py`, Tracking SAM/AOTT 모델 및 GraspGenX 서버/체크포인트는 변경하지 않았다. 신규 기능은 **별도 실행 진입점**으로만 켠다. 기존 명령으로 Baseline을 다시 실행할 수 있다. 전체 후보 전달 업데이트는 Tracking 서비스의 `topk=-1` 지원과 해당 서비스 재시작이 필요하다. 새 모델 설치는 필요 없다.

## Windows 업데이트 / 실행

기존 프로그램을 종료하고 실행한다. SSH 터널은 기존 localhost15557→서버5557을 유지한다.

```powershell
Set-Location "$env:USERPROFILE\graspgenx-realsense\local_client"
scp -P 2222 -i "$env:USERPROFILE\.ssh\id_ed25519" suhwan@astrago:/data/team/users/suhwan/grasp-pose/graspgenx-realsense/dist/grasp-temporal-update.zip .
Expand-Archive .\grasp-temporal-update.zip -DestinationPath . -Force -ErrorAction Stop
.\.venv\Scripts\python.exe -m pip install -r requirements-temporal.txt
.\.venv\Scripts\python.exe main_temporal.py --port 15557 --fps 15 --duration 6
```

클릭하면 새 녹화를 시작한다. R은 현재 녹화를 닫고, 다음 클릭에 새로운 세션을 시작한다. Q로 종료하면 비교 화면을 연다. 신규 모드에서 `--duration`은 첫 **선택 완료** 이후의 시간이며 No Valid Grasp도 포함한다. 지면을 못 찾았다고 타이머가 무한 대기하지 않는다. `--no-review`는 기록만 한다. 기존 Near/Far/Band/Voxel과 단일 진행 요청 구조를 재사용한다. 신규 진입점은 Near=0이나 Near>=Far일 때 마지막 유효 설정을 유지하며 SAM을 리셋하지 않는다.

기존 방식만 실행:

```powershell
.\.venv\Scripts\python.exe main_tracking.py --port 15557 --auto-grasp --duration 6
```

이미 생성된 비교 세션 재생:

```powershell
.\.venv\Scripts\python.exe view_temporal.py "outputs\tracking_temporal\auto_sessions\실제_세션_폴더"
```

## 처리 구조 / 좌표와 시간

- 같은 Tracking 응답의 RGB-D/Mask를 재사용한다. 큰 이미지를 서버로 추가 전송하지 않는다. CPU Open3D 스레드가 Tracking 프레임마다 배경 ICP를 실행하고, A100 GraspGenX inference와 병렬로 진행한다. Windows에 CUDA/PyTorch는 필요 없다.
- 배경은 **대상 SAM mask를 11픽셀 dilation한 영역을 제외**한다. 유효 Depth에서 최대12,000개를 균등 선택하고 2cm voxel로 줄인다. 다른 움직이는 사람/물체는 자동 검출하지 않는다. 배경이 정적이어야 한다.
- ICP 큐는 최대1개로 최신 프레임을 유지한다. 밀린 프레임은 버리고 drop 수를 기록한다. 15Hz는 업로드 목표/상한이며 달성 FPS가 아니다.
- Point-to-plane ICP는 current→previous 방향이다. `T_world_current = T_world_previous @ T_previous_current`. 세계 좌표는 첫 처리 프레임의 카메라 광학 좌표이다.
- Fitness>=0.55, RMSE<=18mm, 상대 이동<=25cm, 회전<=25도, 배경 normal 다양성을 검사한다. 단일 평면으로 6-DoF가 불확정이면 높은 fitness만으로 승인하지 않는다.
- 10개 승인 프레임마다 keyframe을 만들고 최대6개를 유지한다. 5개 승인 프레임마다 과거 keyframe에 직접 재정합한다. 상대 ICP 결과와 차이가 5cm/10도 이상이면 거부한다. 국소 재앵커링이며 전역 SLAM/loop closure나 drift 제거 보장은 아니다.
- **Grasp의 원본 Frame ID**로 해당 촬영 프레임의 ICP transform을 조회한다. 현재/최신 카메라 pose로 대체하거나 보간하지 않는다. 최대150ms를 기다려도 결과가 없으면 Temporal을 사용하지 않는다. 센서 timestamp를 우선 사용하고, 없으면 host capture timestamp, 마지막으로 같은 클라이언트 ready monotonic을 사용한다. 서로 다른 PC 시각을 빼지 않는다.

## 지면 및 geometry

- 객체 주변 배경에서 RANSAC으로 최대4개 평면을 찾는다. 최소250점/12% inliers, RMSE<=6mm, 객체가 평면의 자유 공간 쪽에 위치하는지, 객체의 평면 투영이 관측된 지지면 footprint에 있는지 검사한다.
- 중력이 측정되지 않았다. 기본값은 **가장 큰 검증된 지지 평면을 지면으로 보는 가정**이다. 비슷한 크기의 서로 다른 지지 평면이면 모호함으로 거부한다. 벽/선반이 있는 장면에서는 이 가정이 틀릴 수 있다. 실제 지면 법선이 교정되어 있으면 JSON의 `ground_up_camera`를 설정한다. 카메라를 움직이는 실험에서 이 고정 camera-normal 옵션은 유효하지 않으므로 매 프레임 IMU/외부 교정값 없이 고정 값을 사용하지 않는다.
- 공식 `franka_umi_vis_mesh.obj`의 **열린 상태 실제 정점**과 지면 solid half-space 사이의 최소 거리를 계산한다. 그리퍼 mesh의 최소 signed distance가 추정 평면 RMSE보다 **커야** 통과한다. RMSE 이하(동일값 포함)는 REJECT이며, 추가 3mm는 적용하지 않는다. 구버전 JSON의 collision_margin_m은 호환을 위해 읽지만 판정에 사용하지 않는다. 삼각형 표면에 대한 평면 signed distance 최소는 정점에서 얻어진다. GraspGenX의 원본 SE(3)는 수정하지 않는다.
- 이는 제공된 franka_umi 형상에 대한 검사다. PiPER 실제 TCP/장착/추가 케이블과 같다고 가정하지 않는다. trajectory sweep, 로봇 arm, 기타 환경 충돌은 검사하지 않는다.
- 전부 충돌하면 `no_valid_grasp`, 지면 검증 실패면 `no_ground_plane`을 반환한다. 미검증 후보를 조용히 통과시키지 않는다. 지면을 못 찾은 프레임에서도 Baseline 후보는 비교 화면에 표시한다.

## 선택 및 공정한 비교

- Baseline: `0.6C + 0.1R + 0.25P + 0.05Y`, 기존 방식 그대로, 지면 필터 적용 전 Top-1.
- 신규: 지면 필터 통과 후보 중 `0.6C + 0.4T`의 최대값.
- `T = exp(-0.5*(theta/20deg)^2)`.
- `theta`는 ICP 공통 좌표계에서 **이전 신규 Top-1**과 현재 후보의 SO(3) 최단 회전각이다. 위치 차이는 기록만 하고 Temporal 점수에는 사용하지 않는다. position_sigma_m 설정은 호환용으로 남아 있지만 점수에 영향을 주지 않는다. 그리퍼 180도 대칭을 임의로 동일하게 취급하지 않는다.
- 첫 선택, ICP 신뢰도 부족/결과 누락, 직전 Top-1 없음, 선택 세대/시계가 바뀌면 **Confidence만** 사용한다. C는 기존 모델의 [0,1] 값이며 배치 min-max로 재정규화하지 않는다. Temporal도 [0,1]이다. 사용하지 않은 Temporal은 CSV 빈 칸/NPZ NaN으로 구분한다.
- 두 방식은 **똑같이 반환된 후보 배열**에 적용한다. 기본 기존 서버 top-K=100 이후의 후보에 대해 비교하며, 반환되지 않은 후보를 복원하지 않는다. 더 많은 후보를 받으려면 기존 서버가 지원하는 `--topk 1000 --threshold -1`을 사용한다(서버 top-K 허용 범위 1..1000). 요청 수는 기존 `--num-grasps` 설정을 따른다.

## 기록 및 비교 화면

기본 경로: `outputs/tracking_temporal/auto_sessions/<session>/`

| 파일 | 내용 |
|---|---|
| `all_candidates/*.npz` | 원본 grasps/C, 원본 순서, Baseline/Temporal/신규 점수, ground 통과 mask/clearance, 두 Top-1 index, 정확한 Frame ID의 ICP transform/plane |
| `comparison.csv` | 성공/No Valid 포함 모든 선택, 각 Top-1의 C/점수, 카메라 보정 위치·회전 변화량, ICP 신뢰도/시간/FPS, 추론시간 및 클라이언트 지연 |
| `comparison_summary.json` | Baseline/신규 변화량·Confidence·Temporal·latency 평균/P50/P95 및 실제 pose throughput |
| `poses.csv`, `samples/*.npz` | 신규 방식의 유효 Top-1만, 기존 세션 저장 형식과 호환. 위치/회전 delta는 ICP 기준이며, 원본 카메라 상대 delta는 `camera_relative_delta_mm/deg`로 따로 저장한다. R/P/Y는 참고 Baseline 성분이며 신규 S에 사용하지 않음 |
| `frames/*.npz` | 같은 Frame ID의 실제 RGB-D/마스크/K |
| `icp_frames.csv`, `continuous_icp.npz` | 처리한 프레임별 ICP 상태/시간과 승인된 카메라 변환, 거부/누락 사이를 경로로 연결하지 않음 |
| `icp_summary.json`, `summary.json` | ICP 처리/drop 집계 및 실제 완료 pose FPS/latency |

자홍색 Baseline과 초록색 신규 그리퍼를 함께 표시한다. 카메라 경로는 옅게, 선택 프레임 카메라는 파랑으로 표시한다. 고정 융합 물체 cloud와 현재 관측 cloud, 지면 평면도 표시한다. 속도/루프/가로 시간 슬라이더로 이동 가능하다. 재생은 기존 방식처럼 저장된 전체 RGB-D를 CPU로 다시 정합해 고정 물체/이동 카메라를 복원한다. 이는 시각화 전용이며 선택에 사용한 continuous ICP 및 점수를 바꾸지 않는다. raw 카메라 번호가 건너뛰어도 승인된 정합 사이의 경로를 그린다. 재생 정합이 실패한 시점에도 고정 물체와 전체 유효 궤적은 유지하고, 모르는 현재 카메라/그리퍼만 숨긴다. 다른 카메라 좌표 화면으로 전환하거나 identity를 넣지 않는다. `replay_camera_motion.npz`는 재생 전용 캐시다.

`comparison.csv:end_to_end_ms`는 프레임 준비→선택/후보·컨텍스트 저장 시점, `poses.csv:end_to_end_latency_ms`는 프레임 준비→유효 Top-1 저장 완료 시점이다. `icp_processing_ms`는 CPU 배경 전처리+평면+정합 전체, `icp_ms`는 정합만이다. `icp_fps`는 CPU 작업 완료 수/해당 처리 기간이며, 카메라 FPS나 grasp FPS와 다르다. `summary.json:completed_pose_fps`가 실제 완료 pose throughput이다.

## 서버에서 수행한 검증 / 한계

- 기하/좌표계·Confidence fallback·전 후보 충돌·실제 CAD/지면·배경 제외·ICP 관측 가능성 테스트 및 기존 선택/기록 regression tests.
- 실제 저장 RGB-D/SAM 마스크와 실제 저장 GraspGenX 후보의 통합 저장/비교 렌더링. 새 모델 요청은 보내지 않았다.
- warm-up10 이후 정지 RGB-D100회 반복: CPU 전체 평균46.30ms/P95 50.33ms, 순수 ICP 평균3.78ms. 실제 최신 값은 `outputs/temporal_selection/static_replay.json`을 따른다. 측정은 USB/통신/모델 추론/파일 저장/화면을 제외한 계산 비용이다. 정적 입력의 처리량은 움직이는 카메라의 정확도 검증이나 실제 Windows pose FPS가 아니다.
- 실제 움직이는 공식 Redwood RGB-D 5프레임으로 ICP/keyframe 승인 확인. 객체 마스크 annotation 없는 장면 전용 검사이며, 실제 D435i 물체 추적/grasp 정확도나 ground truth 기반 drift 평가가 아니다.
- Windows에서 실행 후 `comparison_summary.json`과 `icp_summary.json`으로 실측 FPS/latency/안정성을 확인해야 한다. 장면 texture/깊이/카메라 움직임에 따라 지면 또는 ICP가 거부될 수 있다.

설정 수정은 `temporal_default.json` 복사본을 만든 뒤 `--temporal-config 파일.json`으로 지정한다. 기존 모델/환경은 바꾸지 않는다.

재생 초기화 수정: 첫 촬영 프레임은 기준 카메라 transform이 정의상 identity이므로, 경로 복원 대기/실패 중에도 해당 프레임의 실제 Baseline/신규 grasp와 카메라를 표시한다. 다른 프레임에 identity를 대입하지 않는다. 초기 시야는 물체와 기준 카메라 원점을 모두 포함하며, 복원 오류를 오른쪽 패널과 터미널에 출력한다.

### 좌우 패널 재생 화면

왼쪽은 재생 타임라인/배속/표시 설정과 세션 상태, 오른쪽은 RGB/마스크와 Baseline/신규 점수 및 ICP/latency 비교 표다. RGB 높이를 180px로 제한하고 진단 정보를 짧은 표로 정리한다. 저장된 comparison.csv를 그대로 읽으며 과거 점수나 pose를 다시 계산하지 않는다.

Temporal에는 촬영 간격의 상한이 없다. 추론에 1초 이상 걸려도 동일 대상/시계의 이전 유효 Top-1과 현재 프레임의 ICP가 있으면 점수를 계산한다. 구버전 max_temporal_gap_s는 호환용이며 판정에 사용하지 않는다. 대상 변경·시계 변경·비정상 시각 역전·비교할 이전 pose/ICP 없음은 계속 구분한다. 이전에 저장된 점수는 수정하지 않는다.

### View 탭: 고정 회전 중심

회전 중심은 재생 시작 후 한 번만 초기화한다. 첫 비교 프레임에 유효한 평면이 있으면 물체 중심과 평면 투영점의 중점으로, 없으면 기존 scene bounding-box 중심으로 설정한다. 이후 프레임의 추정 평면/물체 변화로 중심을 갱신하지 않는다. View 탭의 XYZ 편집 또는 Reset initial center 버튼으로만 명시적으로 변경한다. Side view는 사용자가 누른 시점의 평면 법선을 화면 up으로 사용하되 중심은 유지한다. Side elevation=0이면 평면 높이의 정확한 옆 시점이며, 기본5도는 약간 위에서 보는 시점이다. Oblique view와 Fit entire scene도 제공한다. 시각화 카메라만 조작하고 실제 point cloud/카메라/grasp 좌표는 변경하지 않는다.
# Candidate filtering order

Temporal mode now requests every candidate returned by the official planner:
`--topk -1 --threshold -1`. It rejects other values to prevent confidence
preselection before ground filtering. The order is all planner candidates →
ground clearance filter → Confidence + Temporal ranking → Top-1.
The baseline still ranks the same complete input without ground filtering for
comparison. Legacy main_tracking.py retains its previous Top-100 default.
This requires the updated Tracking service (topk=-1 means unlimited).

