# Tracking + 자동 Top-1 기록 + 종료 후 재생

기존 Tracking SAM/AOT 및 GraspGenX를 그대로 사용한다. 자동 모드는 추론 요청을 한 개씩 처리한다. 완료 후 최신 유효 Depth/Mask 프레임으로 다음 요청을 보내며, 오래된 요청을 쌓지 않는다. 추적 손실/오래된 프레임에는 새 Grasp 추론을 요청하지 않는다.

각 추론 결과를 `S=0.6C+0.1R+0.25P+0.05Y`로 재정렬하고 **Top-1 pose만** 기록한다. 기록 중에는 3D/Grasp overlay 갱신을 하지 않는다. RGB-D/마스크 추적 미리보기는 계속 동작한다.

## Windows 업데이트 및 실행

기존 클라이언트를 종료한다. PowerShell의 현재 폴더와 관계없이 다음 명령을 실행할 수 있다. 기존 가상환경과 출력은 유지된다.

```powershell
scp -P 2222 -i "$env:USERPROFILE\.ssh\id_ed25519" suhwan@astrago:/data/team/users/suhwan/grasp-pose/graspgenx-realsense/dist/graspgenx-realsense-windows.zip "$env:USERPROFILE\graspgenx-realsense-windows.zip"
Expand-Archive -LiteralPath "$env:USERPROFILE\graspgenx-realsense-windows.zip" -DestinationPath "$env:USERPROFILE\graspgenx-realsense" -Force -ErrorAction Stop
Set-Location "$env:USERPROFILE\graspgenx-realsense\local_client"
.\.venv\Scripts\python.exe main_tracking.py --port 15557 --auto-grasp --duration 60
```

별도 PowerShell 창에서 기존 SSH 터널을 유지한다:

```powershell
ssh -p 2222 -i "$env:USERPROFILE\.ssh\id_ed25519" -N -o ExitOnForwardFailure=yes -L 127.0.0.1:15557:127.0.0.1:5557 suhwan@astrago
```

1. RGB 화면에서 물체를 클릭한다.
2. 유효 추적이 시작되면 자동으로 반복 추론하고 Top-1을 기록한다. G를 반복해서 누를 필요가 없다.
3. 첫 Top-1 저장부터 60초 후 제출을 멈추고 마지막 요청을 완료한다. `--duration`을 생략하면 Q까지 기록한다.
4. Q도 정상 종료/마지막 요청 완료/요약 출력 후 오프라인 Viser 화면을 연다. 대기 중 Q를 한 번 더 누르면 강제 종료하므로 미완료 결과는 기록되지 않는다.
5. 브라우저 Sample 슬라이더나 Play로 각 Top-1의 RGB/Depth/Mask/Point Cloud/공식 gripper CAD를 함께 확인한다. 터미널 Ctrl+C로 재생 서버를 닫는다.

- **A**: 자동 기록 시작 또는 요청 제출 일시정지/재개. 진행 중인 한 개의 결과는 완료하여 기록한다. 추적 프레임 아카이브는 계속된다.
- **R / 새 클릭**: 추적 초기화/대상 교체. 이전 세대 결과는 폐기한다. R은 이전 세션을 닫고 현재 재생 대상에서 제외한다. 다음 클릭에 새 폴더로 녹화를 시작하며, 첫 Top-1 저장부터 duration을 다시 센다. 이전 폴더는 디스크에 별도로 남지만 새 녹화·재생에는 합치지 않는다. 클릭 전에는 세션 아카이브를 시작하지 않는다.
- **S**: 기존 수동 캡처 기능 유지.
- `--max-poses 100`: 100개 기록 후 종료/재생.
- `--no-review`: 기록 후 요약만 출력. 재생은 나중에 실행한다.
- 자동 모드를 쓰지 않으면 기존 G/V 수동 기능이 유지된다. 자동 세션에서는 V 대신 Q로 종료 후 재생한다.

수집/추론 환경은 그대로다. 고정 물체·이동 카메라 재생에는 CPU Open3D가 필요하다: `python -m pip install -r requirements-visualization.txt`. 모델 서버는 기존 localhost 포트5556/5557을 사용한다.

## 세션 파일

`outputs/tracking/auto_sessions/<UTC_timestamp>_<id>/`

| 파일 | 내용 |
|---|---|
| `poses.csv` | 실제로 생성·저장된 Top-1 결과만 한 줄씩 기록 |
| `tracking_frames.csv` | 서버가 처리하여 반환한 추적 프레임별 시각/마스크 상태/깊이 통계/아카이브 상태 |
| `frames_with_poses.csv` | Frame ID로 두 로그를 합친 전체 프레임 표. 추론하지 않은 프레임은 pose/score/inference 칸이 비어 있음 |
| `frames/<FrameID>_<hash>.npz` | 같은 프레임의 RGB·Depth·Mask·intrinsics·scale·시각을 한 파일에 저장 |
| `samples/000000.npz` | Top-1 `(4,4)` pose 한 개, C/S/RPY, 해당 객체 Point Cloud/색상/K/Frame ID |
| `summary.json` | 측정 FPS, latency 평균/P50/P95, 성공/실패/아카이브 누락 통계 |
| `events.jsonl` | 요청·일시정지·오류·세대 변경·종료 이벤트 |
| `session.json` | 모델 요청 설정, 점수식, 좌표/시각 정의 |

RGB는 JPEG90 바이트 배열(`rgb_jpeg`), Depth는 무손실 `depth_raw`/`depth_m`, Mask는 uint8 `instance_mask`/`filtered_mask`(배경 0, 객체 1)다. NPZ의 `frame_id`와 `metadata_json`에 원본 ID/교정값이 들어 있다. 파일명 hash는 안전한 이름과 고유성을 위한 것이며 좌표/Frame ID를 바꾸지 않는다. Windows와 NFS에서 작은 파일 쓰기 비용을 줄이기 위해 RGB/Depth/Mask를 하나의 NPZ에 묶는다.

아카이브는 최대 4프레임의 비동기 쓰기 큐를 사용한다. 디스크가 느리면 오래된 **추적 프레임 아카이브**를 건너뛰고 CSV의 `archive_status=dropped_backlog`와 summary에 남긴다. **저장된 모든 Top-1 pose의 RGB-D/Mask는 별도 우선 저장으로 보존**한다. 누락 프레임에 과거 pose를 채워 넣지 않는다. 저장 대상은 서버에서 처리한 추적 프레임이며, 업로드되지 않은 모든 카메라 원시 30FPS 프레임은 아니다.

## CSV 주요 열

| 항목 | 열 |
|---|---|
| 시각/번호 | `capture_utc`, `host_timestamp_s`, `camera_timestamp_ms`, `frame_number`, `frame_id` |
| Top-1 pose | `x_m,y_m,z_m`, `qx,qy,qz,qw`, `t00..t33` |
| 점수 | `C,R,P,Y,S` |
| Tracking | `tracking_fps`, `mask_area_pixels`, `mask_valid`, `tracking_status`(추적 프레임 표) |
| 추론/지연 | `grasp_infer_ms`, `pose_generation_fps`, `instantaneous_pose_fps`, `end_to_end_latency_ms`, `request_to_result_ms`, `request_to_record_ms` |
| 후보 수 | `candidate_count_generated`, `candidate_count_after_threshold`, `candidate_count_after_filter`(threshold 및 기존 top-K 적용 후), `requested_num_grasps` |
| Point Cloud | `valid_depth_pixels`(마스크/깊이 필터 통과), `point_count`(다운샘플링 후) |
| 안정성 | `position_delta_mm`, `rotation_delta_deg`, `pose_dt_ms`, `stability_segment` |

`requested_num_grasps`는 요청 설정값이며 실제 생성 수와 구분한다. 모델별 후보 수는 실제 응답 배열에서 측정한다. 구버전 Tracking 서버에서 필터 전 개수를 반환하지 않으면 해당 값은 빈 칸이며 summary의 `candidate_count_unknown_records`에 집계한다.

위치 변화는 연속 Top-1의 translation 거리(mm), 회전 변화는 정규화 quaternion 내적의 절댓값을 이용한 SO(3) 최단 회전각(0~180도)이다. q와 -q는 같은 회전이다. 초기 pose/대상 교체/추적 손실/일시정지 경계에서는 변화량을 빈 칸으로 기록한다. 프랑카 그리퍼의 180도 대칭 자세도 회전 변화로 기록한다.

`camera_timestamp_ms`는 센서 시계의 프레임 시각이다. `host_timestamp_s/capture_utc`는 Windows에서 정렬·변환된 RGB-D가 준비된 시각이다. 두 컴퓨터 시각을 빼지 않는다. `end_to_end_latency_ms`는 같은 클라이언트의 프레임 준비→Top-1/컨텍스트 저장 완료 시간이다. 노출 시작→저장 전체 시간으로 해석하지 않는다.

## FPS 해석

- `completed_pose_fps`: Top-1 저장 개수 / (첫 자동 요청부터 마지막 저장 완료까지). 초기 클릭 대기와 마지막 결과 이후 idle은 제외한다. 중간의 일시정지/추적 손실은 포함한다.
- `steady_completion_fps`: 첫 번째 저장 이후 결과 간 간격에서 계산한 FPS.
- `pose_generation_fps`: 각 pose 기록 시점까지의 누적 완료 FPS.
- `model_only_fps_estimate`: `1000 / 평균 grasp_infer_ms`. 서버 계산만의 참고값이며 실제 파이프라인 FPS가 아니다.
- Tracking FPS, pose FPS, 카메라 FPS는 다르다. 정확한 실제 노트북 성능은 위 60초 명령 실행 후 summary를 확인한다.

## 나중에 재생

Q로 종료하면 실제 세션 경로가 출력된다. 그 경로를 넣는다:

```powershell
.\.venv\Scripts\python.exe view_session.py "outputs\tracking\auto_sessions\실제_세션_폴더명"
```

Sample은 pose가 실제 생성된 프레임을 순서대로 재생한다. 현재 기본 화면은 D435i RGB-D로 프레임 간 카메라 자세를 추정하여 첫 Grasp 프레임의 공통 좌표계로 변환한다. 물체는 첫 프레임의 위치에 고정하고 카메라 이동 경로를 표시한다. 로봇 TCP 변환이나 중력 보정은 하지 않는다.

`grasp_rpc_rtt_ms`는 A100 Tracking 서비스→GraspGenX localhost 왕복이다. Windows→Tracking 서비스 왕복은 `tracking_rtt_ms`와 `result_poll_rtt_ms`, 전체 결과 수신 지연은 `request_to_result_ms`로 구분한다.

카메라 없이 동봉된 실제 모델 추론의 첫/마지막 프레임 발췌를 재생할 수도 있다:

```powershell
.\.venv\Scripts\python.exe view_session.py example_auto_session
```

이 예제는 정지된 공식 RGB-D 프레임을 반복 전송한 실제 모델 실행 결과다. 실제 Windows 카메라 실험이나 움직이는 RGB-D 시퀀스 검증 결과가 아니다.

### 클릭 직후 추적 초기화 수정

Grasp 서버가 `One grasp job is already running` 또는 Grasp 작업 오류를 반환해도 SAM 추적을 초기화하지 않는다. 요청은 단일 실행으로 예약하며, 자동 모드는 최대4초 간격의 backoff로 재시도한다. 재선택 후 이전 Grasp 작업이 아직 계산 중이어도 새 물체의 마스크 추적은 계속된다. 실제 Tracking 프레임 통신 오류나 Frame ID 불일치는 기존 reset 처리를 유지한다.

### 고정 물체 / 움직이는 D435i 재생 (현재 기본 모드)

한 번 설치:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-visualization.txt
```

- 첫 Top-1 카메라를 기준 좌표계로 사용한다. 각 Top-1 시점의 저장된 물체 Point Cloud를 정합된 카메라 자세로 변환해 3mm voxel로 융합한 고정 cloud를 메모리에 둔다. 최초 클릭 마스크 한 장만으로 만든 cloud가 아니다.
- 저장된 **전체 RGB-D 추적 프레임**을 사용해 카메라 이동·회전을 CPU RGB-D odometry로 추정한다. 객체 마스크 영역만으로 정합하지 않고 정적인 배경도 사용한다.
- 파란 선/방향 화살표/프러스텀은 **D435i 카메라 이동**이다. 실제 K를 사용해 시야를 그린다.
- Grasp는 `T_reference_camera @ T_camera_gripper`로 같은 공통 좌표계에 표시한다. 시작은 불투명도 25%/옅은 노랑, 중간은 주황, 끝은 불투명/짙은 빨강이다. 희미한 전체 카메라 궤적 위에서 현재 카메라만 굵고 진한 파랑으로 강조한다.
- 오른쪽 RGB/Depth/Mask는 선택한 원본 Frame ID와 일치한다. 거리·C/S 그래프와 슬라이더/시간 재생도 함께 표시한다.
- 원본 Grasp/CSV/Depth는 수정하지 않는다. 결과는 세션 아래 `camera_motion/trajectory.npz`, `report.json`, `aligned_grasps.csv`에 별도 저장하고 다음 재생에서 재사용한다.
- 정합은 백그라운드에서 진행하며 브라우저와 터미널에 진행 상황을 표시한다. 완료 전에는 기준 물체만 표시한다.
- 정합 실패 프레임에 identity나 이전 자세를 넣지 않는다. 해당 시점의 3D 카메라/Grasp를 숨기고, 카메라 경로에 빈 구간을 남긴다. 원본 RGB-D는 계속 볼 수 있다.

D435i IMU를 기록한 것은 아니므로 이 모드는 **RGB-D로 추정한 카메라 자세**다. 첫 카메라 기준의 metric 좌표이며 절대 월드/중력/로봇 좌표가 아니다. 물체와 배경이 정지해 있어야 하며, 무늬 없는 평면·가려짐·빠른 움직임·깊이 누락에서는 실패하거나 드리프트할 수 있다. 정확한 6-DoF tracker의 외부 측정값으로 해석하지 않는다.

CPU 처리에 맞춰 odometry 입력은 최대640×480으로 줄이고 K도 같은 비율로 바꾼다. Depth 단위와 원본 데이터는 유지한다. CUDA나 PyTorch는 노트북에 설치하지 않는다.

### 현재 기본 표시: 복원 후 애니메이션 루프

- 카메라 정합과 물체 cloud 융합이 완료되면 처음부터 자동 반복 재생한다.
- `Playback speed (x)`: 0.1~4배속. 마지막 pose는 기록된 양의 시점 간격 중앙값만큼 유지한 다음 처음으로 돌아간다. 속도 설정은 이 대기에도 적용된다.
- 가로 `Timeline (seconds)` 바를 클릭하거나 드래그하면 일시정지하고 선택 시각의 실제 기록 pose를 고정해서 본다. `Play / pause`를 켜면 그 위치에서 다시 재생한다. `Sample`은 정확한 pose 단위 선택이다.
- 시간은 촬영 프레임의 host timestamp, sensor timestamp, 기록 완료 시각 순으로 사용 가능 여부를 확인한다. 연산 완료 지연을 촬영 간격으로 대신 사용하지 않는다.
- 고정 융합 cloud는 불투명도 30%의 옅은 색, 선택 시점에서 실제 보인 object cloud는 원래 색으로 겹쳐 표시한다. 각각 체크박스로 숨길 수 있다.
- RGB에는 같은 Frame ID의 마스크와 실제 저장 Top-1 좌표축을 투영한다. 카메라 위치는 첫 물체 cloud 중심에 대한 상대 위치도 표시한다. 물체 중심 좌표축의 방향은 첫 카메라와 같고, 물체의 고유 회전을 추정한 것은 아니다.
- 실제 grasp가 생성된 시점만 재생한다. 중간 프레임에 grasp를 보간하거나 새로 추론하지 않는다. 전체 추적 RGB-D는 카메라 궤적 복원에 사용한다.
- `camera_motion/fused_object.npz`에 융합 결과를 저장한다. 현재 관측 cloud와 융합 cloud는 재생 중 메모리에서 재사용한다. RGB-D 이미지는 선택 시점의 원본 파일에서 읽는다.
- `Original camera-coordinate snapshots (static)`은 비교용 이전 화면이며 기본 OFF다.
- 물체/배경은 정지한 한 세션을 사용한다. 이는 RGB-D 정합 기반 부분 표면 융합이며, 완전한 물체 mesh나 외부 측정으로 검증된 카메라 궤적은 아니다.
