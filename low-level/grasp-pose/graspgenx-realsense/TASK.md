# RealSense RGB-D → A100 GraspGenX → 6-DoF Grasp Pose Visualization

요청일: 2026-11-11. 독립적인 실제 모델 검증 및 성능 분석 PoC를 구현하고 실행한다.

## 목표 및 범위

Windows USB RealSense의 RGB-D를 획득하고, 수동 선택한 물체의 metric point cloud를 SSH 터널/ZMQ로 astrago의 A100에 전송한다. 공식 NVIDIA GraspGenX pretrained 모델로 grasp pose (N,4,4), confidence (N,)를 반환하고 노트북에서 3D 및 RGB overlay로 시각화한다. 모든 결과는 최초 RealSense optical camera frame (x 오른쪽, y 아래, z 전방), meter 단위이다. Gripper base와 robot TCP를 동일하다고 가정하지 않는다.

로봇 동작, PiPER/AS2 제어, trajectory policy, whole-body controller, 새 모델 학습은 범위 밖이다. 기존 프로젝트 및 Python 환경과 다른 사용자의 작업을 변경하지 않는다. 가짜 inference나 측정값을 실제 결과로 제시하지 않는다.

## 공식 소스

- https://github.com/NVlabs/GraspGenX
- https://github.com/NVlabs/GraspGenX/tree/main/client-server

최신 README와 실제 Python API를 읽고 재사용한다. API/옵션을 추측하지 않는다.

## 실행 순서

1. 디렉터리, GPU/MIG, driver/CUDA, Python, 디스크, 기존 설치를 조사하고 기록한다.
2. 독립 환경에 공식 소스, pretrained generator/discriminator, gripper assets를 설치한다.
3. **공식 sample point cloud로 실제 추론을 먼저 검증한다.** SE(3), confidence, GPU 메모리 및 추론시간을 확인한다. 실패하면 원인과 재개 방법을 명시하며 성공을 주장하지 않는다.
4. 공식 client-server/graspgenx_server.py로 checkpoint를 한 번 로딩하는 상주 서버를 구성한다. ZMQ 5556, franka_umi 우선, localhost에만 바인딩하고 health/error handling을 검증한다. graspmoe 우선, 실제 지원 시 diffusion 비교를 제공한다.
5. Windows 전용 local_client를 작성한다. PyTorch/CUDA/전체 모델 설치 없이 pyrealsense2, numpy, OpenCV, pyzmq, msgpack/msgpack-numpy 및 시각화 라이브러리로 동작한다.

## Windows 획득 및 선택

- 장치 모델/serial 탐색, 640×480@30 시도, 지원 모드 조회 및 fallback.
- RGB와 depth 스트리밍, depth→color alignment, aligned frame의 실제 intrinsics 및 실제 depth scale 사용.
- 미리보기, 유효 depth 비율 표시, 연결 해제/재연결 처리.
- S: 고정한 RGB-D frame에서 ROI 선택; G: 선택 물체 inference; V: 3D 결과; R: 선택 초기화; Q: 종료.
- SAM/YOLO 설치 없이 ROI, 조정 가능한 depth range/foreground filtering으로 binary mask 생성.
- invalid depth 제거, (N,3) float32 meter cloud 생성, downsampling 및 cloud preview. 임의 회전/metric normalization 금지.
- 카메라 캡처와 추론을 분리하고 요청은 하나만 허용하여 미리보기가 멈추거나 backlog가 생기지 않게 한다.
- 모든 요청/결과에 촬영 frame ID/timestamp를 연결한다. 과거 결과를 현재 영상에 섞지 않는다.

## 원격 API 및 시각화

공식 name-based client.infer(point_cloud, gripper_name='franka_umi', num_grasps=200, grasp_threshold=-1.0, topk_num_grasps=100)를 실제 코드로 확인한다. infer_scene_depth(depth_m, K, instance_mask, sweep_params)와 planner 지원 경로도 조사한다. Segmented cloud 성공이 최우선이다.

실제 cloud, 상위 10 pose, 최고 score 강조, 각 confidence, camera XYZ, pose 위치/회전 확인과 결과 저장이 필수이다. 가능하면 공식 viser 구현을 재사용한다. CAD를 쓸 수 없으면 명시적으로 표시한 pose axes/단순 geometry를 쓴다. 3D 회전/확대와 RGB grasp-axis projection overlay를 제공한다.

## 계측과 산출물

RGB-D acquisition, alignment, segmentation, cloud generation, downsampling, network RTT, server inference, visualization, total latency를 각각 측정한다. 서로 다른 PC의 clock을 빼지 않는다. CSV 저장 및 mean/P50/P95를 출력한다. RTT는 inference를 포함하며 별도로 보고한다.

독립 프로젝트에 server/, local_client/, configs/, tests/, scripts/, outputs/{rgb,depth,pointcloud,grasps,benchmarks} 및 Windows 복사용 ZIP을 제공한다.

완료 보고에는 실제 구현, upstream commit/version, checkpoint, gripper configuration, server 실행/Windows SSH tunnel/설치/실행 명령, 실제 테스트 여부, GPU 메모리, 단계별 latency, 한계/후속 조치를 포함한다. 원격에서 접근 불가한 Windows USB 카메라 검증은 사용자의 실행 절차와 미검증 상태를 명시한다.

## 완료 기준

- [x] A100 공식 sample 추론
- [x] localhost ZMQ health
- [ ] Windows 실제 RGB-D 획득 (Windows에서 검증 필요)
- [x] 선택 물체 cloud 생성 (공식 recorded RGB-D replay; Windows 실물은 미검증)
- [x] ZMQ 전송 및 A100 inference (localhost 실측; Windows SSH 구간은 미검증)
- [x] SE(3) 및 confidence 반환
- [x] 실제 recorded cloud + pose/CAD scene 및 RGB overlay (브라우저 수동 조작은 미검증)
- [x] 결과 및 latency 저장 (Windows acquisition/alignment는 미측정)

실행 상태와 실제 측정 결과는 REPORT.md에 기록한다. 권한 부여 후 프로젝트 전체와 Python runtime을 `/data/team/users/suhwan/grasp-pose/graspgenx-realsense` 내부로 이전한다. 이전 경로에 호환용 심볼릭 링크를 남기지 않으며 서버도 새 경로에서 재시작한다.
