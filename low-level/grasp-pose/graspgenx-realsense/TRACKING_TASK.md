# Tracking SAM + RealSense + GraspGenX 통합

기존 graspgenx-realsense 프로젝트를 확장한다. 로봇 제어/trajectory/WBC 구현은 범위 밖이다. 모든 프로젝트 파일/추가 환경은 grasp-pose 내부에 둔다.

공식 소스: RogerQi/Tracking_SAM (INSTALL.md, MODEL_PREP.md, demo.py, main_tracker, aott), Ericonaldo/visual_wholebody, 설치된 NVlabs/GraspGenX client-server API를 확인한다. SAM 초기 분할 뒤 **AOT VOS의 시간적 memory propagation**을 사용한다. 매 프레임 SAM 독립 추론으로 대체하지 않는다.

실행 순서:

1. 별도 tracking 환경에 공식 checkpoint(SAM ViT-H, AOTT_PRE_YTB_DAV)를 설치한다. 필요 시 optional GroundingDINO를 별도 기능으로 둔다.
2. 공식 DAVIS_bear demo의 SAM→AOT 연속 프레임 propagation을 실제 실행하고 지연/FPS/GPU 메모리/결과 mask를 저장한다. 원격 headless일 경우 GUI 입력만 자동화한 harness라는 사실을 명시한다.
3. GraspGenX 5556을 유지하며 Tracking SAM service 5557을 localhost에만 바인딩한다. 두 모델은 서로 다른 Python/PyTorch/CUDA 환경의 독립 프로세스이다.
4. Windows 기존 RealSenseCapture와 cloud/viewer/recording을 재사용한다. 클릭으로 SAM 초기화, 이후 AOT 추적, R reset, S RGB-D/mask 저장, G 최신 유효 **동일 프레임 RGB/depth/mask** grasp, V 3D, Q 종료.
5. RGB JPEG, depth lossless uint16, 실제 depth scale/K와 frame ID/timestamp를 하나의 payload로 보내고 latest-frame 정책으로 요청이 쌓이지 않게 한다. Windows는 tracking service에만 큰 frame을 보낸다.
6. Tracking service가 frame별 depth/mask/K를 보관하고 G 시 snapshot을 고정해 localhost GraspGenX **infer_scene_depth**에 전달한다. 별도 worker가 grasp 요청을 처리하여 tracking/capture가 계속된다.
7. Mask bg=0, target=1, 실제 optical camera frame/meter를 유지한다. Mask 범위 안에서 near/far/band/voxel을 적용하고 raw tracking mask와 depth-filtered mask를 구분한다. ROI 재선택은 요구하지 않는다.
8. Empty/area jumps/temporal inconsistency, invalid depth, age, session/frame mismatch를 검출하고 lost 상태를 latch한다. 단순 heuristic은 대상 identity의 보장이 아님을 명시하며 lost 상태에서 G를 금지하고 재클릭을 요구한다.
9. Mask overlay, ID/status/depth validity/FPS/model latency, cloud/CAD/grasp/score/pose axes, capture-matched RGB grasp overlay와 결과 저장을 제공한다.
10. capture, encode/request RTT, SAM, VOS, mask-depth processing, GraspGenX, visualization, G end-to-end를 분리하고 CSV/mean/P50/P95를 기록한다. 독립 PC 시계 차를 one-way latency로 사용하지 않는다.

최종 하드웨어 기준: **실제 D435i의 연속 촬영 영상에서 Tracking SAM 추적→동일 시점 depth/mask→실제 grasp 반환 및 시각화**. 원격에서 Windows USB에 접근할 수 없으므로 데이터셋 replay 검증과 Windows hardware 검증을 분리하고, 실행 명령과 남은 절차를 README/TRACKING_REPORT.md에 기록한다. 성공하지 않은 단계를 성공으로 보고하지 않는다.
