# 현재 동작 기준 — 2026-11-17 소스 정리

이 문서는 초기 제안서와 오래된 보고서보다 우선한다. 새 모델 실행이나 새로운 하드웨어 측정을 이번 정리에서 수행하지 않았다.

## 구현 구조

Windows D435i → aligned RGB-D + 실제 K/scale → JPEG RGB/lossless depth → SSH tunnel → Tracking SAM 서비스(5557) → SAM ViT-H 초기 mask → AOTT 연속 VOS. 동일 Frame ID의 RGB/depth/mask를 보관하고, 별도 GraspGenX 서비스(5556)에 scene-depth 요청을 보낸다. Windows에는 PyTorch/CUDA가 필요 없다.

- 수동 ROI: `main.py`
- legacy orientation score/자동 기록: `main_tracking.py`
- background ICP + 지면 필터 + temporal 비교/기록: **`main_temporal.py`**
- 최신 temporal 기록 재생: **`view_temporal.py`** (session 생략 시 nonempty comparison.csv 기준 최신)
- branch 분석: `run_obb_replay.ps1` → export → 서버 독립 분석 → `view_obb_replay.py`
- 실제 카메라+손목/UMI rig: `view_wrist_rig.py` (`--rig-only` 가능)

모델과 서버는 기존 공식 구현을 재사용한다. 최신 UI가 있더라도 오래된 결과를 새 선택 알고리즘으로 다시 추론한 것으로 해석하지 않는다.

## 후보 → 지면 → 선택

1. 기본 planner는 **graspmoe**: diffusion과 공식 analytic OBB를 포함한다. diffusion-only 옵션도 제공한다. `num_grasps=200`은 요청한 diffusion 수이며 GraspMoE 전체 수와 항상 같지는 않다.
2. temporal 모드는 `topk=-1`, threshold=-1로 model-returned 전체 후보를 유지한다. 내부 모델 자체의 sampling/threshold와 client top-k는 구분한다.
3. 공식 static OPEN `franka_umi` CAD 전체 vertex를 각 원본 pose로 변환하여 추정 평면과 최소 signed distance d를 계산한다. pose를 이동/회전 보정하지 않는다.
4. **d > plane fitting RMSE**인 후보만 통과. d ≤ RMSE이면 reject. 추가 3mm margin은 폐기했다. 평면이 없거나 전부 reject이면 No Valid Grasp.
5. 통과 후보에 `S = 0.6 C + 0.4 T` 적용. C는 공식 discriminator confidence [0,1]. T는 아래 회전 유사도.

`theta = acos(clip((trace(R_prev.T @ R_current)-1)/2,-1,1))`

`T = exp(-0.5 * (theta_degrees / 20)^2)`

회전은 ICP의 같은 reference frame에서 비교한다. **위치 차이는 점수에 넣지 않는다. 180도 대칭은 다른 회전으로 취급한다.** 첫 유효 pose/이전 pose 없음/현재 ICP 불신이면 confidence만 쓴다. 1초 capture gap만으로 temporal을 끄지 않는다. `position_sigma_m`, `max_temporal_gap_s`, `collision_margin_m`은 호환용 legacy 설정이며 현재 점수/추가 margin에 사용하지 않는다.

Baseline은 같은 원본 후보에 `0.6C + 0.1R + 0.25P + 0.05Y`. 비교용 baseline은 ground-unfiltered이므로 선택된 baseline이 reject일 수 있다. R/P/Y는 별도 투영·정렬 선호도이며 Euler roll/pitch/yaw가 아니다.

## ICP·평면

정적 배경만 사용: SAM mask 11px dilation으로 물체 제외, voxel 20mm, 최대 12000 points. Point-to-plane ICP: correspondence 40mm, fitness≥0.55, RMSE≤18mm, step≤250mm/25°, normal observability 검사. Keyframe 10 accepted frames, 5마다 재정합, 최대6개. 10~20Hz는 목표이며 Windows 성능 보장이 아니다.

평면 RANSAC: 최대5000 points, inlier distance8mm, 최소250개/12%, plane RMSE≤6mm와 물체 지지 관계 검사. Plane fitting RMSE는 `sqrt(mean((n·p+b)^2))` (unit normal, inliers, meters). ICP RMSE는 서로 다른 두 프레임의 대응점 정합 오차다. 서로 대체하지 않는다. 중력 보정이 없다면 지면은 검증된 support-plane hypothesis이다.

## 좌표계·rig

- 광학 camera: X right/Y down/Z forward, meter. 모델 출력 `T_camera_gripper_base`.
- 공식 franka_umi: +Z approach/+X closing. 실제 PiPER UMI의 nominal TCP와 동일하지 않다.
- rig의 mesh/STL/URDF와 `rig.json`은 함께 유지한다. mount 원본 STL은 mm, URDF scale=.001.
- 최신 손목 장착: **flange Z축 180° clocking, -X 위쪽의 225°/135° 방사형 볼트축**. 카메라–거치대 접합은 그대로 유지한 채 함께 회전했다. camera_roll=0.
- `T_color_tcp`는 TCP → color optical 변환이며 역변환은 `T_tcp_color`. 실제 SDK K/scale/depth-color extrinsic은 `export_realsense_calibration.py`로 별도 저장.
- rig/TCP는 CAD nominal이며 hand-eye 실측값이 아니다. 원본 URDF 자동 생성에는 별도 로봇 저장소가 필요하다.

## 재생·기록

클릭으로 take 시작, R로 재초기화/새 take와 duration reset. Frame ID 기준 RGB-D/Mask와 원본 후보 및 선택 결과 저장. 모델 inference, RTT, ICP, capture/total latency와 completed pose FPS를 분리한다. request count나 capture FPS를 pose FPS라고 보고하지 않는다.

종료 후 object-fixed reference에서 camera trajectory/현재 camera와 grasp를 시간순 재생한다. slider·pause·speed·loop 지원. Replay reconstruction은 선택 시의 background ICP와 별도이며, 그 결과로 당시 점수를 다시 쓰지 않는다. 재구성 실패 시 명시적으로 표시한다.

## 검증 한계

로봇 제어/경로 계획/whole-body policy 학습은 구현 범위가 아니다. 지면 필터는 최종 open mesh의 평면 충돌만 검사한다. 접근 경로, 손가락 닫힘, arm/self collision, reachability는 검사하지 않는다. Confidence/clearance가 성공이나 안전을 보장하지 않는다. 서버 측 recorded replay와 localhost throughput을 Windows USB/SSH 성능으로 보고하지 않는다.
