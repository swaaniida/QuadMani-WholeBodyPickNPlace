# Grasp Pose 선택 기준

모델/서버/Tracking SAM/통신 코드는 변경하지 않는다. Windows UI에서 반환된 후보를 받은 후 다음 점수로 내림차순 재정렬한다.

`S = 0.6C + 0.1R + 0.25P + 0.05Y`

- C: 모델 원본 confidence (0~1). 후보 집합별 min/max 재정규화를 하지 않는다.
- 카메라 optical 좌표: X 오른쪽, Y 아래쪽, Z 전방.
- 공식 GraspGenX gripper base 좌표: **+Z 접근축, +X 손가락 닫힘축**. `T_camera_gripper_base` 회전행렬의 첫 번째 열이 닫힘축 x, 세 번째 열이 접근축 z다. TCP 좌표가 아니다.
- R: x를 카메라 XY 평면에 투영한 방향이 카메라 +X와 이루는 각도(카메라 Z 주위). `R=(1+x_x/hypot(x_x,x_y))/2`.
- P: 객체 point cloud의 중심을 카메라 원점에서 바라보는 단위벡터 d와 접근축 z의 정렬도. `P=(1+dot(z,d))/2`. 반대 방향 접근에는 낮은 점수를 준다.
- Y: z를 카메라 XZ 평면에 투영한 방향이 카메라 +Z와 이루는 각도(카메라 Y 주위). `Y=(1+z_z/hypot(z_z,z_x))/2`.

R/P/Y는 각각 `(1+cos(angle))/2`로 0~1이다. 투영 길이가 0인 특이 자세는 해당 R 또는 Y를 0.5(중립)로 처리한다. 이는 사용자가 지정한 카메라 축 기반 선호도이며, 일반적인 XYZ Euler roll/pitch/yaw 분해가 아니다. 부호 있는 축을 사용하므로 두 손가락 그리퍼의 180도 대칭 자세를 자동으로 같은 자세로 취급하지 않는다. 최종 점수가 같으면 원래 반환 순서를 유지한다. C가 동점이면 원래 반환 순서의 첫 후보를 기존 C#1로 유지한다.

**중력 기준은 입력되지 않는다.** P는 카메라→물체 방향을 선호한다. 카메라가 바닥 물체를 내려다보는 경우 그 방향으로 접근하지만, 정확한 중력 수직 하향 제약은 아니다. 중력 기준을 적용하려면 카메라 자세 보정 정보가 추가로 필요하다.

재정렬 범위는 현재 추론 API가 반환한 후보 전체다. 모델 내부 sampling, threshold, 기존 top-K 후보 추출은 그대로다. S는 선택 선호도이며 학습된 성공 확률이 아니다.

## 표시 및 저장

- 2D overlay와 Viser 모두 S 기준 Top-10 표시.
- 금색: 최종 S#1. 청록색: 원래 C#1. 같으면 `S#1 = C#1`로 금색 표시.
- C#1이 Top-10 밖에 있으면 비교를 위해 추가 표시(최대 11개).
- overlay 상단과 3D 상세 정보에서 C/R/P/Y/S 및 pose를 확인한다.
- NPZ: 정렬된 grasps, 원본 confidences, `selection_scores`, `orientation_scores`(열 순서 R/P/Y), `original_candidate_indices`.
- JSON: 후보별 final_score/R/P/Y/selected/confidence_best와 선택 수식, 기준 방향, 원본 인덱스.
- `view_saved.py`는 camera_optical로 저장된 기존 결과도 새 기준으로 표시한다. 다른 좌표계로 표시된 공식 샘플은 C 순서를 유지한다.

## Windows 업데이트

기존 클라이언트를 종료한 다음, **현재 local_client 폴더에서** 실행:

```powershell
scp -P 2222 -i "$env:USERPROFILE\.ssh\id_ed25519" suhwan@astrago:/data/team/users/suhwan/grasp-pose/graspgenx-realsense/dist/graspgenx-realsense-windows.zip .
Expand-Archive .\graspgenx-realsense-windows.zip -DestinationPath .. -Force
.\.venv\Scripts\python.exe main_tracking.py --port 15557
```

기존 SSH 터널과 서버는 그대로 사용한다. 추가 패키지는 없다. G로 새 후보를 받고 V로 비교한다.

카메라 없이 실제 저장 결과의 비교 화면 확인:

```powershell
.\.venv\Scripts\python.exe view_saved.py example_results/official_rgbd_ranked.npz
```

## 확인한 공식 소스

설치된 GraspGenX commit `44eb1d77d6d27bc97ee361e3afb717a2115e30a9`:

- `graspgenx/serving/types.py`: SweepVolumeParams의 +Z approach / +X closing 정의.
- `graspgenx/samplers/graspmoe.py`: `_build_face_candidates`에서 gz가 물체 내부를 향하며 gripper base가 tip에서 -localZ로 이동하는 실제 구성 확인. 해당 docstring의 'Z (closing direction)' 표현과 달리 코드 및 serving types에서 Z는 approach다.
- franka_umi asset config: fingertip `[0,0,0.177]`, open sweep offset `[0,0,0.14]`.
