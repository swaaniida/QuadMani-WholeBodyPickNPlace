# 상부 손목 카메라 / 전신 좌우 연결 영상

왼쪽은 전신, 오른쪽은 손목 카메라다. 각 화면 320×180, 연결본 640×180, 8fps. 같은 시뮬레이션 시점의 두 카메라를 독립적으로 렌더링·저장한 후 프레임을 좌우로 연결한다.

| 구성 | 전방 tilt | Pick / recover | Pick / drag |
|---|---|---|---|
| plate_only | [연결본](plate_only_tilt_in_place.mp4) | [연결본](plate_only_pick_recover.mp4) | [연결본](plate_only_pick_drag.mp4) |
| profiles_plate | [연결본](profiles_plate_tilt_in_place.mp4) | [연결본](profiles_plate_pick_recover.mp4) | [연결본](profiles_plate_pick_drag.mp4) |

독립 영상은 같은 이름의 `_main.mp4`, `_wrist.mp4`다. 예: [전신 단독](plate_only_pick_recover_main.mp4), [손목 단독](plate_only_pick_recover_wrist.mp4).

카메라는 gripper base 로컬 좌표 `(-0.10, 0, 0.015)m`에서 `(0, 0, 0.20)m`를 향한다. 로컬 -X인 손등 쪽에 좌우 대칭으로 배치해 양 손가락을 내려다본다. 광축은 그리퍼 +Z에서 약 28.4° 기울어지며 수직 FOV는 70° 설계 가정이다. 손목 회전에 따라 카메라도 함께 회전한다. 물체가 손가락에 가려지는 순간은 정상적으로 렌더링한다.

팔 준비 자세와 전방 tilt는 이전 후보를 유지했다. 빠른 시각화용 scripted PD/virtual locomotion 및 soft-grasp proxy이며 학습 정책 평가가 아니다. 카메라 질량과 실제 설치 기구는 포함하지 않았다. 이전 14개 영상은 상위 디렉터리에 보존했다.

재실행: `bash mujoco_mount_compare/run_camera_quick_suite.sh`
