# Locomotion/manipulation pose videos

[설계·검증 보고서](../../LOCOMANI_REPORT_KO.md) · [수치 요약](summary.md) · [자세 평가](pose_evaluation.json)

각 MP4는 외부 전신 시점과 오른쪽 위 wrist-camera inset을 함께 보여준다.

| Variant | Standing | Tilt | Walk omni | Tilt-walk omni | Pick/recover | Pick/drag | Pick/lift/drag |
|---|---|---|---|---|---|---|---|
| profiles_plate | [MP4](profiles_plate_standing.mp4) | [MP4](profiles_plate_tilt_in_place.mp4) | [MP4](profiles_plate_stand_walk_omni.mp4) | [MP4](profiles_plate_tilt_walk_omni.mp4) | [MP4](profiles_plate_pick_recover.mp4) | [MP4](profiles_plate_pick_drag.mp4) | [MP4](profiles_plate_pick_lift_drag.mp4) |
| plate_only | [MP4](plate_only_standing.mp4) | [MP4](plate_only_tilt_in_place.mp4) | [MP4](plate_only_stand_walk_omni.mp4) | [MP4](plate_only_tilt_walk_omni.mp4) | [MP4](plate_only_pick_recover.mp4) | [MP4](plate_only_pick_drag.mp4) | [MP4](plate_only_pick_lift_drag.mp4) |

Walking은 scripted stepping + bounded virtual base servo, grasp는 compliant attachment proxy다. 실제 learned locomotion 또는 contact grasp 검증 결과가 아니다.
