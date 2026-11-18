# Third-party code, models and geometry

This repository has not assigned a new blanket license to code/CAD from other authors. Preserve upstream notices and verify redistribution permissions for your intended use.

| Component | Source / revision | Local notice |
|---|---|---|
| GraspGenX server/samplers/checkpoints | https://github.com/NVlabs/GraspGenX · `44eb1d77d6d27bc97ee361e3afb717a2115e30a9` | `graspgenx-realsense/local_client/licenses/GraspGenX_LICENSE` |
| Official Viser helpers / selected franka_umi assets | Same official source and gripper HF revision in `docs/DEPENDENCIES.json` | Same license + `local_client/assets/provenance.json` |
| Tracking SAM / AOT / SAM / GroundingDINO | https://github.com/RogerQi/Tracking_SAM and pinned submodules; see dependency manifest | Downloaded separately; retain each upstream LICENSE |
| RealSense camera CAD/URDF | https://github.com/realsenseai/realsense-ros · `7a87097c499e3b0b69518d54a8e7a9ac1924e87c` | `local_client/assets/wrist_camera_rig/REALSENSE_LICENSE` and `REALSENSE_SOURCE.json` |
| PiPER/UMI V7 robot parts | Existing local `vbc-as2piper` robot asset pipeline, listed in rig.json source_robot | Included derivative rig meshes retain original ownership; no new license inferred |
| Wrist mount STL | User-provided `RealSense_D435_camera_piper_wrist.stl` | User CAD; distribution conditions are not specified by this repository |

`local_client/` paths above are relative to `graspgenx-realsense/`. Official model weights are not committed and must be obtained from their original repositories under the respective terms. Parameter/latency reports are experimental observations, not model redistribution.

Original mechanical drawing: [RealSense D400 Series Datasheet, March 2026](https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf). The downloaded full PDF is excluded from Git; rear mounting-hole dimensions and their CAD nominal interpretation are described in `local_client/WRIST_CAMERA_RIG.md`.

Before public distribution of the user-provided and robot-derived CAD, the owner should determine its applicable license. This note records missing license metadata; no permission is fabricated and no blanket open-source license has been added.
