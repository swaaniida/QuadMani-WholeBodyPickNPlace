# D435i / PiPER wrist / UMI V7 nominal assembly

The source STL was found at `assets/wrist_camera_mount/RealSense_D435_camera_piper_wrist.stl`.
No mount URDF/STEP was present. A new self-contained URDF was built from its
actual geometry, the official D435i description and the existing fixed-open
UMI V7 model. Source robot files, model services and inference code are unchanged.

## Windows

From the existing `local_client` folder, inspect the assembly:

```powershell
.\.venv\Scripts\python.exe view_wrist_rig.py --rig-only
```

Replay the latest temporal recording with the mounted real UMI geometry:

```powershell
.\.venv\Scripts\python.exe view_wrist_rig.py
```

An explicit recording path is accepted as the positional argument. Existing
object-fixed reconstruction, camera trajectory, timeline, loop and speed controls
are retained. The actual rig follows the exact capture's camera transform;
unregistered captures are not assigned invented transforms. This renders a
CAD-mounted rig, not recorded robot joint telemetry. Fingers are static OPEN.
The updated `view_obb_replay.py` also detects the installed rig and adds an
`Actual camera + UMI nominal TCP` visibility control. The existing OBB replay
workflow therefore shows the same actual wrist assembly alongside candidates.

To read actual camera calibration, close other camera programs first:

```powershell
.\.venv\Scripts\python.exe export_realsense_calibration.py --serial 841512070365
.\.venv\Scripts\python.exe view_wrist_rig.py
```

This writes `realsense_calibration.json`, `camera_tcp_transform.json`, and a new
`assets/wrist_camera_rig/piper_umi_d435i_device_calibrated.urdf`. The original
nominal URDF is retained. The viewer automatically uses that calibration JSON
when present. `--calibration PATH` chooses another device calibration explicitly.
The profile defaults to 640x480/30 FPS; width/height/fps are configurable.
Windows USB capture has not been tested on the remote server.

## Geometry and evidence

- Source STL unit is treated as mm, cross-checked against the camera's 45 mm
  mounting pitch and the PiPER flange's 28.5 mm external radius.
- Camera bracket: two Ø3.4 mm through holes, 45 mm center spacing. Contact-side
  centers are obtained by fitting closed planar facet boundaries.
- D435/D435i: rear M3 threaded holes at raw visual-mesh coordinates
  (+/-22.5, 0, -25.05) mm. These agree with the official mechanical drawing.
- PiPER bracket: two nonparallel radial bolt axes recovered from circle fits.
  The bracket's inner cylindrical seat has radius 28.5 mm, matching the flange.
- Flange radial bore cylinders were fitted from actual STL bore-surface vertices:
  radius 1.7 mm, axis at flange z=7 mm, azimuths +45/-45 degrees in the default
  clocking. Mount and flange holes are aligned coaxially. Outer-plane circle
  centers sit at different distances along the screw axis due to bracket wall
  thickness; forcing those surface centers to coincide would bury the wall.
- Registration residuals are in `assets/wrist_camera_rig/rig.json`. The very small
  numerical fit residuals are not manufacturing tolerances or physical accuracy.
- The default mount is on the upper PiPER flange -X side, with one ordered camera-hole mapping.
  Symmetric screw patterns do not uniquely determine clocking or 180-degree
  camera roll. These are explicit assembly choices, not measured hardware facts.

## Coordinate chain

```text
piper_link6 -> piper_flange_link -> wrist_camera_mount
                                  -> camera_bottom_screw_frame
                                  -> camera_link -> depth/color optical frames
             -> piper_gripper_base -> fixed-open UMI fingers
                                  -> piper_umi_nominal_tcp
```

`T_color_tcp` maps TCP-frame points into the RGB camera optical frame:
`p_color = T_color_tcp @ p_tcp`. Its inverse is `T_tcp_color`.
RGB optical axes: X right, Y down, Z forward. The current point cloud uses this
frame because depth is aligned to color. The camera visual-mesh origin is not
the optical frame; the official URDF visual and sensor transforms are retained.

The inherited nominal TCP is at gripper-base (0,0,0.183851719307) m. With the
inherited 4.5 mm flange-to-base joint, its flange Z is 188.351719307 mm.
The default nominal RGB-frame TCP translation is approximately
(32.5631, -19.5109, 167.3501) mm; the complete matrix is in `rig.json`.

Intrinsics cannot be calculated from a URDF. The Windows exporter reads actual
color/native-depth/aligned-depth intrinsics, distortion, depth scale, serial,
profile and depth-to-color extrinsics from the SDK. Extrinsic rotation storage
order is checked against `rs2_transform_point_to_point` before use. Those SDK
extrinsics replace only the nominal depth/color relationship. They do not
measure camera-housing-to-optical uncertainty or physical TCP/hand-eye error.

The original UMI V7 fit report already describes printed-to-carriage offsets and
TCP as provisional geometric values. They are preserved and explicitly labeled
nominal here. Robot execution requires separate physical TCP/hand-eye validation.
GraspGenX predicted `franka_umi` gripper-base poses remain separate from this
actual PiPER/UMI TCP; no equivalence or automatic target conversion is assumed.

## Source and reproduction

- Official camera: `https://github.com/realsenseai/realsense-ros`, commit
  `7a87097c499e3b0b69518d54a8e7a9ac1924e87c`, `_d435i.urdf.xacro` includes `_d435.urdf.xacro`.
- Official mechanical reference: RealSense D400 datasheet March 2026,
  Figure 10-9, printed page 139 (PDF index 138).
- Existing robot: `vbc-as2piper/low-level-as2piper-hw/resources/robots/as2piper/urdf/as2piper_umi_open_pose_v7.urdf`.
- Generator: `scripts/build_wrist_camera_rig.py`. `--clocking-deg` accepts
  0/90/180/270; `--camera-roll-deg` accepts 0/180. Defaults are 180/0 (upper wrist screw pair).
- Output: `assets/wrist_camera_rig/piper_umi_d435i.urdf`, `rig.json`, local meshes.
- Visual mesh LOD uses a 0.35 mm grid only for browser rendering; fitting and
  URDF source geometry use the full-resolution assets. No source mesh is altered.
- Server proof: URDF parses (21 links/20 fixed joints), circle/axis residuals,
  inverse transforms, SDK matrix conventions and Viser scene/replay smoke tests.
