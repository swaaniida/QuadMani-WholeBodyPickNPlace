# Nominal D435i wrist camera

Camera/mount meshes are separate rigid additions; original robot meshes are unchanged. Optical axes are X right, Y down, Z forward. Per-URDF *_camera.json files preserve each existing nominal TCP. Camera mass is nominal 72 g; camera inertia is a uniform-box estimate. Mount mass/inertia assume solid PLA density 1240 kg/m3; actual printed mass is unknown. Optical helper frames have explicitly zero mass. This is not physical hand-eye/TCP calibration. No camera intrinsics are fabricated: obtain K/distortion/depth scale/extrinsics from the actual device.
