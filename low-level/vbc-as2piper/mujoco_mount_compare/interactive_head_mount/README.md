# Interactive AS2 head PiPER placement preview

## Open

The preview server is listening on `127.0.0.1:8765` on the AstraGo host. Forward port 8765 in VS Code's **Ports** panel, then open the forwarded address. To start it manually from the project root:

```bash
bash mujoco_mount_compare/interactive_head_mount/serve.sh
```

## Use

- Left-drag to orbit; wheel to zoom; right-drag to pan.
- Click an upward-facing surface on the front AS2 top cover. The clicked 3D point is the center of the PiPER base underside footprint.
- The arm preview is translucent and uses the existing compact travel pose `[0, 0.76, -0.89, 0, 0.45, 0]` rad.
- Save the selected point as an AS2 `base_link` frame JSON coordinate.

The initial preview uses the selected point `(0.2151, 0, 0.1319) m` in the AS2 base_link frame. Its y coordinate is fixed at zero for this model. The arm preview is upright; it does not tilt to match the local STL surface. PiPER footprint center correction uses the original base mesh bounds (`x=[-0.0585, 0.0400]`, `y=[-0.0400, 0.0400]`, `z_min=0` m). The preview does not certify the AS2 cover as a load-bearing mount, detect the AS2/PiPER interface holes, or account for an adapter, fasteners, cables, or structural compliance. Check penetration and hardware support before using the saved position for a MuJoCo dynamics model or fabrication.

To regenerate the mesh buffers after changing the source MJCF:

```bash
PYTHONPATH=mujoco_mount_compare/.deps python mujoco_mount_compare/interactive_head_mount/export_preview_geometry.py
```

The page uses a local vendored copy of Three.js 0.160.0 (MIT license in `vendor/three/LICENSE`).
