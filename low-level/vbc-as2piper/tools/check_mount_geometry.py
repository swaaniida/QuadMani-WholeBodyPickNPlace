from pathlib import Path
import struct
import re

ROOT = Path(__file__).resolve().parents[1] / "robot-sources"

paths = {
    "AS2 body": ROOT / "unitree_ros/robots/as2_description/meshes/base_link.STL",
    "PiPER base": ROOT / "agx_arm_urdf/piper/meshes/base_link.stl",
}

def load_vertices(path):
    data = path.read_bytes()

    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + 50 * n:
            vertices = []
            for i in range(n):
                offset = 84 + 50 * i + 12
                values = struct.unpack_from("<9f", data, offset)
                vertices.extend([
                    values[0:3],
                    values[3:6],
                    values[6:9],
                ])
            return vertices

    pattern = rb"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)"
    return [
        tuple(float(v) for v in match)
        for match in re.findall(pattern, data)
    ]

for name, path in paths.items():
    verts = load_vertices(path)
    if not verts:
        raise RuntimeError(f"No mesh vertices found: {path}")

    print(f"\n=== {name} ===")
    print(f"Vertices: {len(verts)}")

    for i, axis in enumerate("XYZ"):
        lo = min(v[i] for v in verts)
        hi = max(v[i] for v in verts)
        print(f"{axis}: [{lo:.4f}, {hi:.4f}] m")

    center = [
        v for v in verts
        if abs(v[0]) <= 0.05 and abs(v[1]) <= 0.05
    ]

    if center:
        print(
            "Near-center vertex Z range:",
            f"[{min(v[2] for v in center):.4f}, "
            f"{max(v[2] for v in center):.4f}] m"
        )
    else:
        print("No vertices near the XY center")
