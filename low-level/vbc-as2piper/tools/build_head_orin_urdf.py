#!/usr/bin/env python3
"""Build a VBC-only head-PiPER + rear-centered Orin URDF variant."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "low-level-as2piper-hw/resources/robots/as2piper/urdf/as2piper_umi_actuated_v7.urdf"
ORIN_SOURCE = ROOT / "reference/orin_source/as2piper_center_umi_orin64.urdf"
OUTPUT = ROOT / "low-level-as2piper-hw/resources/robots/as2piper/urdf/as2piper_head_umi_orin64_backmount.urdf"
REPORT = ROOT / "mujoco_mount_compare/results/orin64_head_piper_preview/placement_validation.json"
HEAD_PIPER_XYZ = "0.22435 0 0.1319"
ORIN_XYZ = "-0.03 0 0.111875"  # box bottom z=0.07605 m, above base collision top z=0.076 m by 0.05 mm

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def indent(element, level=0):
    pad = "\n" + "  " * level
    if len(element):
        if not element.text or not element.text.strip():
            element.text = pad + "  "
        for child in element:
            indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = pad + "  "
        if not element[-1].tail or not element[-1].tail.strip():
            element[-1].tail = pad

def main():
    if OUTPUT.exists():
        raise SystemExit("Refusing to overwrite existing output: {}".format(OUTPUT))
    root = ET.parse(SOURCE).getroot()
    root.set("name", "as2piper_head_umi_orin64_backmount")
    mount = root.find("joint[@name='as2_to_piper']")
    if mount is None:
        raise RuntimeError("Source URDF has no as2_to_piper fixed joint")
    mount_origin = mount.find("origin")
    if mount_origin is None:
        mount_origin = ET.SubElement(mount, "origin")
    mount_origin.set("xyz", HEAD_PIPER_XYZ)
    mount_origin.set("rpy", "0 0 0")

    orin_root = ET.parse(ORIN_SOURCE).getroot()
    orin_link = orin_root.find("link[@name='jetson_agx_orin64_link']")
    orin_joint = orin_root.find("joint[@name='jetson_agx_orin64_fixed']")
    if orin_link is None or orin_joint is None:
        raise RuntimeError("Orin source link/joint not found")
    root.append(copy.deepcopy(orin_link))
    joint = copy.deepcopy(orin_joint)
    joint.find("parent").set("link", "base_link")
    joint.find("origin").set("xyz", ORIN_XYZ)
    joint.find("origin").set("rpy", "0 0 0")
    root.append(joint)

    indent(root)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(OUTPUT, encoding="utf-8", xml_declaration=True)
    report = {
        "output": str(OUTPUT.relative_to(ROOT)),
        "source_v7_urdf": str(SOURCE.relative_to(ROOT)),
        "source_v7_sha256": sha256(SOURCE),
        "orin_box_source_urdf": str(ORIN_SOURCE),
        "orin_dimensions_m": [float(v) for v in orin_link.find("visual/geometry/box").get("size").split()],
        "orin_mass_kg": float(orin_link.find("inertial/mass").get("value")),
        "orin_inertia_kg_m2": {k: float(v) for k,v in orin_link.find("inertial/inertia").attrib.items()},
        "orin_base_xyz_in_as2_base_m": [float(v) for v in ORIN_XYZ.split()],
        "orin_bottom_z_m": 0.07605,
        "orin_footprint_x_m": [-0.085, 0.025],
        "orin_footprint_y_m": [-0.055, 0.055],
        "as2_base_collision_top_z_m": 0.076,
        "base_collision_support_bounds_xy_m": {"x": [-0.1, 0.1], "y": [-0.112, 0.112]},
        "head_piper_base_xyz_in_as2_base_m": [float(v) for v in HEAD_PIPER_XYZ.split()],
        "support_note": "Orin footprint is fully inside the flat top of the AS2 base collision box. STL top-surface sampling at 121 points over the full footprint had complete coverage and z range 0.07577–0.07601 m (0.24 mm relief). Fixed-joint placement is geometric only; actual fasteners and thermal clearances remain unverified.",
        "output_sha256": sha256(OUTPUT),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

