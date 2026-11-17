
from pathlib import Path
import copy
import shutil
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
HOME = REPO.parent
SOURCES = REPO / "robot-sources"

AS2 = SOURCES / "unitree_ros/robots/as2_description"
PIPER = SOURCES / "agx_arm_urdf/piper"

OUT = REPO / "low-level/legged_gym/resources/robots/as2piper"
URDF_DIR = OUT / "urdf"
MESH_DIR = OUT / "meshes"

URDF_DIR.mkdir(parents=True, exist_ok=True)
MESH_DIR.mkdir(parents=True, exist_ok=True)

# Copy required meshes into the combined robot asset.
shutil.copytree(
    AS2 / "meshes",
    MESH_DIR / "as2",
    dirs_exist_ok=True,
)
shutil.copytree(
    PIPER / "meshes",
    MESH_DIR / "piper",
    dirs_exist_ok=True,
)

as2 = ET.parse(AS2 / "urdf/as2.urdf").getroot()
piper = ET.parse(PIPER / "urdf/piper_description.urdf").getroot()

# Isaac Gym uses the asset base as a floating body.
# Remove AS2's explicit world/floating joint.
for elem in list(as2):
    if elem.tag == "mujoco":
        as2.remove(elem)
    elif elem.tag == "link" and elem.get("name") == "world":
        as2.remove(elem)
    elif elem.tag == "joint" and elem.get("name") == "floating_base_joint":
        as2.remove(elem)

# Rewrite AS2 mesh paths.
for mesh in as2.iter("mesh"):
    filename = mesh.get("filename")
    if filename and filename.startswith("../meshes/"):
        mesh.set(
            "filename",
            filename.replace("../meshes/", "../meshes/as2/", 1),
        )

# Insert PiPER links and joints, renaming them to avoid collisions.
mesh_prefix = "package://agx_arm_description/agx_arm_urdf/piper/meshes/"

for elem in piper:
    if elem.tag == "link" and elem.get("name") == "world":
        continue
    if elem.tag == "joint" and elem.get("name") == "world_to_base_link":
        continue
    if elem.tag not in ("link", "joint"):
        continue

    item = copy.deepcopy(elem)
    item.set("name", "piper_" + item.get("name"))

    for ref in item.iter():
        if ref.tag in ("parent", "child") and ref.get("link"):
            ref.set("link", "piper_" + ref.get("link"))

        if ref.tag == "mesh":
            filename = ref.get("filename", "")
            if not filename.startswith(mesh_prefix):
                raise ValueError("Unexpected PiPER mesh: " + filename)

            ref.set(
                "filename",
                "../meshes/piper/" + filename[len(mesh_prefix):],
            )

    as2.append(item)

# Fixed mounting joint: AS2 body -> PiPER base
mount = ET.SubElement(
    as2,
    "joint",
    {"name": "as2_to_piper", "type": "fixed"},
)

ET.SubElement(
    mount, "parent", {"link": "base_link"}
)
ET.SubElement(
    mount, "child", {"link": "piper_base_link"}
)

# Provisional mounting transform, NOT measured.
ET.SubElement(
    mount,
    "origin",
    {"xyz": "0 0 0.15", "rpy": "0 0 0"},
)

# Validate references before saving.
links = [x.get("name") for x in as2.findall("link")]
joints = as2.findall("joint")
link_set = set(links)

assert len(links) == len(link_set), "Duplicate link names"

for joint in joints:
    for tag in ("parent", "child"):
        ref = joint.find(tag)
        assert ref is not None and ref.get("link") in link_set, (
            joint.get("name"), tag
        )

missing = []
for mesh in as2.iter("mesh"):
    filename = mesh.get("filename")
    if not (URDF_DIR / filename).resolve().is_file():
        missing.append(filename)

if missing:
    raise FileNotFoundError(
        "Missing mesh files:\n" + "\n".join(sorted(set(missing)))
    )

as2.set("name", "as2piper")

output = URDF_DIR / "as2piper.urdf"
ET.ElementTree(as2).write(
    output,
    encoding="utf-8",
    xml_declaration=True,
)

movable = [
    j.get("name")
    for j in joints
    if j.get("type") in ("revolute", "continuous", "prismatic")
]

print("Saved:", output)
print("Links:", len(links))
print("Actuated joints:", len(movable))
print("Joint names:")
for i, name in enumerate(movable):
    print(f"{i:2d}: {name}")
print("Mesh paths: validated")
