#!/usr/bin/env bash
# Optional: rig viewer already ships with a self-contained URDF and meshes.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
TARGET=upstream/realsense-ros
REV=7a87097c499e3b0b69518d54a8e7a9ac1924e87c
if [[ -e "$TARGET" ]]; then
  echo "$TARGET already exists; leaving it untouched. Check SOURCE.json / git HEAD."
  exit 0
fi
mkdir -p "$TARGET"
git -C "$TARGET" init
git -C "$TARGET" remote add origin https://github.com/realsenseai/realsense-ros.git
git -C "$TARGET" fetch --depth 1 origin "$REV"
git -C "$TARGET" checkout FETCH_HEAD -- LICENSE realsense2_description
cp local_client/assets/wrist_camera_rig/REALSENSE_SOURCE.json "$TARGET/SOURCE.json"
