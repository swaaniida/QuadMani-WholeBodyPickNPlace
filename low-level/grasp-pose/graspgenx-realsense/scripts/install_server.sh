#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/scripts/bootstrap_uv.sh"
if [[ ! -d upstream/GraspGenX/.git ]]; then
  git clone https://github.com/NVlabs/GraspGenX.git upstream/GraspGenX
  git -C upstream/GraspGenX checkout 44eb1d77d6d27bc97ee361e3afb717a2115e30a9
fi
tools/uv venv --python "$TASK_PYTHON" .venv --allow-existing
tools/uv pip install --python .venv/bin/python 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu121
tools/uv pip install --python .venv/bin/python -e './upstream/GraspGenX[serve,visualization]' pytest 'opencv-python<4.12'
mkdir -p outputs/{rgb,depth,pointcloud,grasps,benchmarks}
.venv/bin/python scripts/download_assets.py
tools/uv pip freeze --python .venv/bin/python > configs/server-freeze.txt
