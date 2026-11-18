#!/usr/bin/env bash
# Source from installers. All downloads remain inside this project.
set -euo pipefail
export UV_PYTHON_INSTALL_DIR="$PROJECT_DIR/.runtime/managed-python"
export UV_CACHE_DIR="$PROJECT_DIR/.cache/uv"
if [[ ! -x "$PROJECT_DIR/tools/uv" ]]; then
  mkdir -p "$PROJECT_DIR/tools"
  curl -LsSf --fail https://astral.sh/uv/0.13.7/install.sh -o "$PROJECT_DIR/tools/uv-install.sh"
  UV_INSTALL_DIR="$PROJECT_DIR/tools" UV_NO_MODIFY_PATH=1 sh "$PROJECT_DIR/tools/uv-install.sh"
fi
TASK_PYTHON=3.10
if [[ -x "$PROJECT_DIR/.runtime/python/bin/python3.10" ]]; then
  TASK_PYTHON="$PROJECT_DIR/.runtime/python/bin/python3.10"
fi
