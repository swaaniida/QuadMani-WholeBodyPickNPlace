#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/scripts/bootstrap_uv.sh"
tools/uv venv --python "$TASK_PYTHON" .venv-client --allow-existing
tools/uv pip install --python .venv-client/bin/python -r local_client/requirements.txt pytest
tools/uv venv --python "$TASK_PYTHON" .venv-odometry --allow-existing
tools/uv pip install --python .venv-odometry/bin/python -r local_client/requirements-temporal.txt pytest matplotlib pycollada xacro
