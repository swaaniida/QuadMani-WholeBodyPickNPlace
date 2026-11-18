#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec .venv/bin/python scripts/analyze_obb_replay.py "$@"
