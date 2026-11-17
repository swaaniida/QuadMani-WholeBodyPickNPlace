#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"
export MAX_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export WANDB_DISABLE_GIT=true
export WANDB_MODE="${WANDB_MODE:-disabled}"
exec python -u train_gait.py "$@"
