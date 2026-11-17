#!/usr/bin/env bash
set -euo pipefail
umask 0002
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.wandb_deps:$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.cache/torch_extensions}"
export MAX_JOBS=2
export WANDB_ENTITY="${WANDB_ENTITY:-quadruped_capstone}"
exec python -m force_ablation.run "$@"
