#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
export MAX_JOBS=2
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLE_GIT=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
python prepare_asset.py
exec python legged_gym/scripts/train_as2.py --task as2piper_plate --proj_name as2piper-low --headless "$@"
