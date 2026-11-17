#!/usr/bin/env bash
set -euo pipefail
umask 0002
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
export MAX_JOBS=2
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.cache/torch_extensions}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-as2piper_plate_C_orig_20261028_01}"
export WANDB_NAME="${WANDB_NAME:-plate_only_C_original_20261028}"
export WANDB_DISABLE_GIT=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
python prepare_asset.py
exec python legged_gym/scripts/train_as2.py --task as2piper_plate --proj_name as2piper-low --exptid "$WANDB_NAME" --headless "$@"
