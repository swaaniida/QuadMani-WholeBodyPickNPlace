#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.cache/torch_extensions}"
export MAX_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"
run_stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
export WANDB_RUN_ID="${WANDB_RUN_ID:-as2piper_vbcport_scratch_${run_stamp}}"
export WANDB_NAME="${WANDB_NAME:-plate_vbcport_${run_stamp}}"
export WANDB_DISABLE_GIT=true
python prepare_asset.py
exec python legged_gym/scripts/train_hw.py --task "${AS2_TASK:-as2piper_plate_vbc_port}" --proj_name as2piper-low --exptid "$WANDB_NAME" --headless "$@"
