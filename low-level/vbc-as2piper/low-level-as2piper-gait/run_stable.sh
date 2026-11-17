#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.cache/torch_extensions}"
export MAX_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export WANDB_DISABLE_GIT=true
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://forge.coreweave.com/wandb}"
stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
export WANDB_RUN_ID="${WANDB_RUN_ID:-as2piper_gait_livefoot_symmetric_phase_scratch_${stamp}}"
export WANDB_NAME="${WANDB_NAME:-$WANDB_RUN_ID}"
if [[ -e "$PWD/logs/as2piper-low/$WANDB_NAME" ]]; then
  printf 'Refusing to overwrite existing run: %s\n' "$WANDB_NAME" >&2
  exit 1
fi
exec python -u legged_gym/scripts/train_hw.py --task as2piper_plate_gait_stable \
  --proj_name as2piper-low --exptid "$WANDB_NAME" --headless "$@"
