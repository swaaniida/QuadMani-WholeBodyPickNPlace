#!/usr/bin/env bash
set -euo pipefail
export AS2_TASK=as2piper_plate_gait
gait_stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
export WANDB_RUN_ID="${WANDB_RUN_ID:-as2piper_gait_${gait_stamp}}"
export WANDB_NAME="${WANDB_NAME:-plate_gait_${gait_stamp}}"
exec bash "$(dirname "$0")/launch_hw.sh" "$@"
