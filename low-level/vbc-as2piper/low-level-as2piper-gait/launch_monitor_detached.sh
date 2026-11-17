#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd "$(dirname "$0")"
case_name="$1"
shift
mkdir -p experiments/gait_reward_proxy_v2/resume_logs
export PYTHONPATH="$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.cache/torch_extensions}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2 WANDB_MODE=disabled WANDB_CONSOLE=off WANDB_DISABLE_GIT=true
stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
logfile="$PWD/experiments/gait_reward_proxy_v2/resume_logs/eval_${case_name}_${stamp}.log"
nohup setsid python -u monitor_full_env.py "$@" >"$logfile" 2>&1 < /dev/null &
monitor_pid=$!
printf '%s\n' "$monitor_pid" >"$PWD/experiments/gait_reward_proxy_v2/resume_logs/eval_${case_name}_${stamp}.pid"
printf 'PID: %s\nLog: %s\n' "$monitor_pid" "$logfile"
