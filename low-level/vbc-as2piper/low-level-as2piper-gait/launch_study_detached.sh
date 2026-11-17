#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd "$(dirname "$0")"
case_name="$1"
shift
mkdir -p experiments/gait_reward_proxy_v2/resume_logs
stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
logfile="$PWD/experiments/gait_reward_proxy_v2/resume_logs/${case_name}_${stamp}.log"
nohup setsid bash "$PWD/run_gait.sh" "$@" >"$logfile" 2>&1 < /dev/null &
study_pid=$!
printf '%s\n' "$study_pid" >"$PWD/experiments/gait_reward_proxy_v2/resume_logs/${case_name}_${stamp}.pid"
for check in 1 2 3 4; do
    sleep 1
    state="$(ps -p "$study_pid" -o stat= || true)"
    if [[ -z "$state" || "$state" == Z* ]]; then
        tail -n 12 "$logfile" >&2
        exit 1
    fi
done
printf 'PID: %s\nLog: %s\n' "$study_pid" "$logfile"
