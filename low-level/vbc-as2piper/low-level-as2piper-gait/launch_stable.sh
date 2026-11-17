#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd "$(dirname "$0")"
mkdir -p launch_logs
stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
logfile="$PWD/launch_logs/train_${stamp}.log"
nohup setsid bash "$PWD/run_stable.sh" "$@" >"$logfile" 2>&1 < /dev/null &
training_pid=$!
printf '%s\n' "$training_pid" >"$PWD/launch_logs/train_${stamp}.pid"
# Report immediate initialization failures instead of presenting a stale PID.
for check in 1 2 3 4 5 6 7 8; do
    sleep 1
    training_state="$(ps -p "$training_pid" -o stat= || true)"
    if [[ -z "$training_state" || "$training_state" == Z* ]]; then
        printf 'Training exited during initialization. Log: %s\n' "$logfile" >&2
        tail -n 12 "$logfile" >&2
        exit 1
    fi
done
printf 'PID: %s\nLog: %s\n' "$training_pid" "$logfile"
