#!/usr/bin/env bash
set -euo pipefail
umask 0002
root="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONUNBUFFERED=1
exec bash "$root/low-level-as2piper-head-orin-deploy/run_force_ablation.sh" --mode train --condition B "$@"
