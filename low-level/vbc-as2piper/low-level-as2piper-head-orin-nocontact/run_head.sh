#!/usr/bin/env bash
set -euo pipefail
umask 0002
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.cache/torch_extensions}"
export MAX_JOBS=2
exec python legged_gym/scripts/train.py --task as2piper_head_orin_nocontact --proj_name as2piper-low --headless "$@"
