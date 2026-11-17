#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="$PWD/../low-level-as2piper-head-orin/.torch_extensions"
export MAX_JOBS=2
exec python -m force_ablation.run "$@"
