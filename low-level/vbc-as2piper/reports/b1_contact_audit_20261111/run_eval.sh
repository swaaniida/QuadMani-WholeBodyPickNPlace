#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd /data/team/users/suhwan/vbc-as2piper
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/data/team/users/suhwan/vbc-legacy/low-level:/data/team/users/suhwan/vbc-legacy/third_party/rsl_rl:$PWD/low-level-as2piper-head-orin/.deps:$PWD/third_party/isaacgym/python"
export TORCH_EXTENSIONS_DIR="$PWD/low-level-as2piper-head-orin/.torch_extensions"
export MAX_JOBS=2
exec python reports/b1_contact_audit_20261111/evaluate.py "$@"
