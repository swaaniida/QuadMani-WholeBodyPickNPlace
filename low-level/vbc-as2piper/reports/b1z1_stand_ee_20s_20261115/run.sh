#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd /data/team/users/suhwan/vbc-as2piper
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
export PYTHONPATH="/data/team/users/suhwan/vbc-legacy/low-level:/data/team/users/suhwan/vbc-legacy/third_party/rsl_rl:$PWD/low-level-as2piper-head-orin/.deps:$PWD/third_party/isaacgym/python"
export TORCH_EXTENSIONS_DIR="$PWD/low-level-as2piper-head-orin/.torch_extensions"
exec /usr/bin/python3 reports/b1z1_stand_ee_20s_20261115/evaluate.py
