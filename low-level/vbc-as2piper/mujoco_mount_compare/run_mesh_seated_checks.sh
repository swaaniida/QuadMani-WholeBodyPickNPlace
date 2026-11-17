#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/mujoco_mount_compare/.deps${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1
python mujoco_mount_compare/build_verified_mounts.py
python mujoco_mount_compare/verify_mounts.py
python tools/check_umi_gripper_v7.py --skip-isaacgym
for variant in profiles_plate plate_only
do
    for mode in stand crouch arm
    do
        python mujoco_mount_compare/run_verified_demo.py --variant "$variant" --mode "$mode" --seconds 5 --video
    done
done
python mujoco_mount_compare/summarize_verified.py
