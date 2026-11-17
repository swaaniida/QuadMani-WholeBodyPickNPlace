#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/mujoco_mount_compare/.deps:$PWD/mujoco_mount_compare${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1
python mujoco_mount_compare/build_locomani_scenes.py
for variant in plate_only profiles_plate
do
    for scenario in tilt_in_place pick_recover pick_drag
    do
        python mujoco_mount_compare/run_locomani_demo.py --variant "$variant" --scenario "$scenario" --video --quick
    done
done
python mujoco_mount_compare/validate_locomani_videos.py --quick
