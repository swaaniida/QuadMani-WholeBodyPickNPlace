#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/mujoco_mount_compare/.deps:$PWD/mujoco_mount_compare${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1
python mujoco_mount_compare/build_locomani_scenes.py
for variant in profiles_plate plate_only
do
    for scenario in standing tilt_in_place stand_walk_omni tilt_walk_omni pick_recover pick_drag pick_lift_drag
    do
        python mujoco_mount_compare/run_locomani_demo.py --variant "$variant" --scenario "$scenario" --video
    done
done
python mujoco_mount_compare/analyze_locomani_results.py
python mujoco_mount_compare/evaluate_locomani_pose.py
python mujoco_mount_compare/validate_locomani_videos.py
