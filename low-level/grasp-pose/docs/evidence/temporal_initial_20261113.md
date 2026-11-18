# Continuous ICP / Ground / Temporal selection implementation

Date: 2026-11-13. All code and data remain under grasp-pose.

## Delivered

- New opt-in entry `local_client/main_temporal.py`, default 15Hz upload cap, click-start isolated recording, reset/reselection opens a new take. Old `main_tracking.py` remains available unchanged.
- Background-only continuous point-to-plane ICP, bounded latest-frame queue, metric transforms keyed by original capture Frame ID, sensor timestamp priority, geometric quality/normal-observability gates, bounded older-keyframe realignment.
- CPU worker executes in parallel with existing A100 inference; no extra RGB-D uploads, no model/server or Python environment modifications.
- RANSAC support-plane validation and exact plane signed-distance test of the official open franka_umi mesh vertices. 3mm+plane-RMSE clearance. No pose correction/snapping. No valid plane or no surviving candidate produces No Valid Grasp.
- Same original candidates: baseline 0.6C+0.1R+0.25P+0.05Y and new 0.6C+0.4T. Temporal is exp(-0.5*((d/.03)^2+(theta/20deg)^2)) in the ICP reference. First/missing/untrusted ICP, previous selection missing, or capture gap>1s -> confidence-only. All actual scores lie in [0,1]; unavailable temporal is blank/NaN.
- Persist raw candidates, baseline/new top1 indices and camera positions/quaternions, scores, plane clearance/mask, exact capture transforms, compensated deltas, processing/network/model timing fields and throughput. Comparison replay shows magenta baseline vs green new CAD, fixed fused object/current cloud, faint camera trajectory/current frustum, supporting plane, timeline and speed/loop controls.

## Real-data validation (no live model requests)

Full suite before final clock fixture: 61 passed. Latest focused temporal tests: 8 passed, including actual Open3D point-to-plane transform/observability, support plane, official CAD half-space, pose preservation and confidence fallback. Regression model/client source checksums unchanged for five legacy files.

Recorded actual-model integration: 2 selected poses, original cached candidate arrays unchanged, first confidence-only and second temporal-enabled. Both comparison frames rendered via CPU Viser APIs. Source is a repeated real static RGB-D frame and cached real SAM/GraspGenX results; not synthetic inference and not fresh model throughput. Raw archive arrays contain no object/pickle payloads.

Actual moving scene check: 5/5 accepted real Redwood RGB-D frames; keyframe realignment exercised. No target-mask annotations or motion ground truth in this check; this is not a dynamic D435i grasp evaluation.

Warm-up 10 + 100 static RGB-D repeats on server CPU, two CPU threads:

| Scope | Mean ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| Full background/plane/ICP compute | 46.30 | 45.73 | 50.33 |
| ICP registration only | 3.78 | 3.10 | 6.12 |
| Plane RANSAC/validation | 14.36 | 14.25 | 15.15 |

Compute-only reciprocal mean: 21.60 FPS. Excludes USB, network, SAM/AOTT/GraspGenX inference, recording and browser. This is NOT measured Windows streaming or grasp-pose FPS. Per-run live pose FPS and full latency are recorded when Windows runs main_temporal.py. No assertion of stability improvement is made from repeated static inputs.

## Limits

Without gravity calibration the largest verified support plane is only a ground hypothesis; walls/shelves can invalidate this assumption. Plane ambiguity/missing plane fails closed. Geometry is the supplied official static-open franka_umi visual CAD, not calibrated PiPER hardware/EE or robot arm/trajectory collision checking. Keyframe realignment limits relative chains, but is not global SLAM/loop closure and cannot eliminate drift. Flat/textureless scenes may not constrain 6-DoF camera motion. Windows moving-camera hardware performance, end-to-end FPS and real-world comparative stability need a new actual recording.

## Reproduce

```
OMP_NUM_THREADS=2 .venv-odometry/bin/python scripts/benchmark_temporal_replay.py --frames 100 --warmup 10
OMP_NUM_THREADS=2 .venv-odometry/bin/python scripts/check_temporal_recording.py
OMP_NUM_THREADS=2 .venv-odometry/bin/python scripts/check_temporal_motion.py
```

Windows commands, knobs, units and output definitions: `local_client/TEMPORAL_SELECTION.md`.

## Replay visualization correction

Raw camera IDs are not consecutive at tracking/upload rates below camera FPS. The previous viewer incorrectly required raw-ID increments of one, removing legitimate trajectory segments. Replay now restores the prior fixed-object RGB-D reconstruction behavior using archived frames, draws valid processed-frame adjacency/segments, and retains the fixed object and full path during a rejected current frame. It does not switch to a local camera display or guess missing poses. Replay registration is separate from, and never changes, the original background ICP used for selection. Two regression tests passed; cached real RGB-D recording/reconstruction/comparison render smoke passed.
