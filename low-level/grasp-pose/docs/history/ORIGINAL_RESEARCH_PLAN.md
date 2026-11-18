# Autonomous Grasping with GraspGen-style Pose Generation and Learned EE Trajectory

> **Project brief / Codex development context**  
> **Status:** Research design; the modules and commands below describe the **target implementation**, not functionality that is already implemented.  
> **Primary robot:** Unitree AS2 quadruped + AgileX PiPER 6-DoF arm + UMI-style parallel-jaw gripper  
> **Perception:** Wrist-mounted RGB-D camera  
> **Core idea:** Generate a feasible 6-DoF grasp pose, learn to produce the **next EE pose** toward that target, and execute it through a **separately developed whole-body EE-pose tracking policy**.

## 1. Project objective

Develop a **semi-autonomous grasping framework** for a quadruped mobile manipulator. During whole-body teleoperation, a human moves the robot near the desired object. At the grasping stage, the system takes over the fine approach and grasp, reducing slow manual alignment and unnecessary corrections.

The autonomous grasping pipeline must:

1. Observe the object through the **wrist RGB-D camera**.
2. Extract an object/scene point cloud, then generate multiple candidate **6-DoF grasp poses**, in the style of **GraspGen / GraspGen-X**.
3. Filter and rank candidates for the **actual PiPER-mounted UMI-style gripper**, including collisions and reachable poses.
4. Given the **current EE pose** and **selected grasp EE pose**, infer a **next EE pose command** using a learned **Grasp Trajectory Policy**.
5. Repeatedly feed the predicted EE commands to an external **Whole-Body Goal-Reaching / EE Tracking Policy** that coordinates the AS2 legs and PiPER arm.
6. Close the gripper at the appropriate moment; verify grasp success; either return control to the human or execute a subsequent subtask.

**Key contribution to focus on:** the learned **grasp-target-conditioned next-EE-pose policy**, not building a new grasp-pose generator or implementing the WBC from scratch in this repository.

### Scope and non-goals

**In scope**
- RGB-D-to-point-cloud preprocessing and grasp-generator integration.
- UMI-style gripper frame/geometry adaptation.
- Candidate grasp ranking, safety checks, and target selection.
- Supervised learning of an EE-level **next-pose policy** in simulation.
- Closed-loop rollout, evaluation, and logging.
- Clean interfaces to teleoperation and a separately maintained WBC policy.

**Out of scope for the first working version**
- Re-training GraspGen from scratch.
- Re-implementing or training the AS2+PiPER whole-body controller here.
- Full autonomous navigation, object transport, and long-horizon VLA task planning.
- Guaranteeing collision-free trajectories using a pose-only policy without environmental context.
- Assuming the original passive handheld UMI gripper is directly motorized or compatible with PiPER.

The initial milestone is **one object, a clear tabletop/floor grasp, a validated gripper, and a reliably tracked EE command**. Scale to oriented/cluttered grasps after the interfaces work.

## 2. System overview

```text
                            HUMAN TELEOPERATION
                                     |
                              EE pose command
                                     |
                           [Mode / Safety Manager]
                                     |  manual mode
                                     v
                        +-------------------------+
                        | WBC EE Goal-Reaching     |------> AS2 legs + PiPER arm
                        | Policy (external)       |
                        +-------------------------+
                                     ^
                                     |  next EE pose command
                                     |
                          autonomous grasp mode
                                     |
 Wrist RGB-D camera ----> RGB-D / object segmentation / point cloud
                                     |
                                     v
                          [Grasp Pose Generator]
                           GraspGen-style backend
                                     |
                            candidate 6-DoF grasps
                                     |
                                     v
                          [Grasp Selection / Filter]
                     gripper geometry + collision + reachability
                                     |
                           selected grasp EE pose
                                     |
                                     v
                        [Grasp Trajectory Policy]
                 current EE pose + grasp EE pose (+ state)
                                     |
                          predicted NEXT EE pose
                                     |
                                     +-------------> WBC EE Goal-Reaching Policy
                                     |
                            grasp trigger / close
                                     |
                            success verification
                                     |
                       hand-back to teleoperation
```

**Keep the three learned/geometric components independent:**

| Component | Main input | Main output | Training / ownership |
|---|---|---|---|
| Grasp Pose Generator | Segmented object point cloud, scene geometry, gripper description | Candidate 6-DoF grasp poses + scores | Initially reuse existing GraspGen-family inference |
| **Grasp Trajectory Policy (main work)** | Current EE pose, selected target grasp EE pose, optional robot/scene state | **Next EE pose** (or an EE-relative increment converted to a next pose) | **Train and validate in this project** |
| Whole-Body Goal-Reaching Policy | Robot proprioception + commanded EE pose | Low-level arm/leg actions | Developed separately; **only integrate through an adapter** |

A grasp pose is a **goal configuration**. It is **not** a collision-free approach trajectory. The trajectory network is a separate module, and WBC tracks its commands rather than deciding which object or grasp to select.

## 3. Hardware assumptions and integration risks

- **Base:** Unitree AS2, moving quadruped body.
- **Arm:** AgileX PiPER, 6-DoF.
- **End effector:** Actuated UMI-style parallel-jaw gripper attached to PiPER. The original UMI is a **handheld demonstration gripper**; a robot-actuated design (e.g., Actuated UMI) or custom actuator/interface must be verified before hardware deployment.
- **Camera:** Wrist-mounted RGB-D camera. GraspGen-family models expect a **3D point cloud**; raw RGB images alone are insufficient for the planned point-cloud-based inference.
- **Compute:** May use an external GPU server for perception/learning and a robot computer for tracking. Do **not** assume particular GPU, CUDA, ROS, network latency, camera model, or WBC action rate without checking the target system.

### Critical compatibility check: GraspGen vs. UMI gripper

The public **GraspGen** repository provides pretrained gripper configurations for **Franka Panda, Robotiq 2F-140, and suction**, **not** the custom UMI gripper. Do not treat its checkpoint as a drop-in UMI-gripper solution. **GraspGen-X** is a candidate because it supports cross-embodiment grasp generation from a gripper representation. Both still require verification of geometry, grasp reference frames, calibration, and real physical feasibility.

Implementation strategy:

1. Build a **backend-agnostic** `GraspGenerator` interface; first test it with synthetic poses and/or a publicly supported GraspGen gripper.
2. Inspect the **actual robot-mounted gripper** (URDF/CAD, finger mesh, jaw aperture/stroke, actuation, TCP, swept volume, contact surfaces, collision geometry).
3. Evaluate GraspGen-X on this gripper. If unsuccessful, consider a compatible-gripper baseline or custom training/finetuning; document any approximation explicitly.
4. Never assume a high discriminator score means the grasp is reachable, collision-free, or executable by PiPER/AS2.

## 4. Coordinate frames and transforms — mandatory convention

Use **SE(3) rigid transforms with explicit source and destination frames**. Define `T_A_B` to transform 3D coordinates expressed in frame **B** into frame **A**.

Required frames:

- `O`: odometry/world frame (if reliable); may not exist in the first static-base experiment.
- `B`: AS2 base frame.
- `E`: PiPER EE command frame (**must match the WBC contract**).
- `C`: wrist camera optical frame.
- `H`: gripper's grasp/reference frame, explicitly calibrated relative to `E` (may differ from the flange or TCP).

Typical transforms:

```text
T_B_C = T_B_E @ T_E_C
T_B_H_goal = T_B_C @ T_C_H_goal
T_B_E_goal = T_B_H_goal @ inverse(T_E_H)
```

If base motion matters and odometry is available, transform the selected target into `O` at the **image timestamp**, and recompute its current `B`-frame command using the latest `T_O_B` at every control step. If reliable odometry is unavailable, **restrict the first precision-grasp experiments to a stationary base**, or reobserve/replan with the wrist camera. Never treat a stale base-frame target as a fixed world pose while the base moves.

**Do not assume the grasp-generator output frame equals `H` or `E`.** Inspect its documentation and create an explicit `generator_grasp_frame -> H -> E` adapter. All transforms must be checked with a visual marker and numeric tests before integration.

Conventions:

- Position: meters. Angles: radians. Time: seconds.
- Rotation storage: normalized quaternion `(x, y, z, w)` at API boundaries, with an explicitly documented choice; transformation matrices for composed geometry.
- Network rotation representation: prefer a continuous 6D rotation representation or SE(3) log / twist, instead of unwrapped Euler angles.
- Camera intrinsics/extrinsics, robot EE/TCP/gripper offsets, timestamps, and calibration versions must be saved with each episode.
- Align camera frames and robot states temporally; reject stale observations or stale commands.

## 5. Module A — RGB-D grasp pose generation

### Input and preprocessing

1. Read synchronized RGB, depth, intrinsics, and camera timestamp.
2. Detect/segment the target object (initially provide a manual mask or ground-truth simulator mask; swap in a learned segmentation model later).
3. Back-project depth to a **metric object point cloud**, after depth filtering and removal of invalid values.
4. Retain a **scene point cloud** for environmental collision checks.
5. Normalize/resize/sample the point cloud per the selected backend's requirements; preserve transforms to recover the original metric frame.

### Inference and selection

- Generate `N` grasp candidates with confidence/quality scores.
- Convert candidate grasp frames into the **physical UMI gripper EE target** using calibrated transforms.
- Reject poses that violate jaw opening/geometry, approach constraints, collision checks, PiPER reachability, singularity/joint limits, or WBC-provided feasible regions.
- Rank remaining grasps by generator score **and** feasibility, required EE travel, orientation change, clearance, and optional stability margin.
- Produce a selected `target_ee_pose` with score, timestamp, and provenance.
- If no candidate is safe or feasible: **do not move**; reobserve, request a viewpoint change, or hand control to the operator.

A grasp may require a **pre-grasp pose** along the calibrated gripper approach axis. Keep `pregrasp -> final grasp -> close -> lift/verify` as separate phases. The exact approach-axis sign is **not** to be hard-coded until tested against the gripper's local coordinate convention.

## 6. Module B — learned Grasp Trajectory Policy (primary development)

### Problem statement

At time `t`, the policy receives the actual EE pose and a selected grasp target and predicts the **next EE pose command**:

```text
policy observation:
    T_B_E_actual(t)         current actual EE pose
    T_B_E_goal(t)           selected grasp target
    optional proprioception / previous command / scene embedding

policy output:
    T_B_E_cmd(t+1)          next commanded EE pose
```

Recommended relative formulation (first prototype):

```text
relative_goal = inverse(T_B_E_actual) @ T_B_E_goal
state = [relative_goal_translation,
         relative_goal_rotation_representation,
         optional_previous_action_or_robot_state]

action = delta_ee_in_local_frame  # 3D translation + rotation increment
T_B_E_cmd_next = T_B_E_actual @ Exp_SE3(action)
```

An alternative is direct prediction of the next pose; if used, keep the same public `next_ee_pose` interface. Relative actions reduce dependence on the global frame, but **do not remove collision or kinematic constraints**. Document the sampling period `dt` and whether actions are position deltas, twists, or absolute poses; never mix these in the WBC interface.

### First baseline

- **Policy:** small MLP, goal-conditioned and receding-horizon, predicting a bounded incremental EE motion at each step.
- **Input minimum:** translation and rotation error from actual EE to selected grasp EE pose.
- **Target supervision:** next EE pose along a **kinematically and collision-checked teacher trajectory**; the teacher can initially use interpolation plus IK and collision checking in simple scenes, then a proper Cartesian/whole-body motion planner for harder ones.
- **Data:** randomized valid initial EE poses and grasp targets in the **real reachable workspace**, including position/orientation perturbations and varying starting distances.
- **Loss (initial):** translation-action loss + rotation-geodesic/SE(3) error + optional smoothness and final-pose loss. Log each term separately.
- **Rollouts:** train on demonstrations, then evaluate **closed loop** from perturbed initial states; consider DAgger/rollout aggregation to mitigate compounding errors.

Do **not** train only from perfect analytical interpolations without testing actual execution errors. The trajectory model must condition on the **measured actual** EE pose, not blindly integrate its own previous predictions.

### What the network cannot infer from pose alone

Two identical current/target poses can require different paths in cluttered environments. A pose-only MLP cannot in general know where obstacles are. Therefore:

- **Stage 1:** clear, uncluttered approach; external collision validation and bounded actions.
- **Stage 2:** add scene point-cloud features / occupancy representation or a planner's intermediate guidance for obstacle-aware approach.
- **Stage 3:** assess generalization to unseen object geometry and clutter.

Use a planner/intervention when necessary; never present the pose-only MLP as a universal safe motion planner.

### Control loop and stopping behavior

```python
while autonomous_grasp_enabled:
    observation = get_time_aligned_robot_and_camera_state()
    if unsafe_or_stale(observation):
        stop_and_return_control()
        break

    goal = update_goal_in_current_base_frame(selected_grasp, observation)
    next_ee_pose = trajectory_policy.predict(observation.ee_pose, goal)
    next_ee_pose = safety_filter(next_ee_pose, observation)
    wbc_adapter.send_ee_pose_command(next_ee_pose)

    if reached_pregrasp_or_final_grasp(observation, goal):
        advance_grasp_phase()

    if verified_grasp_or_timeout_or_failure():
        exit_autonomous_mode()
        break
```

This is **illustrative pseudocode**, not runnable code. The actual state machine must distinguish perception, pre-grasp, approach, gripper closure, success verification, abort, and teleoperation hand-back.

## 7. Module C — Whole-Body EE Tracking Policy (external dependency)

The whole-body policy is **developed independently** (VBC-style simulation RL goal-reaching for AS2 + PiPER). This repository should use it as a black box via a tested interface.

Proposed interface contract:

```python
class WholeBodyEETracker:
    def reset(self) -> None: ...
    def get_state(self) -> RobotState: ...          # current EE pose + timestamps + proprioception
    def command_ee_pose(self, target: PoseStamped) -> None: ...
    def stop(self) -> None: ...
    def healthy(self) -> bool: ...
```

The adapter must enforce:

- Correct `B`/`O` pose frame, TCP, and timestamp.
- Explicit control period and maximum pose-command age.
- Pose step / linear velocity / angular velocity / acceleration limits.
- Observable tracking error and fault reporting.
- Robot-level joint/torque/base-stability/collision checks where available.
- A **safe stop** on timeout, lost WBC communication, invalid pose, or operator emergency stop.

The WBC should use **actual tracking feedback**. Do not assume a commanded pose was reached. If the existing WBC repository exposes a different API, adapt it at this boundary without rewriting WBC or duplicating its training code.

## 8. Shared-autonomy state machine

```text
MANUAL_TELEOP
    |  user requests autonomous grasp
    v
OBSERVE_TARGET -> GENERATE_GRASPS -> FILTER_AND_SELECT
    |                                 |
    |                                 +-- no valid grasp --> MANUAL_TELEOP
    v
MOVE_TO_PREGRASP -> APPROACH_GRASP -> CLOSE_GRIPPER
    |                                  |
    +---------- failure / timeout ----+----> ABORT / SAFE_STOP
                                       |
                                       v
                                 VERIFY_GRASP
                                       |
                           success -> MANUAL_TELEOP / next subtask
```

- Human commands and autonomous commands must never be sent concurrently.
- Start from the **latest measured EE pose** when switching modes; prohibit discontinuous command jumps.
- Human cancel/e-stop preempts autonomous motion immediately.
- Do not close the gripper based solely on the network output; require phase-specific geometric/feedback checks.
- Log command ownership, switching timestamps, reasons for failure, and grasp result.

## 9. Data and training protocol

### Suggested training record

Use a versioned episode schema, e.g. `npz`/Parquet for trajectories plus JSON/YAML metadata. Exact storage implementation can change; the **semantics must not**.

```yaml
episode_id: string
scene_id: string
robot_config_id: string
gripper_config_id: string
calibration_id: string
sim_or_real: sim
frames:
  ee_pose: B
  grasp_target_pose: B
  camera_pose: B
units: {distance: m, angle: rad, time: s}
per_step:
  - timestamp_s
  - actual_ee_pose          # position[3], quat_xyzw[4]
  - target_grasp_ee_pose    # position[3], quat_xyzw[4]
  - teacher_next_ee_pose    # training label
  - policy_next_ee_pose     # when evaluating
  - executed_ee_pose       # actual tracked result
  - gripper_width_or_state
  - robot_state_optional
  - collision_or_safety_flags
per_episode:
  - target_reached
  - grasp_success
  - abort_reason
```

*The YAML above is an illustrative field inventory, not a literal fixed serialization format.* Save camera images/point clouds separately with frame timestamps and linkage, if captured.

### Simulation sampling

1. Load verified AS2+PiPER+UMI-like gripper geometry/URDF (or a clearly labeled proxy gripper, before real geometry is ready).
2. Randomize **reachable** EE initial poses and target grasp poses; filter invalid samples.
3. Compute collision-checked reference trajectories; log `pose_t -> pose_{t+1}` examples.
4. Train a small pose-conditioned policy; normalize translation and rotation error with saved statistics.
5. Evaluate on **unseen initial poses, target poses, and objects/scenes**. Split by scene/object as well as trajectory to avoid leakage.
6. Integrate with the real WBC tracking interface in sim; compare commanded vs. measured next EE pose and close-loop failures.

Avoid reporting success based only on supervised prediction loss. It must work when rolled out through the WBC.

## 10. Evaluation and baselines

### Module-level metrics

| Module | Evaluation |
|---|---|
| Grasp pose generation | Valid candidate rate, physical-gripper compatibility, collision-free/reachable candidate fraction, candidate generation latency |
| Next-EE policy | EE translation error (m), rotation geodesic error (deg), target reach rate, convergence time, path length, velocity/acceleration/jerk, safety interventions |
| WBC tracking | Position/orientation tracking error while **standing** and **walking**, command latency, instability and stop events |
| End-to-end grasp | Grasp success rate, time from autonomous switch to grasp, completion time, abort rate |
| Shared autonomy | Human teleop time, effective demonstrations per unit time, trajectory smoothness |

### Experiments

1. **Easy grasp:** isolated object on a flat surface — minimum viable baseline.
2. **Oriented grasp:** object requires significant gripper rotation — evaluate 6-DoF correction.
3. **Complex-object grasp:** difficult geometry / partial observation — measure generalization.
4. **Standing vs. walking:** first validate standing; then quantify whether moving-base WBC and frame tracking remain accurate.

Baseline comparisons for the broader research (when available):

- **Full manual teleoperation** (human performs fine grasp).
- **UMI-collected learned grasping policy + WBC** (separate optional comparison; **not** assumed already trained).
- **GraspGen-family goal + simple geometric/interpolated trajectory + WBC** (critical ablation).
- **OURS: GraspGen-family goal + learned next-EE trajectory policy + WBC**.

For downstream imitation-learning studies, compare data collected by each approach at matched demonstration count or matched frame budget; do not conflate throughput improvements with policy-quality gains.

## 11. Proposed repository structure

> Suggested structure only. **Inspect the real repository before creating/renaming files.** Keep existing AS2/PiPER/WBC code intact.

```text
.
├── README.md
├── configs/
│   ├── robot.yaml              # frames, geometry, rates (validated values only)
│   ├── gripper.yaml            # geometry, TCP, opening limits, actuator
│   ├── perception.yaml
│   ├── grasp_generator.yaml
│   ├── trajectory_policy.yaml
│   └── safety.yaml
├── src/
│   ├── interfaces/             # typed Pose, Frame, State, GraspCandidate APIs
│   ├── calibration/            # camera/EE/gripper transforms
│   ├── perception/             # RGB-D -> point clouds, masks
│   ├── grasp_generation/       # backend adapters, filtering, ranking
│   ├── trajectory_policy/      # dataset, MLP, training, inference
│   ├── control/                # WBC adapter, command filtering
│   ├── autonomy/               # mode manager, grasp state machine
│   └── evaluation/             # metrics, plots, episode logging
├── scripts/
│   ├── inspect_robot.py
│   ├── validate_frames.py
│   ├── demo_grasp_generation.py
│   ├── generate_trajectory_data.py
│   ├── train_trajectory_policy.py
│   ├── eval_trajectory_policy.py
│   └── run_grasp_demo.py
├── tests/
│   ├── test_transforms.py
│   ├── test_gripper_frame_adapter.py
│   ├── test_next_pose_policy.py
│   ├── test_safety_filter.py
│   └── test_mode_switch.py
└── docs/
    ├── INTERFACES.md
    ├── CALIBRATION.md
    ├── EXPERIMENTS.md
    └── DECISIONS.md
```

### Proposed commands — not yet available

```bash
# These are target CLI entry points, NOT claims that scripts already exist.
python scripts/inspect_robot.py --config configs/robot.yaml
python scripts/validate_frames.py --config configs/robot.yaml
python scripts/demo_grasp_generation.py --config configs/grasp_generator.yaml
python scripts/generate_trajectory_data.py --config configs/trajectory_policy.yaml
python scripts/train_trajectory_policy.py --config configs/trajectory_policy.yaml
python scripts/eval_trajectory_policy.py --config configs/trajectory_policy.yaml
```

Use reproducible configs, seeded experiments, logged dependency versions, checkpoint metadata and evaluation artifacts. Do not invent success metrics or hard-code unmeasured hardware limits.

## 12. Development order / Codex tasks

**P0 — Inspect and confirm**
- [ ] Inspect the repository, installed dependencies, existing PiPER/AS2 models, WBC interface, and available sim.
- [ ] Confirm UMI **robot-actuated** gripper design, gripper URDF/CAD/TCP and calibration requirements.
- [ ] Choose grasp generation backend (GraspGen / GraspGen-X / mock baseline) and record why.
- [ ] Record unresolved hardware limits and available compute in `docs/DECISIONS.md`.

**P1 — Interfaces and geometry**
- [ ] Implement typed pose/transform and grasp-candidate data structures.
- [ ] Implement and test `T_B_C`, `T_B_H_goal`, and `T_B_E_goal` conversions.
- [ ] Validate physical gripper approach-axis/TCP convention; visualize test grasps.
- [ ] Implement mock WBC tracker and collision/safety checks for dry runs.

**P2 — Grasp goal generation**
- [ ] Implement RGB-D point-cloud loading and mask-based object extraction.
- [ ] Build grasp-generator adapter with synthetic or real checkpoint output.
- [ ] Add filtering for frame conversion, reachability, gripper geometry, and collision.
- [ ] Export a chosen target pose with a reproducible debug visualization.

**P3 — Next-EE trajectory policy: primary deliverable**
- [ ] Implement teacher trajectory dataset generation in simulation.
- [ ] Train MLP to predict next EE increment/pose from the measured current EE pose and selected target.
- [ ] Evaluate open-loop errors **and closed-loop convergence** from randomized initial poses.
- [ ] Add reference geometric-trajectory baseline and smoothness comparison.
- [ ] Connect policy to mock WBC, then to the actual WBC adapter in simulation.

**P4 — Integrated autonomous grasp**
- [ ] Implement teleop/autonomy handoff and finite-state grasp executor.
- [ ] Add pregrasp, approach, gripper close, verification, abort/timeout, and operator override.
- [ ] Measure standing-mode success before walking-mode and real deployment.
- [ ] Run real hardware only after calibration, safety checks, and supervised low-speed trials.

**P5 — Research evaluation**
- [ ] Run easy/oriented/complex-object experiments with repeatable seeds and trial counts.
- [ ] Record grasp success, end-to-end time, policy latency, EE tracking error, smoothness and human intervention.
- [ ] Compare **learned next-EE trajectory** against **geometric next-EE trajectory**, using the same grasp targets and WBC.
- [ ] Optionally add full-manual and UMI-collected-policy comparisons when those implementations exist.

## 13. Acceptance criteria for the first working prototype

A first prototype is **done** only if:

- [ ] A wrist-camera RGB-D frame produces a metric point cloud with a documented camera/EE transform.
- [ ] A candidate grasp pose can be transformed to the **correct PiPER commanded EE frame**, with unit tests.
- [ ] An initial measured EE pose and a valid selected grasp pose produce a **sequence of bounded next EE commands** from the trained network.
- [ ] A simulated WBC tracks those commands and reaches the grasp pose within **configurable, reported** position/orientation tolerances.
- [ ] An approach, grip-close, and explicit success/failure path runs with a simulated gripper.
- [ ] An invalid target, missing data, unsafe action, or timeout leads to a safe abort rather than continued motion.
- [ ] Geometric trajectory and trained policy baselines can be compared on the **same scenarios**.
- [ ] All reported quantitative results come from executed experiments, not assumptions.

## 14. Instructions for Codex agents

1. **Read this README, inspect existing code, and summarize actual repository state before editing.** Distinguish existing modules from proposed ones.
2. **Focus first on the Grasp Trajectory Policy** and its data/pose interfaces; WBC training is outside this workstream.
3. Prefer small, testable, independently runnable modules and dependency injection (mock camera, mock grasp generator, mock WBC).
4. Never silently substitute a supported GraspGen gripper for the physical UMI gripper. Label any proxy/model mismatch in logs and docs.
5. Avoid irreversible changes to the existing WBC repo, robot configs, or hardware. Do not send real robot commands without an explicit hardware run request and confirmed safety gating.
6. Maintain frame labels, metric units, timestamps, deterministic training seeds, and calibration provenance across components.
7. Do not write fake data, metric values, sensor measurements, successful grasps, calibrated transforms, hardware specifications, or unverified shell commands.
8. After each implementation milestone: run tests, report what passed/failed, list the changed files, and update `docs/DECISIONS.md` with architectural changes and unresolved blockers.
9. If a choice is unspecified, implement the simplest **simulation-first** baseline, expose it via config, and document the assumption; do not overengineer the first milestone.
10. Use the separately developed AS2+PiPER WBC **only through an explicit adapter**; do not couple the learned EE trajectory network to WBC internals.

### Suggested first Codex task

> Inspect the working repository and identify available AS2+PiPER/UMI gripper geometry, camera interfaces, simulator, and whole-body tracking policy APIs. Then implement the frame-safe data structures, transform tests, and a **mock-input goal-conditioned MLP** that predicts the next EE pose. Generate a minimal simulated teacher dataset, train the MLP, and evaluate closed-loop convergence against a geometric baseline. Do not control the physical robot yet. Report unresolved calibration/GraspGen-backend decisions separately.

## 15. Research references

- **GraspGen** — NVIDIA, *A Diffusion-based Framework for 6-DOF Grasping with On-Generator Training* (ICRA 2026): https://github.com/NVlabs/GraspGen
- **GraspGen-X / GraspGenX** — NVIDIA, cross-embodiment grasp generation (CVPR 2026): https://github.com/NVlabs/GraspGenX
- **UMI** — *Universal Manipulation Interface* (RSS 2024): https://umi-gripper.github.io/
- **Actuated UMI gripper** — robot-actuated gripper hardware concept: https://actuated-umi.github.io/
- **VBC** — *Visual Whole-Body Control* (CoRL 2024); EE goal tracking inspiration.
- **WildLMa** — whole-body teleoperation / data collection context.
- **Research presentation** — `캡스톤 1차발표 최종(1).pdf`, especially system overview (slides 16–23) and experiments (slides 27–29). This presentation is a **design reference**, not proof of implemented code or hardware validation.

---

**One-sentence specification:** From a wrist-camera point cloud, generate a **UMI-gripper-compatible 6-DoF grasp target**; use a **learned current-pose + target-pose -> next-EE-pose policy** to iteratively command a separately trained **whole-body EE tracking controller** until an autonomous grasp is completed safely.
