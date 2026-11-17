# AS2+PiPER: audited VBC hardware port

Current default: **`as2piper_plate_vbc_port`**, from iteration 0. See
[VBC_PORT_AUDIT_KO.md](VBC_PORT_AUDIT_KO.md) for the full source audit, restored
rough terrain/domain randomization/reward settings, VBC coordinate convention,
expanded EE ranges, hardware-derived exploration, and controlled checks.

```bash
# Authenticated vbc account, project root. Preserve the prior run for comparison.
python low-level-as2piper-hw/restart_fresh.py \
  --old-run-id as2piper_hw_gaitfix_vbcsampler_scratch_20261029_134701 \
  --keep-old-run

# Fresh independent 0 -> 45000 launch; no checkpoint/transfer initialization.
WANDB_BASE_URL=https://api.wandb.ai bash low-level-as2piper-hw/launch_hw.sh \
  --max_iterations 45000
```

W&B project `as2piper-low`, new IDs `as2piper_vbcport_scratch_<UTC time>_<PID>`.
The 20-iteration smoke model is not used to initialize the long run.

## Superseded restricted sampler experiment

The following section records the previous plane/nominal/narrow-workspace task.
It is retained for comparison and is not the current default.

Independent copy of `low-level-as2piper-forward`. B1 baseline, old AS2 training,
and the forward test configuration are preserved. This is a new training task,
not a resume of the old 7000-iteration model.

That experiment started **from iteration 0**, without a gait checkpoint or arm
input transfer initialization. Default task: `as2piper_plate_walk_wbc_vbc`.
EE pose IK/body collision screening runs offline in `audit_ee_range.py`.
Training uses the original VBC spherical sampler and its ten-point EE box/floor
checker; it does not instantiate or call the additional iterative IK/OBB guard.
See [VBC_SAMPLER_FIX_KO.md](VBC_SAMPLER_FIX_KO.md) for measurements and range limits.

```bash
# Fresh 0 -> 45000 run, from the project root in the authenticated vbc account.
WANDB_BASE_URL=https://api.wandb.ai bash low-level-as2piper-hw/launch_hw.sh \
  --max_iterations 45000

# Replace only the superseded slow run (stop + remove its local/W&B records).
python low-level-as2piper-hw/restart_fresh.py \
  --old-run-id as2piper_hw_gaitfix_ee_scratch_20261029_130112
```

W&B project: `as2piper-low`. New default ID:
`as2piper_hw_gaitfix_vbcsampler_scratch_<UTC timestamp>_<PID>`.
The flat nominal task retains friction/leg-strength randomization and disables
payload/COM/gripper-mass/push randomization. It is not a ground-picking task.

## Historical gait/transfer diagnostics (not the current launch recipe)

The measured arm-held walking checkpoint is
`logs/as2piper-low/gait_pilot500_20261029/model_500.pt`.
The measured walking+arm-goal starter is
`logs/as2piper-low/walk_wbc_transfer520_20261029/model_520.pt`.
See [GAIT_FIX_KO.md](GAIT_FIX_KO.md) for actual velocity/contact measurements.

```bash
# Historical transfer diagnostic, not the current fresh-start experiment.
# Run from the project root, using an environment already authenticated to W&B.
AS2_TASK=as2piper_plate_walk_wbc_nominal bash low-level-as2piper-hw/launch_hw.sh \
  --resumeid walk_wbc_transfer520_20261029 --checkpoint 520 --max_iterations 45000

# Optional: reproduce the initial arm-held locomotion curriculum from scratch.
bash low-level-as2piper-hw/launch_gait.sh --num_envs 2048 --max_iterations 500
```

W&B project `as2piper-low`, WBC task `as2piper_plate_walk_wbc`, name
`plate_walk_wbc_<UTC timestamp>_<PID>`, fresh ID `as2piper_walk_wbc_<...>`.
The launcher prints its PID/log path and rejects immediate startup failures.
`as2piper_plate_gait` holds the arm while learning locomotion;
`as2piper_plate_walk_wbc_nominal` adds arm commands before payload/COM/push DR.
Select `AS2_TASK=as2piper_plate_walk_wbc` explicitly for the full scaled DR stage.
The naive `walk_wbc_nominal600_20261029` transfer is an unsuccessful diagnostic
branch and should not replace the validated transfer-initialized starter.
Authenticate W&B using your existing environment. No authentication material is
copied by these scripts. There is no pretrained locomotion policy in this task.
The base gait is learned, without scripted joint trajectories. This first stage
uses a physical flat plane; rough-terrain robustness and real-hardware gait are
not implied by the simulated velocity/contact evaluations.

`derive_hardware.py` creates `resources/commands/hardware.json` from the model's
mass/inertia/support loads and the available official AS2 simulation preset.
The calibration report explains equations, assumptions and implementation changes.
`train_hw.py` saves the actual runtime config, hardware specification, source and
URDF hashes for each run. Extended checkpoints also save history-encoder Adam,
learning counters and RNG states. On restart, simulator/contact/episode state is
new; extended checkpoints are not a bit-exact PhysX restart.
