# AS2+PiPER walking policy and WBC curriculum

Independent copy of `low-level-as2piper-forward`. B1 baseline, old AS2 training,
and the forward test configuration are preserved. This is a new training task,
not a resume of the old 7000-iteration model.

The measured arm-held walking checkpoint is
`logs/as2piper-low/gait_pilot500_20261029/model_500.pt`.
The measured walking+arm-goal starter is
`logs/as2piper-low/walk_wbc_transfer520_20261029/model_520.pt`.
See [GAIT_FIX_KO.md](GAIT_FIX_KO.md) for actual velocity/contact measurements.

```bash
# Continue the validated nominal walking+arm-goal stage.
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
