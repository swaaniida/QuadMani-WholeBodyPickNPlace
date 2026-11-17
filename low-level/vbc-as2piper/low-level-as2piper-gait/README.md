# AS2 + PiPER gait reward study

Independent copy of `../low-level-as2piper-hw`. The preserved baseline task is
`as2piper_plate_vbc_port` in that directory; see its `BASELINE_VBC_PORT.md`.
The baseline includes the documented contact sign correction and inherited
defects. It is not identical to upstream B1 VBC.

Public training entry: `run_stable.sh --max_iterations 45000`, task
`as2piper_plate_gait_stable`, project `as2piper-low`. This entry uses the original
rough/full-DR defaults and target-iteration semantics. W&B is online by default;
the local ablation/fresh studies below use W&B disabled.

## Matched reward ablations

| Preset | Live foot velocity | Working contact-change penalty | Four-foot stop target | Symmetric speed/yaw tracking | Phase clearance/air time |
|---|---|---|---|---|---|
| original | No | No | No | No | No |
| velocity | Yes | No | No | No | No |
| jerk | No | Yes | No | No | No |
| stop | No | No | Yes | No | No |
| data_stop | Yes | Yes | Yes | No | No |
| speed | Yes | Yes | Yes | Yes | No |
| stable | Yes | Yes | Yes | Yes | Yes |

Each paired trial starts with the same copied model313, Adam/history optimizer,
seed7,1024 environments and120 additional updates. The controlled environment
uses a plane with DR disabled; original +/-20% joint reset and original random
EE goals/command distribution remain. This is a recovery ablation, not seven
independent scratch-training convergence demonstrations. The selected candidate
also needs separate scratch and terrain/DR verification.

`speed` changes the velocity term to symmetric planar error (sigma0.20m/s),
and yaw to sigma0.35rad/s with weight1.0. The weight of planar tracking remains2.
`stable` additionally replaces front-foot world-height norm with four-foot,
phase-dependent4cm clearance; uses all four feet with0.25s swing timing at2Hz
and bounded air-time credit; reduces air-time weight2→0.5 and walking_dof1.5→0.5.
These bundled groups test combinations, not the isolated effect of each weight.

Plane clearance uses foot center minus 28mm radius. Experimental `stable_contact` uses cached
loaded-foot support heights and a smoothed mean support height for the body;
clearance is a lower bound on rough ground to allow higher viable steps.
These are contact-based approximations, not terrain ray casts. `stable_grid`
preserves the grid-based terrain reward, also used by the final `stable` default.
Contact v2 was rejected as the default after worse matched rough-terrain results.
See `RESULTS_KO.md` for measured results and remaining full-DR limitations.

## Commands

No W&B authentication is required for the local study. W&B project remains
`as2piper-low` when explicitly enabled through `WANDB_MODE=online`.

```bash
# Scratch training; iterations is the number of NEW updates.
bash run_gait.sh --preset stable --controlled --num-envs 1024 \
  --iterations 1000 --run as2piper_gait_stable_scratch_example

# Default environment without --controlled retains rough terrain and full DR.
# Existing run directories are never overwritten; --checkpoint restores Adam.
```

Results and runtime snapshots are in `experiments/gait_ablation_v1` and
`logs/as2piper-low`. Every run stores full config, asset hash and source hashes.
PPO/network/storage and AS2 hardware/URDF are unchanged. Only the new subclass
overrides the selected reward functions and contact target.
