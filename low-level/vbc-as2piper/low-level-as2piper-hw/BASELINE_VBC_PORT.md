# Preserved AS2 VBC-port baseline

- Task: `as2piper_plate_vbc_port`
- Directory: `low-level-as2piper-hw`
- Entry: `run_hw.sh` (default task remains this baseline)
- This is the AS2 hardware port with VBC reward restoration, including the documented contact-shaping sign correction. It is not byte-identical to upstream B1 VBC.
- Known inherited defects (stale foot velocity / inactive feet_jerk) are preserved here for comparison. No gait-study fixes are applied to this baseline.
- New gait experiments and fixes are isolated in `../low-level-as2piper-gait`.
- Baseline policy: `logs/as2piper-low/vbc_port_walk_verify_20261029_155856/model_300.pt`; continuation313 is in `logs/as2piper-low/vbc_port_walk_verify_extended_20261029_164834/model_313.pt`.
- Source SHA256 at experiment fork: `../low-level-as2piper-gait/baseline_source_sha256.json`.
- Existing vbc-owned45000 run uses task `as2piper_plate_walk_wbc_vbc`; it is a different older task and is left untouched.
