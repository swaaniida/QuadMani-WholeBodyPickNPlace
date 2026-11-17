# Continuation pointer — 2026-10-30 ~10:19 UTC

User goal: under the original VBC physical environment, isolate reward effects and obtain a learnable, stable moderate AS2+PiPER gait baseline. Continue actual work; do not call standing/foot-tapping success. Working clone is `low-level-as2piper-gait`. B1 and preserved AS2 hardware/VBC code remain read-only.

## Active training (durable nohup+setsid, PPID1)

- Existing **vbc PID512627** is the user's online run. NEVER STOP/MODIFY it. B1 `vbc-legacy` also protected.
- Fresh D: **PID720756**, preset `progress_gain2_proxy_smooth_yaw4`, run `gait_reward_smooth_yaw4_fresh_s7_v3`.
- Fresh E: **PID722632**, preset `progress_gain2_proxy_smooth_yaw4_contactmatch`, run `gait_reward_smooth_yaw4_contactmatch_fresh_s7_v3`.
- Both6144 envs, seed7, checkpointNone/initial_iteration0, max600 iterations, WANDB disabled. Original rough trimesh/full DR/random EE/reset/commands/PPO/network799obs/18action/20DOF unchanged.
- Last observed D204/E164. Monitor full64env×20s, seed20261102 at model100/200/400/600; periodic model200 means201 completed iterations. D201 evaluation currently running; E201 not yet saved.
- D101 measured forward0.0107/yaw0.0330; E101 forward0.0151/yaw0.0370. Both fail, not yet meaningful gait. Do not overinterpret100-iteration pilots.

## Changes in fresh pair

D/E use smooth progress mixing: `mix=.5*min(command_norm/sigma,1)` (yaw absolute command). Actual functions preserve global max at target and are continuous at zero. Original constant mixing was discontinuous near command0.001 and gave half-max credit to overspeed at low commands. `probe_smooth_tracking.py` passed actual function goal/sign/zero/boundary tests. Raw jump~0.49/0.435 became~0.000006. Command sampling/ranges unchanged. Planar/yaw weights4/4; sigma.20/.35; grid/symmetric4cm height in both.

Only E adds **missing stance contact** to the old unwanted-swing-contact presence cost: `(1-desired)*presence + desired*(1-presence)`, mean overfeet, old weight-2. Presence is the existing VBC default foot sensor norm>1.5 bit, NOT GRF. Perfect diagonal pair cost0; one proper foot.25; allair.5; allcontact duringmoving.5; invertedpair1. Stop4contacts0. CPU function contracts passed.

`matched_fresh_v3.json`: pair differs only in preset name + symmetric_contact; identical PPO/asset/terrain/origins. Both checkpointNone/iteration0.33 non-history model0 tensors EXACT match original scratch control initialization. Model0 follows history-only firstDAGGER, so do not claim all history/sim states bit-identical.

`source_snapshots/<run>/`:52 runtime source hashes matched exactly and source bytes archived. First D module reconstructed by reverting later unused E branch, SHA matched. Entry scripts copied unchanged. Do not edit live learning settings casually.

## Evaluation / criteria

- Stop vx MAE<.08/yaw<.15; forwardvxMAE<.10 and meanbias<.10; turnvxMAE<.10/yaw<.25; combined same; lateralMAE<.10 each; >=95% environments nofall across full20s; eachcondition absRP<.35 for>=99% after2s. Strict0falls separately. Body force proxy under1% but SDK triangle-force unreliable.
- Need actual motion, normal foot lifting/planting and low slip, not just metrics/returns. `check_gait_geometry.py <eval> --stride 4` uses actual full terrain cache and moving box. stride4 covers all25 phases of2Hz/50Hz; earlier stride5 aliases phases and is archived under geometry_stride5_archive.
- Arrays sample width120 now: footXYZ49:61, grid61:65, lastsupport65:69, target69:73, counterfactualheight73/74, boxpose75:82, vy82, rootpose83:90, foot linear/angularvelocity90:114, boxvelocity114:120. `geometric_gait.json` includes exact closest-surface tangent slip, geometric contact agreement/missingstance. Geometric touch is not normal load or penetration-depth proof.
- E gives positive incentive to presence: **recheck actual geometry precision/contact planting**, don't assume old-policy precision persists or accept bit-only improvement.
- Once a candidate is viable, evaluate independent seed20261105 (e.g. vx.35/yaw-.6), full rough/DR/randomEE64×20s. Original first5000 iterations trains forwardonly, no lateralcmd; do not claim all-direction/negativevx validation before training those.
- Render actual learned states via OSMesa `render_gait.py <eval> --grid --seconds10 --start-seconds0 --stride2 --fixed-azimuth`. Real10s125frames12.5fps; recent episode reset labeled. Inspect image/temporal data. No PD/scripted gait.

## Rejected/stopped controls (all checkpoints retained)

Original full-environment stage: original600, stable600, speed200, progress201 done; gain2 stopped37; GaussianGain2 NEVER trained. No candidate passed.

v2 controls:
- stable_proxy first201 -> resume; last335 saved/stopped.301 forward.012, still tapping.
- stable_proxy_ground first101 -> resume; last305 saved/stopped.301 forward.039/yaw.122.
- progress_gain2_proxy first201 -> resume; last503 saved/stopped.201/301/401 forward.164/.193/.226, yaw.117/.225/.237, falls39/47/63.
- yaw4 branch from same201 -> last433 saved/stopped,401 evaluated: fwd.214, yaw.401, falls42. Partial benefit vs above63, still poor tracking/stability.
- ground_floor branch from same201 -> last332 saved/stopped,301 falls72 vscontrol47. Height group REJECTED; **do not launch prepared smooth_ground_floor variant** based on an intuitive argument alone.

All cooperative stops after complete optimizer/history update via own stop marker, never signal/kill user workloads. `branch_and_rejection_history.json` + `study_finished.json` authoritative counters. Controls did not finish600; do not label them600.

## Other evidence

- source `manip_loco.py` byte-identical to preserved HW. Baseline source51 + B1Python19 hash checks passed. `baseline_source_sha256.json`; B1 manifest `../low-level-as2piper/audit/source_hashes.json` reference `vbc-legacy/low-level`. Recheck beforefinal.
- Advanced-index cachedfootvelocity stale; jerk state object mismatch; stopcontact.5 corrected to1; one-sided overspeed saturation; world-z front norm and airtime mismatch. See `CAUSES_FULL_ENV_KO.md` + REWARD_CHANGES.
- Raw netcontactFz rough report~30% weight vsplane~100%; SDK bug. Corrected sensor flags failed (4.6%weight); new filter worse than existingbit; ALL production sensor/observation/substep changes reverted. Only rewards use existingbit.
- Teacher/history301: same staticDR; yawmean.210/.205 and std~.474 both. Not solely history adaptation lag.
- All47 falls in control301 were absRP>.8. No terminationcontactnames; none nearbox within.02m during prior.5s. Torque>90%0 at201.
- Repeated geometry stride4 control201→401: geometric >=2contacts95.4%→85.7%; missing front stance13.4/10.0%→30.9/22.9%; proxy precision~99.4%,recall~92%. Thus missed planting isn't only sensor reporting. Geometric touch doesn't prove support force.
- Constant progress401 nominal-plane diagnostic failed even without hardwareDR: forwardcommand.25 ->.437; pureturn ->vx.435. First plane save errored on `height_samples=None`; guard fixed in evaluate_hw/train_gait, samecheckpoint rerun succeeded. Use evaluation_retry results. No training affected.

## Reports / tools

- `python report_proxy_v2.py`: atomic RESULTS/CURRENT_STATE/comparison updates; exact log filename regex avoids D-prefix matching E; checks processcmdline againstrun.
- watcher PID697260 refreshes1minute up to4hours from~08:22 UTC (expires~12:22); logs `resume_logs/report_watch.log`. Its old liveness loop recognizes new monitor tags ending_resume1 (these tags are workers, actual fresh training is iteration0). May need restart only read-only reporter if long study exceedsbound.
- `monitor_full_env.py` serializes evaluations with v1/evaluation.lock + child gpu_evaluation.lock. No duplicate GPU evals. Fixedstatic DR hashes do NOT mean trajectory-dependent resets/EE/push RNG streams identical.
- Training logs in `resume_logs/progress_gain2_proxy_smooth_yaw4_20261030_093803_720749.log` and `...smooth_yaw4_contactmatch_20261030_094442_722626.log`. PID files available. Check before starting anything; no duplicate launches.
- `launch_study_detached.sh` / `launch_monitor_detached.sh`: robust detachedwork. All currentprocesses independent of toolPTY. Earlier foreground3trials died together07:12 withouttraceback or currentOOMrecord; exactcauseUNKNOWN. Own resumedtrial optimizer/history restored but physical state not; caveatdocumented.

## Required completion

1. Get fresh201/401 physical results. Do not settle for standing/no-fall success. If still no learner, investigate justified new reward change rather than pretend ready. A possible NOT IMPLEMENTED fallback is bounded quadratic progress `(||cmd||²-||v-cmd||²)/(||cmd||²+sigma²)` clipped lower-1: dense gradient, targetmaximum, no lowcommand ratio/overspeed plateau; negative-reward/termination tradeoff needs checking. Don't blindly launch more variants before current201 data.
2. Once learned moderate gait + holdout/geometry/video validated, update public `AS2PiPERStableGaitCfg` ALL study fields/scales to selected preset (currently old Gaussian/grid UNVALIDATED on fullDR). Check exact config against configure(selected). `run_stable.sh` project as2piper-low, unique scratchrunid reflecting fixes; no resume520. Prepare45000 command, don't auto-start production unless intent authorizes.
3. Cooperative stop own studies after complete checkpoints if validation sufficient. Keep all artifacts. Existing512627/B1 untouched.
4. Formal concise Korean results + selected video/report links, measured benefits, remaining fullDR/realhardware limits; regenerate cumulative sourcepatch + changed_files; recheckprotected51/19hashes andPID; permissions directories2775/files664/scripts775 for ownednewworktree artifacts only. Do not touch private codexhome/auth.

No proactive sub-agents allowed. No goal API unless explicitly requested. Korean commentary updates<=60s; no permission/confirmation loops. Work continues until baseline goal handled.


## Update ~10:52 UTC: fresh F added

- F training PID741451, monitor741500, preset `progress_gain2_proxy_smooth_yaw4_footground`, run `gait_reward_smooth_yaw4_footground_fresh_s7_v4`, fresh0/seed7/6144/max600/WANDBoff. No more than3 owntraining workers (D/E/F) + protected original512627. Verify alive afterboot, don't duplicate.
- F differs fromD ONLY foot_height_reference='proxy'. Body terrain_height_mode remainsgrid; base-height objective unchanged (CPU exactcontract). Symmetric4cm Gaussian preserved (no floor, no contactmatch). Separate foot helper optionalmode, cacheupdatedfromexistingcontactbit; no newobservations/physics/reset or geometry checks online.
- Reason: actual D/E201 geometry frontlift~7–10cm vsrear~1–2cm, phaseagreement~.65, both>=2nearcontacts~99.5%; missing-stance reward not solvingrearfootdrag. Conservativegrid belowactualcontact surface makes4cm footgoalmisleading. Earlier rejectedC changed bodyreference +swingfloor too, so didn't isolate footmeasurement. F cleanly isolatesfootreference.
- F prelaunch52 sourcefiles+entry archived under source_snapshots/run. After runtime_snapshot/model0 appear verify PRELAUNCHSHA againstruntime, cfg differences onlystudy.preset/foot_height_reference vsD, samePPO/asset/terrain/init33nonhisttensors. Do not editcore module until snapshot hasbeenverified.
- D201: forward.0584,yawturn.195,falls13. E201: forward.0822,yawturn.191,falls13. Bothfail, but growingfrom101; notfrozen. No extra trackingkernel trial launched; leave fallbackquad unimplemented unless requiredafter400/600.
- Both201 staticevalDR/terrainhashmatch, EEGmean2.4–2.7cm/angular~5deg. D/E201 geometry ready and includesactualphase/slip. Eprecision>=.998, no clear sensorfalsepositive exploitation yet. Contactmatch extra stillhasnot improvedstabilityat201; preferminimal unlesslaterdatajustifies.
- Last seenD385/E345 beforeFlaunch. D/E400 checkpoint due soon, freshF100in~30min. Evaluate sameprimary criteria, actualallfootgait &holdout. Publicdefault stilloldGaussiangrid, not selected; no45000productionstarted.
- If F goespast12:22 UTC, original boundedreportwatcher expires; restart only own read-only reporter after it exits, nevertouchtraining. Reports can be refreshedmanually anytime.
