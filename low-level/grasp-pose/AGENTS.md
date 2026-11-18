# Working in grasp-pose

Read `README.md`, `docs/CURRENT_STATE.md`, `docs/REPRODUCE.md` and `docs/LESSONS.md` before changes. Older reports and the original research plan describe earlier milestones; they are not the current algorithm.

- Keep all new PoC code/environments under this repository. Existing `graspgenx-realsense/` paths are public entry points; do not move them casually.
- Preserve running services and shared GPU work. Do not reset GPUs, restart servers, update system CUDA/drivers, or install into existing model environments merely to inspect/test code.
- Tests are CPU/postprocessing checks. Real model inference, hardware camera tests and service restarts are separate operations; report which were actually performed.
- SAM/AOTT and GraspGenX environments are separate. Windows has no model/CUDA dependency. Reuse official pinned sources; never invent inference/poses or claim remote USB tests.
- Preserve Frame ID association, meter units, optical axes and explicit transform direction. Official grasp base is not PiPER TCP. Rig is CAD nominal, not measured hand-eye calibration.
- Current selection: retain all model-returned candidates, plane mesh clearance > plane RMSE, then 0.6 confidence + 0.4 rotation temporal. See current-state documentation for fallback and baseline. Do not reintroduce obsolete 3mm margin, position temporal term or 1s age cutoff accidentally.
- Keep legacy entry points available; add opt-in behavior when needed. Do not change robot control or `vbc-as2piper` assets without a separate explicit request.
- Never commit credentials, environments, downloaded upstream/model weights, recordings, generated ZIPs or device-specific calibration. The root `.gitignore` is an allowlist. Run `python3 scripts/audit_publish.py` on the Git index before publishing; never bypass ignore rules with `git add -f` for runtime data.
- Respect third-party/CAD notices. User-provided CAD has no inferred blanket open-source license.
- Keep reproducibility docs and meaningful regression tests current. Avoid rewriting original meshes solely for formatting. Logs and experimental outputs go under ignored `outputs/`.
