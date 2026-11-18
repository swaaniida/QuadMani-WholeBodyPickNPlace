# Repository preparation checks — 2026-11-17

## What changed

- Initialized a standalone Git repository at `grasp-pose/`, branch `main`, and staged the publication allowlist. No remote, commit or push was created.
- Preserved the original root research plan under `docs/history/ORIGINAL_RESEARCH_PLAN.md`; root README now points to the actual PoC and current implementation.
- Added a current-state guide, fresh setup instructions, file map, experimental lessons, pinned dependency/checkpoint provenance, selected past evidence, third-party notices and future-agent instructions.
- Existing entry points, model code, services, recordings, checkpoints and virtual environments were retained. No service restart, GPU model request, driver/toolkit installation or robot command was performed for this task.
- Bootstrap now shares project-local uv/Python discovery instead of requiring the historical `.runtime/python/bin/python3.10` directory. Added CPU tools installation and optional pinned RealSense source retrieval.
- Exported the previously output-directory-only Tracking profiler as source and adjusted its project/output roots. It was not rerun.
- Removed wrist test dependence on ignored historical outputs by preserving the two required reference matrices as a small test fixture. Optional official OBB geometry check explicitly skips if its pinned upstream source is absent.
- Windows packaging tolerates absent optional sample outputs and excludes runtime/device calibration files; sample outputs are not required to package a fresh source checkout.
- Environment freeze editable path normalized to relative. Freeze snapshots remain historical observations, not a complete portable lockfile.

## Executed validation

| Check | Result |
|---|---|
| Python source parsing | Passed |
| Shell syntax: bootstrap/install/fetch scripts | Passed |
| Existing `.venv` test suite | 80 passed, 2 skipped (Open3D unavailable in that env) |
| CPU suite with existing Open3D env + existing pytest packages | **82 passed**; one external Dash pytest-plugin deprecation warning |
| Source-only export, no outputs/upstream, same CPU dependencies | **81 passed, 1 skipped** (official OBB helper source not downloaded) |
| Source-only Windows ZIP generation without previous outputs | Passed; ZIP integrity checked by packager |
| Staged file audit | No excluded runtime/data/credentials paths, no detected secret signature, no individual file ≥50MiB |
| Existing tracked source or active model code changed in upstream | None; upstream excluded. Local official depth LFS hydration remains outside Git |

Test subprocesses reused installed Python packages; this is **not** a claim that downloading and rebuilding all dependencies on a clean OS was performed. No new model or camera inference was used for these checks. The two CPU environments were composed for test execution only via `PYTHONPATH`; installed environments were not modified.

Exact raw logs and staged path/size/SHA256 inventories are local under `graspgenx-realsense/outputs/repository_prep/`. The index is the source of `dist/grasp-pose-github-source.zip`; this ZIP is ignored by Git. Rebuild via `python3 scripts/package_source.py` after staging changes.

`git diff --cached --check` reports inherited Markdown hard-break whitespace and blank endings in original/generated OBJ assets. These source/CAD bytes were intentionally preserved instead of rewriting geometry for style.

## Still outside this check

- Clean-host network downloads, compiler/CUDA-extension rebuild and actual CUDA compatibility on a different GPU.
- Live Windows USB acquisition, GUI/network throughput and new recordings.
- Physical hand-eye/TCP calibration, actual hardware mass/inertia and robot collision/reachability/control.
- Public redistribution permissions for user-provided/robot-derived CAD; provenance is preserved and no new blanket license is inferred.
