# Three-branch recording analysis and replay

Run from the existing Windows `local_client` folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_obb_replay.ps1
```

The command exports the most recent temporal recording, uploads its completed
grasp captures over SSH port 2222, evaluates them in an independent A100 process,
downloads results, and replays the exported recording in time order.
It uses the existing `id_ed25519` key. Existing model services are not modified,
restarted, or used by this offline analysis. No model runs on Windows.
Use `-Session "C:\...\auto_sessions\recording"` to select a recording explicitly.

Each completed grasp attempt is analyzed, not every intervening tracking frame.
Original RGB-D captures are used by the existing offline camera reconstruction.
The fixed-object scene, camera trajectory, time slider, looping and speed
controls are reused. No pose interpolation or invented confidence is used.
Missing planes or failed inference are recorded as unavailable frames, never
silently replaced by another frame's plane or model result.

The default view shows five candidates per enabled branch in the selected sort
order, plus its confidence leader, global highlights and inspected candidate.
Red = diffusion; blue = camera OBB; green = world OBB. Thick outlines indicate
leaders/highlights. All candidates remain available via the rank slider.
`Only inspected candidate` isolates a single candidate with CAD and axes.
`Show all candidates` reveals the complete set. Each branch can be hidden.
`Sort` switches shallow-penetration/confidence order. Labels show confidence,
penetration and both ranks. Several highlight definitions can select one pose.

Camera OBB approach is +camera Z. World OBB approach is -estimated plane normal.
Both reuse official PCA OBB/top-down helpers and the 10-yaw, five-offset sparse
sweep. The canonical frame is converted back to the original metric camera
frame before scoring. Diffusion uses the official pretrained generator.
All candidate confidence values are evaluated together with the official
discriminator on the same filtered camera-frame cloud. Original diffusion
generation confidences are saved separately in NPZ for comparison.

Classification: penetrating if d<0; contact_like if 0<=d<=plane fitting RMSE;
safe if d>RMSE. These are final OPEN visual CAD versus plane measurements.
No arm, approach trajectory, robot TCP or grasp-success guarantee is evaluated.
World means the local support-plane frame; no gravity/IMU measurement is used.
Lowest-point region names are local-Z heuristics, not CAD link identities.

Results are stored on A100 under `outputs/obb_branch_analysis/<recording>_<UTC>/`.
They contain full/branch CSVs, top-30 per-frame reports, pose NPZs, PNG snapshots,
`manifest.json` and aggregate `REPORT.md`. File names include Frame ID.
Downloaded ZIP extraction preserves prior result folders in the recording.

To replay without recomputing:

```powershell
.\.venv\Scripts\python.exe view_obb_replay.py --results .\obb-branch-results.zip
```

If a newer recording has been created since analysis, specify the original
recording path as the positional argument. Analysis/session mismatches reject.

Verification used server-side cached actual real_world/00 RGB-D, not a user's
Windows recording: genuine diffusion + both OBB branches + official scoring,
CSV/PNG generation and Viser scene construction passed. Windows USB, PowerShell
transport and the user's latest recording require the local execution above.
