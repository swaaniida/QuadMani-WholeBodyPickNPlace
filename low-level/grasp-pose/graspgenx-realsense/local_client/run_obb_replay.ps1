param([string]$Session = "")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$key = Join-Path $env:USERPROFILE '.ssh\id_ed25519'
$remoteRoot = '/data/team/users/suhwan/grasp-pose/graspgenx-realsense'

if ($Session) {
    & $py prepare_obb_replay.py $Session
} else {
    & $py prepare_obb_replay.py
}
if ($LASTEXITCODE -ne 0) { throw 'Recording export failed' }
& scp -P 2222 -i $key .\obb-replay-input.zip "suhwan@astrago:${remoteRoot}/outputs/analysis/obb-replay-input.zip"
if ($LASTEXITCODE -ne 0) { throw 'Input upload failed' }
& ssh -p 2222 -i $key suhwan@astrago "bash ${remoteRoot}/scripts/run_obb_analysis.sh ${remoteRoot}/outputs/analysis/obb-replay-input.zip"
if ($LASTEXITCODE -ne 0) { throw 'A100 analysis failed; see output above' }
& scp -P 2222 -i $key "suhwan@astrago:${remoteRoot}/dist/obb-branch-results.zip" .\obb-branch-results.zip
if ($LASTEXITCODE -ne 0) { throw 'Result download failed' }
# Bind playback to the exported recording, even if a newer recording appears.
$recordedSession = & $py -c "import json,zipfile; print(json.loads(zipfile.ZipFile('obb-replay-input.zip').read('manifest.json'))['source_session'])"
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve exported recording' }
& $py view_obb_replay.py $recordedSession --results .\obb-branch-results.zip
if ($LASTEXITCODE -ne 0) { throw 'Replay failed' }
