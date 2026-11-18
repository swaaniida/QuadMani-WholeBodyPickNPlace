"""Fetch official HF revisions without requiring a system git-lfs install."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'upstream/GraspGenX/ext'
MODEL_REV = '7c834043c11a11417e31d6d5ea9355801e40a2c1'
GRIPPER_REV = '19a03c00d19aeaf052d0f6801f0041982d676e8a'

if __name__ == '__main__':
    snapshot_download('adithyamurali/GraspGenXModel', revision=MODEL_REV,
                      local_dir=EXT / 'graspgenx_checkpoints', max_workers=4)
    snapshot_download('adithyamurali/gripper_descriptions', repo_type='dataset',
                      revision=GRIPPER_REV, local_dir=EXT / 'gripper_descriptions',
                      allow_patterns=['gripper_descriptions/__init__.py',
                                      'gripper_descriptions/assets/x_grippers/franka_umi/*',
                                      'README.md', 'LICENSE*'], max_workers=4)
    from graspgenx.serving.types import SweepVolumeParams
    assets = EXT / 'gripper_descriptions/gripper_descriptions/assets'
    params = SweepVolumeParams.from_gripper_config('franka_umi', str(assets))
    export = {k: v.tolist() if hasattr(v, 'tolist') else v
              for k, v in params.to_dict().items()}
    for folder in ('configs', 'local_client/assets'):
        target = ROOT / folder
        target.mkdir(parents=True, exist_ok=True)
        (target / 'franka_umi_sweep.json').write_text(json.dumps(export, indent=2))
    import shutil
    for name in ('vis_mesh.obj', 'config.json', 'README.md'):
        shutil.copy2(assets / 'x_grippers/franka_umi' / name,
                     ROOT / 'local_client/assets' / ('franka_umi_' + name))
    print('Official release and complete franka_umi assets downloaded; sweep params exported.')
