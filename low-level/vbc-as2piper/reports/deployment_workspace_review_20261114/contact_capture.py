"""Repeat zero-velocity world-fixed EE trajectory and record actual foot forces."""
import json
import hashlib
import shutil
from pathlib import Path
import sys
from force_ablation import ground_trajectory_compare as original
from force_ablation import run as training_run
from force_ablation.runtime import Meter
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
records = []
names = {}
old = Meter.sample


def sample(self):
    old(self)
    e = self.env
    feet = e.rigid_body_state[:,e.feet_indices,:3]-e.env_origins[:,None,:]
    forces = e.contact_forces[:,e.feet_indices]
    names['feet'] = [e.body_names[int(i)] for i in e.feet_indices]
    records.append(torch.cat([feet.flatten(1),forces.flatten(1),e.contact_filt.float(),
                              e.base_lin_vel, e.base_ang_vel, e.base_quat],-1).detach().cpu().numpy())


Meter.sample = sample
source = Path(original.__file__).read_text()
assert source.count('iterations = [10000, 20000, 30000, 45000]')==1
source = source.replace('iterations = [10000, 20000, 30000, 45000]','iterations = [20000, 28000, 30000, 45000]')
source = source.replace('errors = []','errors = []; contact_records.clear()')
ns = {'__name__':'force_ablation.contact_audit','__package__':'force_ablation',
      '__file__':original.__file__,'contact_records':records}
clone = Path(original.__file__).resolve().parents[1]
run = clone/'experiments/contact_force_ablation/runs/as2piper_headpiper_orin_200N_0.18m'
manifest = json.loads((run/'manifest.json').read_text())
asset_key = 'resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf'
root = clone.parent
backup = root/'wrist_camera_integration/backups/20261114T123254_841632Z/urdfs/low-level-as2piper-head-orin-nocontact'/asset_key
assert hashlib.sha256(backup.read_bytes()).hexdigest() == manifest['source_sha256'][asset_key]
asset_directory = OUT/'trained_asset'
(asset_directory/'urdf').mkdir(parents=True,exist_ok=True)
shadow = asset_directory/'urdf/as2piper_head_orin_training.urdf'
if not shadow.exists():shutil.copyfile(backup,shadow)
assert shadow.read_bytes()==backup.read_bytes()
if not (asset_directory/'meshes').exists():(asset_directory/'meshes').symlink_to(clone/'resources/robots/as2piper/meshes',target_is_directory=True)
original_environment = training_run.environment
original_configs = training_run.configs
def frozen_environment(*a,**kw):
    def cfg(*a,**kw):
        env,train=original_configs(*a,**kw);env.asset.file=str(shadow);return env,train
    training_run.configs=cfg
    try:return original_environment(*a,**kw)
    finally:training_run.configs=original_configs
def effective_hashes():
    values=training_run.source_hashes()
    values[asset_key]=hashlib.sha256(shadow.read_bytes()).hexdigest()
    assert values==manifest['source_sha256']
    return values
sys.argv = [original.__file__,'--run',str(run),'--output',str(OUT/'contact_capture')]
exec(compile(source,original.__file__,'exec'),ns)
ns['environment']=frozen_environment;ns['source_hashes']=effective_hashes
ns['main']()
a = np.asarray(records)
assert a.shape==(3200,32,38),a.shape
np.savez_compressed(OUT/'contact_capture/foot_physics.npz',values=a,columns='foot_xyz12,foot_force_xyz12,contact_filt4,base_lin_vel3,base_ang_vel3,base_quat4',feet=names['feet'],iterations=[20000,28000,30000,45000],dt=.02)
(OUT/'contact_capture/foot_names.json').write_text(json.dumps(names,indent=2)+'\n')
meta=json.loads((OUT/'contact_capture/metadata.json').read_text())
meta['urdf']=str(shadow)
meta['asset_note']='Exact-byte training URDF restored only into evaluation report directory from wrist-camera integration backup; production current-camera URDF preserved.'
meta['production_current_urdf_sha256']=hashlib.sha256((clone/asset_key).read_bytes()).hexdigest()
(OUT/'contact_capture/metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Foot physics saved',a.shape,names,flush=True)
