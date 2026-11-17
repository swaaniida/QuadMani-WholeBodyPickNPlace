"""No training/import of SDK: verify files, hashes, assets and run provenance."""
from pathlib import Path
import ast,hashlib,json,re,sys,xml.etree.ElementTree as ET
root=Path(__file__).resolve().parents[1];errors=[];assets=[]
for dirname in ['low-level-as2piper-head-orin-deploy','low-level-as2piper-head-orin-nocontact']:
 runtime=root/dirname
 for p in (runtime/'legged_gym').rglob('*.py'):
  try:ast.parse(p.read_text(),filename=str(p))
  except SyntaxError as e:errors.append({'syntax':str(p.relative_to(root)),'error':str(e)})
 for p in (runtime/'experiments/contact_force_ablation/runs').glob('*/manifest.json'):
  m=json.loads(p.read_text())
  for filename,wanted in m['source_sha256'].items():
   f=runtime/filename
   if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=wanted:errors.append({'source_hash':str(f.relative_to(root))})
for filename,expected_dofs in [('low-level-as2piper-head-orin-deploy/resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf',20),('low-level-as2piper-head-orin-nocontact/resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf',20),('reference/vbc-legacy/low-level/resources/robots/b1z1/urdf/b1z1.urdf',19)]:
 f=root/filename;r=ET.parse(f).getroot();n=sum(j.get('type') in ['revolute','continuous','prismatic'] for j in r.findall('joint'));mass=sum(float(m.get('value')) for m in r.findall('link/inertial/mass'));missing=[]
 for mesh in r.findall('.//mesh'):
  name=mesh.get('filename');q=Path(name) if name.startswith('/') else f.parent/name
  if not q.is_file():missing.append(name)
 if n!=expected_dofs or missing:errors.append({'urdf':filename,'dofs':n,'missing_meshes':missing})
 assets.append({'urdf':filename,'dofs':n,'mass_kg':mass,'mesh_count':len(r.findall('.//mesh')),'missing':missing})
secret=re.compile(rb'(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|(?:WANDB_API_KEY|api_key|access_token)\s*[:=]\s*[\x22\x27][A-Za-z0-9_-]{30,}[\x22\x27])')
files=[p for p in root.rglob('*') if p.is_file() and not p.is_symlink() and not any(v in ['.git','.venv-train','.venv-viz','__pycache__','.cache'] for v in p.relative_to(root).parts)]
checkpoints=0;large=[]
for p in files:
 rel=p.relative_to(root)
 if p.name in ['.netrc','auth.json','id_rsa','id_ed25519'] or p.name.startswith('.env') or p.suffix in ['.pem','.key']:
  errors.append({'forbidden_filename':str(rel)});continue
 if p.suffix in ['.pt','.pth']:
  checkpoints+=1
  with p.open('rb') as f:head=f.read(128)
  if head.startswith(b'version https://git-lfs.github.com/spec/v1'):errors.append({'lfs_not_downloaded':str(rel)})
 if p.suffix in ['.py','.sh','.md','.json','.yml','.yaml','.txt'] and p.stat().st_size<5_000_000:
  if secret.search(p.read_bytes()):errors.append({'secret_pattern':str(rel)})
 if p.stat().st_size>100*2**20:large.append(str(rel))
manifest=root/'MANIFEST_SHA256.json';verified=0
if manifest.exists():
 for name,wanted in json.loads(manifest.read_text())['files'].items():
  p=root/name
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=wanted:errors.append({'integrity':name})
  verified+=1
report={'ok':not errors,'file_count':len(files),'bytes':sum(p.stat().st_size for p in files),'checkpoint_payloads':checkpoints,'assets':assets,'verified_manifest_files':verified,'files_above_100MiB':large,'external_sdk_required':True,'errors':errors}
print(json.dumps(report,indent=2));sys.exit(0 if not errors else 1)
