"""Freeze SHA256 inventory. Existing manifest overwrite requires an explicit flag."""
from pathlib import Path
import argparse,hashlib,json,datetime
root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--accept-changes',action='store_true');a=p.parse_args();target=root/'MANIFEST_SHA256.json'
if target.exists() and not a.accept_changes:p.error('Manifest already exists; review changes before --accept-changes')
exclude={'.git','__pycache__','.cache','.venv-train','.venv-viz','.deps','.wandb_deps','.torch_extensions','wandb','local_logs','.ssh','.aws','.config'}
files={};sizes={}
for f in sorted(root.rglob('*')):
 if not f.is_file() or f.is_symlink():continue
 rel=f.relative_to(root)
 if f==target or any(v in exclude for v in rel.parts):continue
 if rel.parts[:2]==('third_party','isaacgym') and f.name!='README_REQUIRED.md':continue
 if f.name in ['auth.json','.netrc','id_rsa','id_ed25519'] or f.name.startswith('.env') or f.suffix in ['.pem','.key']:continue
 files[str(rel)]=hashlib.sha256(f.read_bytes()).hexdigest();sizes[str(rel)]=f.stat().st_size
unique={}
for name,h in files.items():unique[h]=sizes[name]
value={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'algorithm':'SHA256','file_count':len(files),'bytes':sum(sizes.values()),'unique_content_bytes':sum(unique.values()),'files':files}
target.write_text(json.dumps(value,indent=2)+'\n');target.chmod(0o664)
print(json.dumps({k:v for k,v in value.items() if k!='files'},indent=2))
