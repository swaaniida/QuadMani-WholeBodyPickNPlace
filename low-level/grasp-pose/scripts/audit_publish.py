#!/usr/bin/env python3
"""Read-only publication audit of staged paths (or untracked candidates).

Does not inspect excluded environments, credentials or recordings. It never
prints matching secret material. Not a complete security/license scanner.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = {'.venv', '.venv-client', '.venv-tracking', '.venv-odometry',
           '.cache', '.deps', '.runtime', 'upstream', 'outputs', 'dist',
           'example_results', 'example_auto_session', '__pycache__'}
SUFFIXES = {'.pth', '.pt', '.ckpt', '.safetensors', '.zip', '.pem', '.key', '.pyc', '.bag'}
SECRET = re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}')


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',action='store_true',help='Check tracked and nonignored untracked paths instead of the staged index')
    p.add_argument('--manifest',type=Path,help='Write file size/hash inventory (excluded outputs path recommended)')
    a=p.parse_args()
    raw=git('ls-files','--cached','--others','--exclude-standard','-z') if a.candidates else git('diff','--cached','--relative','--name-only','--diff-filter=ACMR','-z')
    prefix=git('rev-parse','--show-prefix').decode().strip()
    paths=sorted(set(x.decode() for x in raw.split(b'\0') if x))
    errors=[];rows=[]
    for name in paths:
        f=ROOT/name
        if f.is_symlink():errors.append((name,'symlink not allowed'));continue
        parts=Path(name).parts
        if any(x in BLOCKED or x.startswith(('outputs', '.venv')) for x in parts) or f.suffix.lower() in SUFFIXES:
            errors.append((name,'excluded runtime/data/credential type'))
        if f.name.startswith(('realsense_calibration','camera_tcp_transform','id_ed25519','id_rsa')):
            errors.append((name,'private device/authentication file'))
        data=f.read_bytes() if a.candidates else git('show',':'+prefix+name)
        if len(data)>=50*1024*1024:errors.append((name,'file >=50 MiB; use release/data storage'))
        if SECRET.search(data):errors.append((name,'possible secret signature; content suppressed'))
        rows.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    result=dict(files=len(rows),bytes=sum(x['bytes'] for x in rows),errors=errors,files_manifest=rows)
    if a.manifest:
        a.manifest.parent.mkdir(parents=True,exist_ok=True)
        a.manifest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files_manifest'},indent=2))
    if not rows:raise SystemExit('No files to audit')
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
