"""Hydrate official recorded RGB-D sample, validating Git LFS SHA-256."""
import hashlib
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]/'upstream/GraspGenX'
REV='44eb1d77d6d27bc97ee361e3afb717a2115e30a9'

if __name__=='__main__':
    for name in ('depth.npy','rgb.png','seg.png'):
        p=ROOT/'assets/sample_data/real_world/00'/name
        pointer=p.read_bytes()
        if not pointer.startswith(b'version https://git-lfs.github.com/spec/v1'):
            continue
        expected=pointer.decode().split('oid sha256:')[1].splitlines()[0]
        r=requests.get(f'https://media.githubusercontent.com/media/NVlabs/GraspGenX/{REV}/{p.relative_to(ROOT).as_posix()}',timeout=60)
        r.raise_for_status()
        assert hashlib.sha256(r.content).hexdigest()==expected
        p.write_bytes(r.content)
        print(name,len(r.content),expected)
