"""Rebase only historical text files. Core training code/source hashes stay intact."""
from pathlib import Path
import argparse,json
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
mapping=[('/data/team/users/suhwan/vbc-as2piper',str(root)),('/data/team/users/suhwan/vbc-legacy',str(root/'reference/vbc-legacy')),('/data/team/users/suhwan/robot-sources',str(root/'robot-sources')),('/data/team/users/suhwan/deep-wbc-as2piper/as2piper/legged_gym/resources/robots/as2piper/urdf/as2piper_center_umi_orin64.urdf',str(root/'reference/orin_source/as2piper_center_umi_orin64.urdf'))]
changed=[]
for folder in ['reports','tools','mujoco_mount_compare','low-level-as2piper/audit','low-level-as2piper-forward/audit_forward','low-level-as2piper-hw/audit_hw','low-level-as2piper-gait/experiments']:
 for f in (root/folder).rglob('*'):
  if not f.is_file() or f.suffix not in ['.py','.sh','.xml']:continue
  text=f.read_text();new=text
  for old,replacement in mapping:new=new.replace(old,replacement)
  # Archived scripts launch a former shared visualization venv.
  new=new.replace(str(root/'unitree_rl_mjlab/.venv/bin/python'),str(root/'.venv-viz/bin/python'))
  if new!=text:
   changed.append(str(f.relative_to(root)))
   if a.apply:
    backup=f.with_suffix(f.suffix+'.server-original')
    if not backup.exists():backup.write_text(text)
    f.write_text(new)
print(json.dumps({'apply':a.apply,'files':changed},indent=2))
print('Rebasing intentionally changes exported archive hashes; regenerate manifest only after review.')
