#!/usr/bin/env python3
"""Read-only structural, protected-source and MuJoCo import validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from sync_wrist_camera import ROOT,WORK,strip_block,validate,signature,fk


def normalized_urdf(path,data):
    """MuJoCo cannot read DAE: use existing STL counterpart, like project converter.

    Only an in-memory validation copy is changed; original robot files are not.
    """
    root=ET.fromstring(data)
    for node in root.iter('mesh'):
        raw=node.get('filename')
        if raw.startswith('package://'):raise ValueError('Unresolved package URI '+raw)
        p=(path.parent/raw).resolve()
        if p.suffix.lower()=='.dae':p=p.parent.parent/(p.stem+'.stl')
        if not p.is_file():raise FileNotFoundError(str(p))
        node.set('filename',str(p))
    mujoco=root.find('mujoco')
    if mujoco is None:mujoco=ET.SubElement(root,'mujoco')
    compiler=mujoco.find('compiler')
    if compiler is None:compiler=ET.SubElement(mujoco,'compiler')
    compiler.set('strippath','false');compiler.set('fusestatic','true');compiler.set('discardvisual','false')
    return ET.tostring(root,encoding='unicode')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mujoco',action='store_true');a=p.parse_args()
    report=json.loads((WORK/'integration_report.json').read_text())
    manifest_path=Path(report['backup_manifest']);manifest=json.loads(manifest_path.read_text())
    rows=[]
    if a.mujoco:import mujoco
    for entry in manifest['urdfs']:
        path=ROOT/entry['path'];before=(manifest_path.parent/'urdfs'/entry['path']).read_bytes();after=path.read_bytes()
        assert hashlib.sha256(before).hexdigest()==entry['before_sha256']
        assert strip_block(after.decode()).encode()==strip_block(before.decode()).encode()
        original=ET.fromstring(before);current=ET.fromstring(after);validate(current)
        orig_j=[signature(j) for j in original.findall('joint') if j.get('type')!='fixed']
        new_j=[signature(j) for j in current.findall('joint') if j.get('type')!='fixed']
        assert orig_j==new_j
        for node in current.iter('mesh'):
            raw=node.get('filename')
            assert not raw.startswith('package://')
            assert (path.parent/raw).exists(),raw
        row=dict(path=entry['path'],structure='pass',original_bytes_preserved=True,active_dof=len(new_j))
        if a.mujoco:
            old=None;new=None;old_error=None;new_error=None
            try:old=mujoco.MjModel.from_xml_string(normalized_urdf(path,before))
            except Exception as exc:old_error=str(exc)
            try:new=mujoco.MjModel.from_xml_string(normalized_urdf(path,after))
            except Exception as exc:new_error=str(exc)
            row.update(mujoco_before_error=old_error,mujoco_after_error=new_error)
            if old is not None and new is not None:
                names=lambda m:[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
                assert names(old)==names(new),'Runtime joint order changed'
                assert old.nq==new.nq and old.nv==new.nv
                row.update(mujoco='pass',nq=new.nq,nv=new.nv,original_total_mass_kg=float(old.body_mass.sum()),
                           total_mass_kg=float(new.body_mass.sum()),added_mass_kg=float(new.body_mass.sum()-old.body_mass.sum()))
            elif old is None and new is not None:row.update(mujoco='pass_after_preexisting_before_error',nq=new.nq,nv=new.nv)
            elif old_error==new_error:row['mujoco']='preexisting_import_error'
            else:row['mujoco']='new_import_error'
        rows.append(row)
        print(entry['path'],row.get('mujoco','structure pass'),flush=True)
    protected=json.loads((WORK/'protected_before.json').read_text());modified=[]
    for filename,digest in protected.items():
        path=Path(filename)
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:modified.append(filename)
    if modified:raise AssertionError('Protected originals changed: '+str(modified))
    result=dict(models=len(rows),structural_pass=len(rows),protected_files_checked=len(protected),protected_files_modified=modified,
                mujoco_requested=a.mujoco,models_report=rows)
    if a.mujoco:
        result['mujoco_version']=mujoco.__version__
        result['mujoco_pass']=sum(r.get('mujoco','').startswith('pass') for r in rows)
        result['preexisting_import_errors']=sum(r.get('mujoco')=='preexisting_import_error' for r in rows)
        result['new_import_errors']=sum(r.get('mujoco')=='new_import_error' for r in rows)
    output=WORK/('verification_mujoco.json' if a.mujoco else 'verification_structure.json')
    output.write_text(json.dumps(result,indent=2));os.chmod(output,0o664)
    print(json.dumps({k:v for k,v in result.items() if k!='models_report'},indent=2),flush=True)
    if result.get('new_import_errors',0):raise SystemExit(1)


if __name__=='__main__':main()
