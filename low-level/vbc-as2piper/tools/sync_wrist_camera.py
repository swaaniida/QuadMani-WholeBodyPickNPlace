#!/usr/bin/env python3
"""Synchronize the verified nominal D435i wrist assembly into AS2/PiPER URDFs.

The source bundle is local to this worktree. No legacy files or running services
are written. Original URDF bytes are backed up before an in-place application.
"""
import argparse
from datetime import datetime,timezone
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'wrist_camera_integration'
SOURCE=WORK/'source_bundle'
BEGIN='<!-- BEGIN SYNCED D435I WRIST ASSEMBLY -->'
END='<!-- END SYNCED D435I WRIST ASSEMBLY -->'
EXCLUDE={'third_party','.git','.deps','site-packages','experiments','audit','reports','backups','wrist_camera_integration','__pycache__'}


def within(path):
    try:Path(path).resolve().relative_to(ROOT);return True
    except ValueError:return False


def mkdir(path):
    path=Path(path)
    if not within(path):raise ValueError('Refusing directory outside worktree: '+str(path))
    missing=[];p=path
    while not p.exists():missing.append(p);p=p.parent
    path.mkdir(parents=True,exist_ok=True)
    for p in missing:os.chmod(p,0o2775)


def write(path,data):
    path=Path(path);mkdir(path.parent)
    if not within(path):raise ValueError('Refusing write outside worktree: '+str(path))
    tmp=path.with_name('.'+path.name+'.wrist-tmp')
    if isinstance(data,str):data=data.encode('utf-8')
    tmp.write_bytes(data);os.chmod(tmp,0o664);os.replace(str(tmp),str(path))


def dump(path,value):write(path,json.dumps(value,indent=2)+'\n')


def sha(data):return hashlib.sha256(data).hexdigest()


def signature(node):
    # Ignore top-level whitespace following an element; preserve its parameters.
    return ET.tostring(ET.fromstring(ET.tostring(node)))


def strip_block(text):
    if BEGIN not in text:
        # Some existing generators round-trip XML and discard comments. Recover
        # the same reserved camera subtree so re-sync never duplicates links.
        owned=r'(?:wrist_d435i_[^"\']*|piper_wrist_camera_mount(?:_joint)?)'
        pattern=r'<(link|joint)\b(?=[^>]*\bname=["\']'+owned+r'["\'])[^>]*(?:/>|>.*?</\1>)\s*'
        return re.sub(pattern,'',text,flags=re.S)
    if text.count(BEGIN)!=1 or text.count(END)!=1:raise ValueError('Ambiguous existing wrist block')
    return re.sub(re.escape(BEGIN)+'.*?'+re.escape(END)+'\n?', '',text,flags=re.S)


def transform(node):
    origin=node.find('origin');T=np.eye(4)
    if origin is None:return T
    r,p,y=map(float,origin.get('rpy','0 0 0').split())
    cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    T[:3,:3]=[[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
              [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]]
    T[:3,3]=list(map(float,origin.get('xyz','0 0 0').split()))
    return T


def fk(robot,base):
    poses={base:np.eye(4)}
    for _ in range(len(robot.findall('joint'))+1):
        for j in robot.findall('joint'):
            parent=j.find('parent').get('link');child=j.find('child').get('link')
            if parent in poses:poses[child]=poses[parent]@transform(j)
    return poses


def discover():
    found=[];excluded=[]
    for directory,dirs,files in os.walk(str(ROOT),followlinks=False):
        dirs[:]=[d for d in dirs if d not in EXCLUDE and not d.startswith('.venv')]
        for name in files:
            if not name.lower().endswith('.urdf'):continue
            path=Path(directory)/name
            if not within(path):excluded.append(dict(path=str(path),reason='outside-worktree symlink'));continue
            try:robot=ET.fromstring(strip_block(path.read_text()))
            except ET.ParseError:continue
            links={n.get('name') for n in robot.findall('link')}
            if 'piper_link6' in links:found.append(path)
    return sorted(found),excluded


def asset_root(path,robot):
    if path.parent.name=='urdf':candidate=path.parent.parent
    else:
        candidate=None
        for mesh in robot.iter('mesh'):
            raw=mesh.get('filename','')
            if raw.startswith('package://'):continue
            resolved=(path.parent/raw).resolve()
            for parent in resolved.parents:
                if parent.name=='meshes':candidate=parent.parent;break
            if candidate is not None:break
        if candidate is None:candidate=path.parent/'wrist_camera_assets'
    if not within(candidate):candidate=path.parent/'wrist_camera_assets'
    return candidate


def inertial(link,physics):
    old=link.find('inertial')
    if old is not None:link.remove(old)
    item=ET.SubElement(link,'inertial');center=physics.get('center_m',[0,0,0]);I=np.asarray(physics.get('inertia_kg_m2',np.zeros((3,3))),float)
    if I.shape==():I=np.zeros((3,3))
    ET.SubElement(item,'origin',xyz=' '.join('%.12g'%v for v in center),rpy='0 0 0')
    ET.SubElement(item,'mass',value='%.12g'%physics.get('mass_kg',0))
    ET.SubElement(item,'inertia',**{k:'%.12g'%I[i,j] for k,i,j in
        [('ixx',0,0),('iyy',1,1),('izz',2,2),('ixy',0,1),('ixz',0,2),('iyz',1,2)]})


def renamed(name):
    if name.startswith('camera_'):return 'wrist_d435i_'+name[len('camera_'):]
    if name.startswith('wrist_camera_mount'):return name.replace('wrist_camera_mount','piper_wrist_camera_mount',1)
    return name


def validate(robot):
    links=[n.get('name') for n in robot.findall('link')]
    joints=[n.get('name') for n in robot.findall('joint')]
    if len(set(links))!=len(links) or len(set(joints))!=len(joints):raise ValueError('Duplicate link/joint name')
    parents={}
    for j in robot.findall('joint'):
        p=j.find('parent').get('link');c=j.find('child').get('link')
        if p not in links or c not in links:raise ValueError('Unknown parent/child')
        if c in parents:raise ValueError('Multiple parents: '+c)
        parents[c]=p
    roots=set(links)-set(parents)
    if len(roots)!=1:raise ValueError('Expected single connected root: '+str(roots))
    for child in links:
        visited=set()
        while child in parents:
            if child in visited:raise ValueError('Joint graph cycle')
            visited.add(child);child=parents[child]


def plan(path,source,reference,payload):
    before=path.read_bytes();text=before.decode('utf-8');base=strip_block(text);robot=ET.fromstring(base)
    root_assets=asset_root(path,robot);folder=root_assets/'meshes/wrist_d435i';config=root_assets/'wrist_camera'
    rel=os.path.relpath(str(folder),str(path.parent)).replace(os.sep,'/')
    existing={n.get('name') for n in robot.findall('link')};add_flange='piper_flange_link' not in existing
    wanted={n.get('name') for n in source.findall('link') if n.get('name').startswith('camera_') or n.get('name')=='wrist_camera_mount'}
    if add_flange:wanted.add('piper_flange_link')
    nodes=[]
    for source_node in source:
        if source_node.tag=='link':include=source_node.get('name') in wanted
        elif source_node.tag=='joint':include=source_node.find('child').get('link') in wanted
        else:continue
        if not include:continue
        node=ET.fromstring(ET.tostring(source_node));old=node.get('name');node.set('name',renamed(old))
        if node.tag=='joint':
            for tag in ('parent','child'):
                node.find(tag).set('link',renamed(node.find(tag).get('link')))
        else:
            if old=='wrist_camera_mount':inertial(node,payload['mount'])
            elif old=='camera_link':inertial(node,payload['camera'])
            elif old.startswith('camera_'):inertial(node,{'mass_kg':0})
            for kind in ('visual','collision'):
                for element in node.findall(kind):
                    mesh=element.find('geometry/mesh')
                    if mesh is not None:
                        if old=='wrist_camera_mount':filename='wrist_mount.stl'
                        elif old=='camera_link':filename='d435i_visual.stl'
                        elif old=='piper_flange_link':filename='flange_'+kind+'.stl'
                        else:raise ValueError('Unexpected camera mesh: '+old)
                        mesh.set('filename',rel+'/'+filename)
                    if old=='camera_link' and kind=='collision':
                        for child in list(element):element.remove(child)
                        ET.SubElement(element,'origin',xyz=' '.join('%.12g'%v for v in payload['camera']['center_m']),rpy='0 0 0')
                        geometry=ET.SubElement(element,'geometry');ET.SubElement(geometry,'box',size=' '.join('%.12g'%v for v in payload['camera']['collision_box_size_m']))
                    if kind=='visual':
                        material=element.find('material')
                        if material is None:material=ET.SubElement(element,'material')
                        material.set('name','wrist_d435i_material_'+old)
                        if material.find('color') is None:ET.SubElement(material,'color',rgba='0.15 0.4 0.8 1' if old=='wrist_camera_mount' else '0.5 0.52 0.55 1')
        nodes.append(node)
    comments=('\n  <!-- CAD nominal optical/TCP frames; payload inertias are provisional. '+
              'Source/provenance and per-model transforms: '+os.path.relpath(str(config),str(path.parent)).replace(os.sep,'/')+' -->\n')
    block=BEGIN+comments+'\n'.join(ET.tostring(n,encoding='unicode') for n in nodes)+'\n'+END+'\n'
    end=base.rfind('</robot>')
    if end<0:raise ValueError('Missing robot closing tag')
    after=base[:end]+block+base[end:]
    if strip_block(after)!=base:raise AssertionError('Original robot bytes changed')
    candidate=ET.fromstring(after);validate(candidate)
    for old_joint,new_joint in zip(robot.findall('joint'),candidate.findall('joint')):
        if signature(old_joint)!=signature(new_joint):raise AssertionError('Existing joint changed')
    before_active=[signature(j) for j in robot.findall('joint') if j.get('type')!='fixed']
    after_active=[signature(j) for j in candidate.findall('joint') if j.get('type')!='fixed']
    if before_active!=after_active:raise AssertionError('Actuated joint order or parameters changed')
    poses=fk(candidate,'piper_flange_link');T=poses['wrist_d435i_color_optical_frame']
    if not np.allclose(T,reference['T_flange_color'],atol=2e-10):raise AssertionError('Camera/flange transform mismatch')
    tcp='piper_umi_nominal_tcp' if 'piper_umi_nominal_tcp' in existing else None
    meta=dict(urdf=str(path.relative_to(ROOT)),status='CAD_nominal_not_hand_eye_calibrated',
        original_sha256=sha(base.encode()),added_links=[n.get('name') for n in nodes if n.tag=='link'],
        added_fixed_joints=[n.get('name') for n in nodes if n.tag=='joint'],added_flange=add_flange,
        active_joint_count_before=len(before_active),active_joint_count_after=len(after_active),
        camera_frame='wrist_d435i_color_optical_frame',tcp_frame=tcp,
        T_flange_color=T.tolist(),T_color_tcp=(np.linalg.inv(T)@poses[tcp]).tolist() if tcp else None,
        T_tcp_color=(np.linalg.inv(poses[tcp])@T).tolist() if tcp else None,
        tcp_note='Existing nominal TCP retained' if tcp else 'Source robot has no UMI nominal TCP; none invented',
        payload=payload,reference_geometry={k:reference[k] for k in ('camera_commit','clocking_deg','camera_roll_deg','camera_hole_center_residual_mm','wrist_axis_coaxial_residual_mm')},
        device_intrinsics_source='Actual camera SDK JSON required; intrinsic K is not inferred from URDF')
    return dict(path=path,before=before,after=after.encode(),asset_root=root_assets,folder=folder,config=config,
                metadata=meta,changed=before!=after.encode())


def copy_assets(item):
    folder=item['folder'];mkdir(folder);config=item['config'];mkdir(config)
    for name in ('wrist_mount.stl','d435i_visual.stl','flange_visual.stl','flange_collision.stl'):
        src=SOURCE/'meshes'/name;data=src.read_bytes();target=folder/name
        if not target.exists() or sha(target.read_bytes())!=sha(data):write(target,data)
    for name in ('rig.json','integration_payload.json','REALSENSE_LICENSE','REALSENSE_SOURCE.json'):
        write(config/name,(SOURCE/name).read_bytes())
    write(config/'README.md',
        '# Nominal D435i wrist camera\n\nCamera/mount meshes are separate rigid additions; original robot meshes are unchanged. '
        'Optical axes are X right, Y down, Z forward. Per-URDF *_camera.json files preserve each existing nominal TCP. '
        'Camera mass is nominal 72 g; camera inertia is a uniform-box estimate. Mount mass/inertia assume solid PLA density 1240 kg/m3; actual printed mass is unknown. '
        'Optical helper frames have explicitly zero mass. This is not physical hand-eye/TCP calibration. '
        'No camera intrinsics are fabricated: obtain K/distortion/depth scale/extrinsics from the actual device.\n')


def main():
    os.umask(0o002)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',help='Write all applicable active robot URDFs after backup')
    parser.add_argument('--check',action='store_true',help='Fail if existing URDFs are not synchronized')
    parser.add_argument('--restore',type=Path,help='Restore exact URDF bytes from this backup manifest')
    args=parser.parse_args()
    if args.restore:
        manifest=json.loads(args.restore.read_text());base=args.restore.parent
        for entry in manifest['urdfs']:
            target=ROOT/entry['path'];backup=base/'urdfs'/entry['path']
            if sha(backup.read_bytes())!=entry['before_sha256']:raise ValueError('Corrupt backup')
            write(target,backup.read_bytes())
        print('Restored',len(manifest['urdfs']),'URDFs; added asset files retained but unreferenced.');return
    source=ET.parse(SOURCE/'piper_umi_d435i.urdf').getroot()
    reference=json.loads((SOURCE/'rig.json').read_text());payload=json.loads((SOURCE/'integration_payload.json').read_text())
    paths,excluded=discover();plans=[plan(p,source,reference,payload) for p in paths]
    changed=[p for p in plans if p['changed']]
    print('Applicable URDFs: %d; changes: %d; asset packs: %d'%(len(plans),len(changed),len({str(p['asset_root']) for p in plans})))
    for p in plans:print(('UPDATE ' if p['changed'] else 'OK     ')+str(p['path'].relative_to(ROOT))+(' [add required flange]' if p['metadata']['added_flange'] else ''))
    if args.check:
        if changed:raise SystemExit(1)
        for p in plans:
            for n in ET.fromstring(p['after']).iter('mesh'):
                raw=n.get('filename')
                if 'wrist_d435i' in raw and not (p['path'].parent/raw).exists():raise FileNotFoundError(raw)
        print('PASS: original parameters, active joints, camera transforms and new asset paths');return
    if not args.apply or not changed:return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');backup=WORK/'backups'/stamp;mkdir(backup)
    manifest=dict(created_utc=stamp,urdfs=[],excluded=excluded,updated_asset_roots=sorted({str(p['asset_root'].relative_to(ROOT)) for p in plans}),
                  note='Only PiPER URDFs in this worktree; vendor/B1Z1/experiments/audit/before and legacy files excluded')
    for p in changed:
        rel=str(p['path'].relative_to(ROOT));write(backup/'urdfs'/rel,p['before'])
        manifest['urdfs'].append(dict(path=rel,before_sha256=sha(p['before']),after_sha256=sha(p['after'])))
        diff=''.join(difflib.unified_diff(p['before'].decode().splitlines(True),p['after'].decode().splitlines(True),fromfile=rel+' (before)',tofile=rel+' (after)'))
        write(backup/'diffs'/(rel+'.diff'),diff)
    dump(backup/'manifest.json',manifest)
    applied=[]
    try:
        for p in plans:copy_assets(p)
        for p in changed:
            write(p['path'],p['after']);applied.append(p)
            dump(p['config']/(p['path'].stem+'_camera.json'),p['metadata'])
    except Exception:
        for p in applied:write(p['path'],p['before'])
        raise
    report=dict(backup_manifest=str(backup/'manifest.json'),urdf_count=len(plans),changed_count=len(changed),
                asset_packs=manifest['updated_asset_roots'],models=[p['metadata'] for p in plans],
                original_meshes_modified=False,services_restarted=False,legacy_written=False,
                dynamic_payload_note='Nominal camera + solid-PLA mount estimates; not measured hardware inertias')
    dump(WORK/'integration_report.json',report)
    print('BACKUP:',backup/'manifest.json');print('REPORT:',WORK/'integration_report.json')


if __name__=='__main__':main()
