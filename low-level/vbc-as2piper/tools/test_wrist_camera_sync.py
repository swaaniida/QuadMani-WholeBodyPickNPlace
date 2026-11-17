"""Regression checks against the applied CAD payload and all original backups."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import sync_wrist_camera as sync


def test_all_runtime_models_are_idempotent_and_historical_files_are_excluded():
    source=ET.parse(sync.SOURCE/'piper_umi_d435i.urdf').getroot()
    reference=json.loads((sync.SOURCE/'rig.json').read_text())
    payload=json.loads((sync.SOURCE/'integration_payload.json').read_text())
    paths,_=sync.discover()
    report=json.loads((sync.WORK/'integration_report.json').read_text())
    assert len(paths)==report['urdf_count']
    for path in paths:
        assert 'audit' not in path.parts and 'backups' not in path.parts
        assert not sync.plan(path,source,reference,payload)['changed']


def test_original_robot_bytes_and_joint_parameters_are_preserved():
    report=json.loads((sync.WORK/'integration_report.json').read_text())
    manifest_path=Path(report['backup_manifest']);manifest=json.loads(manifest_path.read_text())
    for row in manifest['urdfs']:
        before=(manifest_path.parent/'urdfs'/row['path']).read_bytes()
        after=(sync.ROOT/row['path']).read_bytes()
        assert sync.sha(before)==row['before_sha256']
        assert sync.strip_block(before.decode())==sync.strip_block(after.decode())


def test_existing_tcp_is_preserved_and_no_tcp_is_invented_for_bare_arms():
    report=json.loads((sync.WORK/'integration_report.json').read_text())
    for model in report['models']:
        if model['tcp_frame'] is None:
            assert model['T_color_tcp'] is None
        else:
            T=np.asarray(model['T_color_tcp']);inv=np.asarray(model['T_tcp_color'])
            np.testing.assert_allclose(T@inv,np.eye(4),atol=1e-10)
            assert model['tcp_frame']=='piper_umi_nominal_tcp'
        assert model['active_joint_count_before']==model['active_joint_count_after']


def test_estimated_payload_inertias_are_positive_and_labeled():
    payload=json.loads((sync.SOURCE/'integration_payload.json').read_text())
    for name in ('camera','mount'):
        I=np.asarray(payload[name]['inertia_kg_m2']);e=np.linalg.eigvalsh(I)
        assert np.min(e)>0 and e[-1]<=e[:2].sum()+1e-12
        assert payload[name]['mass_kg']>0
        assert payload[name]['mass_source']
    assert payload['camera']['mass_kg']==.072


def test_coordinate_helpers_are_massless_fixed_links():
    paths,_=sync.discover()
    for path in paths:
        robot=ET.parse(path).getroot()
        for link in robot.findall('link'):
            name=link.get('name')
            if name.startswith('wrist_d435i_') and name!='wrist_d435i_link':
                assert float(link.find('inertial/mass').get('value'))==0
                joints=[j for j in robot.findall('joint') if j.find('child').get('link')==name]
                assert len(joints)==1 and joints[0].get('type')=='fixed'


def test_comment_stripping_generators_do_not_duplicate_camera_subtree():
    paths,_=sync.discover();robot=ET.parse(paths[0]).getroot()
    text=ET.tostring(robot,encoding='unicode')  # ElementTree drops marker comments.
    restored=ET.fromstring(sync.strip_block(text))
    assert 'piper_link6' in {n.get('name') for n in restored.findall('link')}
    assert not any(n.get('name').startswith('wrist_d435i_') for n in restored.findall('link'))
