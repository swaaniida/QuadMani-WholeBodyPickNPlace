"""Package CPU-only client, provenance and real recorded-sample inference."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    client=ROOT/'local_client'
    examples=client/'example_results'
    examples.mkdir(exist_ok=True)
    for source,target in [
        ('outputs/grasps/official_rgbd_replay.npz','official_rgbd_replay.npz'),
        ('outputs/grasps/official_rgbd_replay.json','official_rgbd_replay.json'),
        ('outputs/rgb/official_rgbd_replay_overlay.png','official_rgbd_replay_overlay.png')]:
        if (ROOT/source).is_file():shutil.copy2(ROOT/source,examples/target)
    for source,target in [
        ('outputs/tracking/grasps/integration_static_rgbd.npz','tracking_static_rgbd.npz'),
        ('outputs/tracking/grasps/integration_static_rgbd.json','tracking_static_rgbd.json'),
        ('outputs/tracking/rgb/integration_static_rgbd_overlay.png','tracking_static_rgbd_overlay.png'),
        ('outputs/tracking/official_demo/last_overlay.png','tracking_DAVIS_last_overlay.png')]:
        if (ROOT/source).exists():shutil.copy2(ROOT/source,examples/target)
    for name in ('official_rgbd_ranked.npz', 'official_rgbd_ranked.json'):
        source=ROOT/'outputs/ranking/grasps'/name
        if source.exists():shutil.copy2(source,examples/name)
    ranked_overlay=ROOT/'outputs/ranking/rgb/official_rgbd_ranked_overlay.png'
    if ranked_overlay.exists():shutil.copy2(ranked_overlay,examples/ranked_overlay.name)
    (examples/'README.txt').write_text(
        'Optional examples from actual A100 inference; only present when generated outputs were available at packaging.\n'
        'This is dataset replay, not Windows USB camera validation.\n'
        'python view_saved.py example_results/official_rgbd_replay.npz\n'
        'Ranked comparison: python view_saved.py example_results/official_rgbd_ranked.npz\n'
        'Tracking example: actual SAM+AOTT mask and GraspGenX on a repeated static official RGB-D frame.\n'
        'python view_saved.py example_results/tracking_static_rgbd.npz\n'
        'tracking_DAVIS_last_overlay.png: actual temporal AOTT tracking on the official moving bear sequence.\n',encoding='utf-8')
    benchmark=ROOT/'outputs/auto_grasp/latest_benchmark.json'
    if benchmark.exists():
        report=json.loads(benchmark.read_text())
        session=Path(report['session_path'])
        with (session/'poses.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        if rows and 'frame_file' in rows[0]:
            excerpt=[rows[0]] if len(rows)==1 else [rows[0],rows[-1]]
            target=client/'example_auto_session'
            if target.exists():shutil.rmtree(target)  # Generated package example only.
            target.mkdir()
            for row in excerpt:
                for key in ('file','frame_file'):
                    destination=target/row[key]
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(session/row[key],destination)
            with (target/'poses.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(excerpt[0]));writer.writeheader();writer.writerows(excerpt)
            (target/'summary.json').write_text(json.dumps(dict(excerpt=True,poses=len(excerpt),
                full_session_poses=report['poses'],full_session_completed_pose_fps=report['completed_pose_fps'],
                windows_hardware_tested=False,source='Actual models on repeated static official RGB-D, first/last recorded frames'),indent=2))
            (target/'README.txt').write_text('Two-frame excerpt of an actual continuous SAM/AOT + GraspGenX run.\n'
                'Static recorded RGB-D replay; NOT Windows camera validation.\n'
                'python view_session.py example_auto_session\n'
                'Per-row cumulative FPS belongs to the full benchmark run, not this two-frame excerpt.\n')
    out=ROOT/'dist'
    out.mkdir(exist_ok=True)
    archive=out/'graspgenx-realsense-windows.zip'
    manifest={}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(client.rglob('*')):
            if (not p.is_file() or '__pycache__' in p.parts or any(part.startswith(('.venv', 'outputs')) for part in p.relative_to(client).parts)
                or p.name.startswith(('realsense_calibration', 'camera_tcp_transform')) or 'device_calibrated' in p.name):
                continue
            name=p.relative_to(ROOT).as_posix()
            z.write(p,name)
            manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (out/'SHA256SUMS').write_text(f'{digest}  {archive.name}\n')
    print(archive,archive.stat().st_size,'bytes',digest)
