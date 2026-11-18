"""Independent A100 offline analysis; does not connect to or modify services."""
import argparse
from datetime import datetime,timezone
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile
import numpy as np
import torch
import trimesh

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_client'))
from obb_branch_analysis import BRANCHES,generate_obb,analyze,write_csv,table
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.serving.types import SweepVolumeParams
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.utils.point_cloud import point_cloud_outlier_removal
from graspgenx.samplers.graspmoe import (_compute_obb,_world_aligned_top_down_grasp,
                                       _score_grasps_with_discriminator,OBB_YAW_COUNT,OBB_Z_OFFSETS_CM)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--result-zip',type=Path,default=ROOT/'dist/obb-branch-results.zip')
    p.add_argument('--num-grasps',type=int,default=200);p.add_argument('--max-frames',type=int)
    p.add_argument('--no-snapshots',action='store_true');a=p.parse_args()
    if not 1<=a.num_grasps<=1000:p.error('num-grasps must be 1..1000')
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    free,total=torch.cuda.mem_get_info()
    if free<6*2**30:raise RuntimeError('Less than 6 GiB free: analysis not started; existing services left running')
    with zipfile.ZipFile(a.input) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        safe=re.sub('[^a-zA-Z0-9_.-]','_',manifest['session_name'])
        out=a.output or ROOT/'outputs/obb_branch_analysis'/f'{safe}_{stamp}'
        out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir();(out/'tables').mkdir()
        ck=ROOT/'upstream/GraspGenX/ext/graspgenx_checkpoints/release'
        cfg=load_model_cfg(str(ck/'gen'),str(ck/'dis'))
        params=json.loads((ROOT/'local_client/assets/franka_umi_sweep.json').read_text())
        sampler=GraspGenXSampler.from_sweep_volume(cfg,SweepVolumeParams.coerce(params))
        mesh=trimesh.load(ROOT/'local_client/assets/franka_umi_vis_mesh.obj',force='mesh',process=False)
        original_frames=manifest['frames'];manifest['frames']=[]
        manifest.update(analysis_utc=stamp,device=torch.cuda.get_device_name(),
            weights='official release/gen/epoch_736.pth + dis/epoch_1056.pth',
            scoring='All poses rescored together with official discriminator on identical camera-frame object points. Diffusion generator confidences also retained.',
            classification='penetrating d<0; contact_like 0<=d<=plane RMSE; safe d>plane RMSE',
            camera_approach='+camera Z',world_approach='-plane normal (local support frame, no gravity sensor)',
            generation='Official PCA OBB + top-down helper, 10 yaws x offsets -8,-6,-4,-2,0 cm; gripper depth correction 177 mm',
            note='No Temporal ranking applied: all candidates retained for analysis. Source poses/recording/services unchanged.')
        all_rows=[];summaries=[]
        for entry in original_frames[:a.max_frames]:
            record=dict(entry);start=time.perf_counter()
            with np.load(BytesIO(archive.read(entry['input_file'])),allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
            try:
                plane=data['ground_plane'];rmse=float(data['plane_rmse_m'])
                if not np.isfinite(plane).all() or np.linalg.norm(plane[:3])<1e-8 or not np.isfinite(rmse):
                    raise ValueError('No verified ground plane/RMSE; world branch and collision status unavailable')
                pc=np.asarray(data['point_cloud'],np.float32)
                if len(pc)<20:raise ValueError('Fewer than 20 object points')
                filtered,_=point_cloud_outlier_removal(torch.from_numpy(pc),threshold=.014)
                filtered=filtered.cpu().numpy().astype(np.float32)
                if len(filtered)<10:filtered=pc.copy()
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();infer_start=time.perf_counter()
                diff,original_c=GraspGenXSampler.run_inference(filtered,sampler,grasp_threshold=-1,
                    num_grasps=a.num_grasps,topk_num_grasps=a.num_grasps,remove_outliers=False)
                if not len(diff):raise ValueError('Official diffusion returned no candidates')
                gd=diff.cpu().numpy().astype(np.float32);gd[:,3,3]=1.
                obb,meta=generate_obb(filtered,plane,float(params['fingertip_depth']),_compute_obb,
                    _world_aligned_top_down_grasp,OBB_YAW_COUNT,OBB_Z_OFFSETS_CM)
                poses=np.concatenate([gd,obb['obb_camera'],obb['obb_world']])
                tags=np.array(['diffusion']*len(gd)+['obb_camera']*len(obb['obb_camera'])+['obb_world']*len(obb['obb_world']))
                center=filtered.mean(axis=0).astype(np.float64)
                centered=torch.from_numpy(filtered-center.astype(np.float32)).cuda()
                confidence=_score_grasps_with_discriminator(poses,centered,center,sampler)
                torch.cuda.synchronize();infer_ms=(time.perf_counter()-infer_start)*1000
                rows,summary,highlights=analyze(poses,confidence,tags,mesh.vertices,plane,rmse)
                name=f'{entry["index"]:06d}_{re.sub("[^a-zA-Z0-9_.-]","_",entry["frame_id"])}'
                for row in rows:row.update(frame_id=entry['frame_id'],capture_utc=entry.get('capture_utc',''))
                write_csv(out/'tables'/f'{name}_all.csv',rows)
                for tag in BRANCHES:write_csv(out/'tables'/f'{name}_{tag}.csv',[r for r in rows if r['branch_tag']==tag])
                original=np.full(len(poses),np.nan,dtype=np.float32);original[:len(gd)]=original_c.cpu().numpy()
                result_file=f'frames/{name}.npz'
                np.savez_compressed(out/result_file,**data,grasps=poses,confidences=confidence,branch_tags=tags,
                    diffusion_original_confidences=original,filtered_model_points=filtered,
                    clearance_m=np.array([r['minimum_signed_distance_m'] for r in sorted(rows,key=lambda r:r['candidate_index'])]),
                    highlights_json=json.dumps(highlights),generation_json=json.dumps(meta),
                    shallow_order=np.array([r['candidate_index'] for r in rows]),confidence_order=np.argsort(-confidence,kind='stable'))
                record.update(result_file=result_file,status='ok',candidate_count=len(poses),inference_ms=infer_ms,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    processing_ms=(time.perf_counter()-start)*1000,highlights=highlights)
                for s in summary:s.update(frame_id=entry['frame_id'])
                summaries.extend(summary);all_rows.extend(rows)
                columns=['branch','count','safe','contact_like','penetrating','confidence_top1','penetration_min_mm','penetration_mean_mm','penetration_max_mm']
                report=f'# Frame {entry["frame_id"]}\n\n'+table(summary,columns)+'\n\n'+table(rows[:30],['shallow_rank','confidence_rank','branch_tag','candidate_index','confidence','penetration_mm','clearance_mm','collision_status','pose_summary'])
                (out/'tables'/f'{name}_report.md').write_text(report+'\n',encoding='utf-8')
                print(report,flush=True)
            except Exception as exc:
                record.update(status='error',error=f'{type(exc).__name__}: {exc}')
                print(entry['frame_id'],record['error'],flush=True)
                if isinstance(exc,torch.cuda.OutOfMemoryError):
                    manifest['frames'].append(record);raise
            manifest['frames'].append(record)
        write_csv(out/'all_candidates.csv',all_rows);write_csv(out/'branch_summary.csv',summaries)
        for tag in BRANCHES:write_csv(out/f'{tag}.csv',[r for r in all_rows if r['branch_tag']==tag])
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    build_report(out,manifest,summaries)
    if not a.no_snapshots:
        subprocess.run([str(ROOT/'.venv-odometry/bin/python'),str(ROOT/'scripts/snapshot_obb_replay.py'),str(out)],check=True)
    a.result_zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.result_zip,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():z.write(path,str(path.relative_to(out)))
    print(f'RESULT: {out}\nZIP: {a.result_zip}',flush=True)
    if not any(e['status']=='ok' for e in manifest['frames']):raise RuntimeError('No frames successfully analyzed; see manifest errors')


def build_report(out,manifest,summaries):
    lines=['# OBB branch replay analysis',f'Source recording: `{manifest["session_name"]}`',
        f'Analyzed captures: {sum(e["status"]=="ok" for e in manifest["frames"])} / {len(manifest["frames"])}',
        'Each frame uses the same filtered camera-frame object point cloud for all branches. All candidates retained.',
        'Uniform comparison confidence is the official discriminator re-evaluation; original diffusion confidence is separately saved.',
        'Safe/contact-like/penetrating describe final OPEN CAD versus estimated plane; no approach-path, reachability or grasp-success guarantee.',
        'Camera OBB approaches +camera Z. World OBB approaches -support-plane normal; yaw samples full 360 degrees and offsets preserve the official five values.',
        'Minimum-region labels are local-Z heuristics; the merged visual OBJ has no reliable anatomical part labels.',
        'Shallow ranking = penetration ascending, confidence descending. Least-penetration highlight breaks ties with maximum clearance; representative breaks ties with confidence. Highlights may coincide.',
        '\n## Aggregate observations (this recording only)']
    aggregate=[]
    for tag in BRANCHES:
        ss=[s for s in summaries if s['branch']==tag]
        if not ss:continue
        n=sum(s['count'] for s in ss);safe=sum(s['safe'] for s in ss)
        aggregate.append(dict(branch=tag,candidates=n,safe=safe,contact_like=sum(s['contact_like'] for s in ss),
            penetrating=sum(s['penetrating'] for s in ss),safe_fraction=safe/n,
            penetration_mean_mm=sum(s['penetration_mean_mm']*s['count'] for s in ss)/n))
        lines.append(f'- {tag}: safe {safe}/{n}; mean penetration {aggregate[-1]["penetration_mean_mm"]:.2f} mm. Highest-confidence grasp penetrates in {sum(s["confidence_top1_penetration_mm"]>0 for s in ss)}/{len(ss)} captures.')
    if aggregate:
        winner=max(aggregate,key=lambda s:s['safe_fraction'])
        lines.append(f'Highest measured safe fraction: {winner["branch"]}. This does not establish grasp success; OBB positions and incomplete visible geometry must also be inspected.')
    lines.extend(['\n## Selection implications','Use safe fraction, confidence among safe candidates, clearance distribution and opening-volume occupancy/contact evidence together. Zero penetration alone can favor a grasp floating above the object. Temporal and trajectory safety are not evaluated here.',
        '\n## Errors',*[f'- {e["frame_id"]}: {e["error"]}' for e in manifest['frames'] if e['status']!='ok']])
    (out/'REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
