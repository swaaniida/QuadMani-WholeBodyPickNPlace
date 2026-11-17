"""Keep instantaneous metrics and separately measure original2Hz-cycle motion."""
import argparse,json
from pathlib import Path
import numpy as np
NAMES=['stand','forward','turn','forward_turn']

def analyze(folder,summary=None):
    folder=Path(folder);meta=json.loads((folder/'policy_evaluation.json').read_text())
    x=np.load(folder/'policy_evaluation.npz')['samples'];group=meta['envs_per_command']
    if x.shape[-1]<90:return None
    # Existing fixed gait2Hz / control50Hz.26 endpoint poses span0.5sec.
    length=25;rows={}
    for i,name in enumerate(NAMES):
        velocity=[];ripple_xy=[];ripple_yaw=[];valid=0;total=0
        for env in range(i*group,(i+1)*group):
            for start in range(100,len(x)-length,length):
                b=x[start:start+length+1,env];total+=1
                if (b[:,2]>.5).any() or (np.abs(b[:,40:42])>=.35).any():continue
                valid+=1;velocity.append(b[:-1,[0,1,82]].mean(0))
                root=b[:,83:90];t=np.linspace(0,1,len(root))
                straight=(1-t[:,None])*root[0,:2]+t[:,None]*root[-1,:2]
                ripple_xy.append(np.linalg.norm(root[:,:2]-straight,axis=-1).max())
                qx,qy,qz,qw=root[:,3:7].T
                yaw=np.unwrap(np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz)))
                linear=(1-t)*yaw[0]+t*yaw[-1]
                ripple_yaw.append(np.rad2deg(np.abs(yaw-linear).max()))
        if not velocity:return None
        v=np.array(velocity);cx=meta[name]['command_vx_m_s'];cw=meta[name]['command_yaw_rad_s']
        rows[name]={'vx_mean':float(v[:,0].mean()),'yaw_mean':float(v[:,1].mean()),'cycle_vx_MAE_m_s':float(abs(v[:,0]-cx).mean()),'cycle_yaw_MAE_rad_s':float(abs(v[:,1]-cw).mean()),'cycle_vy_MAE_m_s':float(abs(v[:,2]).mean()),'valid_cycle_fraction':valid/total,'cycles':valid,'xy_departure_from_endpoint_chord_p95_m':float(np.quantile(ripple_xy,.95)),'yaw_departure_from_endpoint_linear_p95_deg':float(np.quantile(ripple_yaw,.95))}
    result={'completed_iterations':meta['completed_updates'],'definition':'Supplementary0.5s cycle-average metrics at original2Hz; keep original instantaneous strict gate unchanged. Exclude cycles with resets or absRP>=.35; require >=85% valid cycles. Pose residual relative to endpoint chord/linear yaw includes commanded curvature, not a pure disturbance estimate.','groups':rows}
    if summary is None:
        file=folder/'gait_summary.json'
        if file.exists():summary=json.loads(file.read_text())
    if summary is not None:
        a,b,c,d=[rows[k] for k in NAMES]
        criteria={'stop':a['cycle_vx_MAE_m_s']<.08 and a['cycle_yaw_MAE_rad_s']<.15,'forward':b['cycle_vx_MAE_m_s']<.10 and abs(b['vx_mean']-meta['forward']['command_vx_m_s'])<.10,'turn':c['cycle_vx_MAE_m_s']<.10 and c['cycle_yaw_MAE_rad_s']<.25,'combined':d['cycle_vx_MAE_m_s']<.10 and d['cycle_yaw_MAE_rad_s']<.25,'lateral':all(v['cycle_vy_MAE_m_s']<.10 for v in rows.values()),'enough_valid_cycles':all(v['valid_cycle_fraction']>=.85 for v in rows.values()),'xy_cycle_ripple_p95_under3cm':all(v['xy_departure_from_endpoint_chord_p95_m']<.03 for v in rows.values()),'yaw_cycle_ripple_p95_under5deg':all(v['yaw_departure_from_endpoint_linear_p95_deg']<5 for v in rows.values()),'no_fall_environments_at_least95pct':np.mean([summary[k]['no_fall_environment_fraction'] for k in NAMES])>=.95,'orientation_stable_at_least99pct':all(summary[k]['orientation_stable_fraction']>=.99 for k in NAMES)}
        result['cycle_baseline_criteria']={k:bool(v) for k,v in criteria.items()};result['cycle_baseline_pass']=all(criteria.values())
    (folder/'cycle_tracking.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evaluation');a=p.parse_args();print(json.dumps(analyze(a.evaluation),indent=2))
