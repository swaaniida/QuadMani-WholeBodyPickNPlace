"""Summarize motion, contact cadence and explicit local acceptance criteria."""
import argparse,json
from pathlib import Path
import numpy as np


def summarize(folder):
    folder=Path(folder);d=json.loads((folder/'policy_evaluation.json').read_text())
    x=np.load(folder/'policy_evaluation.npz')['samples'];trace=np.load(folder/'trajectories.npz')
    group=d.get('envs_per_command',8)
    for i,name in enumerate(['stand','forward','turn','forward_turn']):
        s=x[100:,i*group:(i+1)*group];c=s[:,:,5:9]>5
        f=np.fft.rfftfreq(len(s),.02);power=abs(np.fft.rfft(c-c.mean(0),axis=0))**2
        psd=power.mean((1,2));band=(f>.5)&(f<10)
        d[name]['contact_dominant_Hz']=float(f[band][psd[band].argmax()])
        d[name]['fall_resets_total']=int(((x[:,i*group:(i+1)*group,2]>.5)&(x[:,i*group:(i+1)*group,3]<.5)).sum())
        full=x[:,i*group:(i+1)*group]
        failures=(full[:,:,2]>.5)&(full[:,:,3]<.5)
        d[name]['no_fall_environment_fraction']=float((~failures.any(0)).mean())
        d[name]['fall_resets_first_2s']=int(failures[:100].sum())
        d[name]['fall_resets_after_2s']=int(failures[100:].sum())
        if full.shape[-1]>=44:
            d[name]['orientation_stable_fraction']=float((np.abs(full[100:,:,40:42]).max(-1)<.35).mean())
            d[name]['roll_pitch_termination_events']=int((failures&(np.abs(full[:,:,40:42]).max(-1)>.8)).sum())
            d[name]['world_z_termination_events']=int((failures&(full[:,:,42]<.1)).sum())
        if full.shape[-1]>=45:
            d[name]['body_contact_gt5N_fraction']=float((full[100:,:,44]>5).mean())
        if full.shape[-1]>=49:
            proxy=full[100:,:,45:49]>.5
            d[name]['phase_contact_agreement_proxy']=float((proxy==(full[100:,:,17:21]>.5)).mean())
            d[name]['proxy_contact_fraction']=proxy.mean((0,1)).tolist()
            d[name]['stance_slip_proxy_m_s']=float(full[100:,:,35:39][proxy].mean()) if proxy.any() else 0.
        root=trace['root'][50:,i];qx,qy,qz,qw=root[:,3:7].T
        roll=np.arctan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy))
        pitch=np.arcsin(np.clip(2*(qw*qy-qz*qx),-1,1))
        d[name]['representative_roll_rms_deg']=float(np.rad2deg(np.sqrt(np.mean(roll**2))))
        d[name]['representative_pitch_rms_deg']=float(np.rad2deg(np.sqrt(np.mean(pitch**2))))
    a,b,c,e=[d[k] for k in ['stand','forward','turn','forward_turn']]
    criteria=dict(stop=a['vx_abs_error_m_s']<.08 and a['yaw_abs_error_rad_s']<.15,
                  forward=b['vx_abs_error_m_s']<.10 and abs(b['mean_vx_m_s']-b['command_vx_m_s'])<.10,
                  turn=c['vx_abs_error_m_s']<.10 and c['yaw_abs_error_rad_s']<.25,
                  combined=e['vx_abs_error_m_s']<.10 and e['yaw_abs_error_rad_s']<.25,
                  no_fall=sum(t['fall_resets_total'] for t in [a,b,c,e])==0,
                  upright=all(t['upright_fraction']>=.99 for t in [a,b,c,e]))
    d['local_acceptance_criteria']=criteria
    d['local_acceptance_pass']=all(criteria.values())
    if x.shape[-1]>=45 and 'Terrain=trimesh' in d['conditions']:
        groups=[a,b,c,e]
        baseline_criteria={key:criteria[key] for key in ['stop','forward','turn','combined']}
        baseline_criteria.update(
            no_fall_environment_fraction_at_least_95pct=np.mean([g['no_fall_environment_fraction'] for g in groups])>=.95,
            orientation_stable_at_least_99pct=all(g['orientation_stable_fraction']>=.99 for g in groups),
            body_contact_under_1pct=all(g['body_contact_gt5N_fraction']<.01 for g in groups))
        if all('vy_abs_error_m_s' in g for g in groups):
            baseline_criteria['lateral_velocity_mae_under_0_10_m_s']=all(g['vy_abs_error_m_s']<.10 for g in groups)
        else:
            baseline_criteria['lateral_velocity_verified']=False
        d['full_environment_baseline_criteria']={k:bool(v) for k,v in baseline_criteria.items()}
        d['full_environment_baseline_pass']=all(baseline_criteria.values())
        d['strict_no_fall_pass']=criteria['no_fall']
    d['selection_error_score']=sum(t['vx_abs_error_m_s']+.3*t['yaw_abs_error_rad_s']+3*t['fall_resets_total'] for t in [a,b,c,e])
    (folder/'gait_summary.json').write_text(json.dumps(d,indent=2))
    return d


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evaluation');a=p.parse_args()
    d=summarize(a.evaluation)
    print(json.dumps({k:v for k,v in d.items() if k in ('local_acceptance_criteria','local_acceptance_pass','selection_error_score')},indent=2))
