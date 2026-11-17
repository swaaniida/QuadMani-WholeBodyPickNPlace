#!/usr/bin/env python3
"""Summarize the six completed candidate demonstrations without locomotion claims."""
import csv,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'results/mesh_seated'

def main():
    result={}
    for v in ['profiles_plate','plate_only']:
        for mode in ['stand','crouch','arm']:
            rows=[{k:float(val) for k,val in r.items()} for r in csv.DictReader((OUT/(v+'_'+mode+'.csv')).open())]
            assert rows[-1]['time_s']>4.95
            settled=[r for r in rows if r['time_s']>=1]
            feet=['FL','FR','RL','RR']
            s=dict(duration_s=rows[-1]['time_s'],base_min_z_m=min(r['base_z_m'] for r in rows),
                base_final_z_m=rows[-1]['base_z_m'],com_mean_z_m=sum(r['com_z_m'] for r in rows)/len(rows),
                pitch_rms_deg=math.sqrt(sum(r['pitch_deg']**2 for r in rows)/len(rows)),
                max_abs_roll_deg=max(abs(r['roll_deg']) for r in rows),
                mean_sum_abs_joint_torque_Nm=sum(r['sum_abs_joint_torque_Nm'] for r in rows)/len(rows),
                tcp_min_z_m=min(r['tcp_z_m'] for r in rows),tpu_min_z_m=min(r['tpu_lowest_z_m'] for r in rows),
                tpu_final_z_m=rows[-1]['tpu_lowest_z_m'],
                all_four_feet_loaded_fraction_after_1s=sum(all(r[l+'_ground_normal_N']>1 for l in feet) for r in settled)/len(settled),
                foot_force_ranges_N_after_1s={l:[min(r[l+'_ground_normal_N'] for r in settled),max(r[l+'_ground_normal_N'] for r in settled)] for l in feet},
                max_self_contact_count=max(r['self_contact_count'] for r in rows),
                min_self_contact_distance_m=min(r['self_min_contact_dist_m'] for r in rows),
                mujoco_warning_count=max(r['mujoco_warning_count'] for r in rows))
            import mujoco
            m=mujoco.MjModel.from_xml_path(str(HERE/'models/mesh_seated'/(v+'.xml')))
            s['actuator_saturation_fraction']={}
            s['max_joint_limit_violation_m_or_rad']=0
            s['joint_limit_violations']={}
            for a in range(m.nu):
                name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,a)
                cap=max(abs(x) for x in m.actuator_ctrlrange[a]);key=name+'_command'
                s['actuator_saturation_fraction'][name]=sum(abs(r[key])>=cap-1e-6 for r in rows)/len(rows)
                j=int(m.actuator_trnid[a,0]);lo,hi=m.jnt_range[j]
                if m.jnt_limited[j]:
                    violation=max(max(lo-r[name+'_qpos'],r[name+'_qpos']-hi,0) for r in rows)
                    s['max_joint_limit_violation_m_or_rad']=max(s['max_joint_limit_violation_m_or_rad'],violation)
                    if violation>0:s['joint_limit_violations'][name]={'maximum':violation,'unit':'m' if int(m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_SLIDE) else 'rad'}
            result[v+'_'+mode]=s
    (OUT/'dynamics_summary.json').write_text(json.dumps(result,indent=2))
    lines=['| Variant | Mode | Pitch RMS (deg) | Four feet >1N after 1s | TPU min / final z (m) | Self contacts max |',
           '|---|---|---:|---:|---:|---:|']
    for key,s in result.items():
        v,mode=key.rsplit('_',1)
        lines.append('| %s | %s | %.3f | %.1f%% | %.3f / %.3f | %d |'%(v,mode,s['pitch_rms_deg'],100*s['all_four_feet_loaded_fraction_after_1s'],s['tpu_min_z_m'],s['tpu_final_z_m'],s['max_self_contact_count']))
    (OUT/'dynamics_table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__':main()
