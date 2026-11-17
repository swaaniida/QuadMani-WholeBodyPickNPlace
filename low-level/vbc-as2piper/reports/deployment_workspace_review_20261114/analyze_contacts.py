import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
z=np.load(OUT/'contact_capture/foot_physics.npz');a=z['values'];time=(np.arange(len(a))+1)*float(z['dt'])
feet=a[:,:,:12].reshape(len(a),32,4,3);forces=a[:,:,12:24].reshape(len(a),32,4,3)
clearance=feet[:,:,:,2]-.028;norm=np.linalg.norm(forces,axis=-1)
summary={}
for j,it in enumerate(z['iterations']):
    phases={}
    for name,mask in [('initial0_3s',time<=3),('all_after2s',time>=2),('ground12_60s',(time>=12)&(time<=60))]:
        rows={}
        for k,leg in enumerate(['FL','FR','RL','RR']):
            h=clearance[mask,j*8:(j+1)*8,k];f=forces[mask,j*8:(j+1)*8,k,2];n=norm[mask,j*8:(j+1)*8,k]
            floating=(h>.005)&(n<2);durations=[]
            for column in floating.T:
                run=longest=0
                for value in column:run=run+1 if value else 0;longest=max(longest,run)
                durations.append(longest*.02)
            rows[leg]={'contact_norm_above2N_fraction':float((n>2).mean()),'clearance_above5mm_fraction':float((h>.005).mean()),
                'lifted5mm_force_below2N_fraction':float(floating.mean()),'clearance_mean_mm':float(h.mean()*1000),
                'clearance_p95_mm':float(np.quantile(h,.95)*1000),'clearance_max_mm':float(h.max()*1000),
                'Fz_mean_N':float(f.mean()),'Fz_p95_N':float(np.quantile(f,.95)),
                'longest_unloaded_lift_per_replica_s':durations}
        phases[name]=rows
    summary[str(it)]=phases

normal={}
for it in [20000,28000,30000,45000]:
    p=ROOT/'reports/checkpoint_selection_20261114/evaluations'/('%d_flat'%it)/'samples.npz'
    physics=np.load(p)['physics'];f=physics[:,:8,9:13]
    normal[str(it)]={leg:{'mean_force_norm_N':float(f[:,:,k].mean()),'force_norm_above2N_fraction':float((f[:,:,k]>2).mean())} for k,leg in enumerate(['FL','FR','RL','RR'])}
b=np.load(ROOT/'reports/b1_contact_audit_20261111/flat_normal/samples.npz')['forces']
b1={leg:{'mean_force_norm_N':float(np.linalg.norm(b[:,:8,k],axis=-1).mean()),'force_norm_above2N_fraction':float((np.linalg.norm(b[:,:8,k],axis=-1)>2).mean())} for k,leg in enumerate(['FL','FR','RL','RR'])}
result={'method':'Actual pre-reset PhysX foot positions and net contact force; 64s world-fixed EE, all base velocity commands0; original exact-byte training asset;8 nominal replicas per policy. Force-reporting limitation addressed by checking positive sphere-bottom clearance as well.','feet_names':z['feet'].tolist(),'results':summary,'in_distribution_stand_nominal':normal,'b1_normal_stand_central_EE':b1}
(OUT/'contact_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for it,phases in summary.items():
    print(it,'ground', {leg:{k:round(v[k],3) for k in ['contact_norm_above2N_fraction','clearance_mean_mm','clearance_p95_mm','Fz_mean_N','lifted5mm_force_below2N_fraction']} for leg,v in phases['ground12_60s'].items()})
print('In distribution normal stand',normal)
print('B1 reference normal stand',b1)
