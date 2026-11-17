"""Summarize stored evaluation only; never loads or updates a policy."""
from pathlib import Path
import json,csv
import numpy as np
ROOT=Path('/data/team/users/suhwan/vbc-as2piper')
OUT=Path(__file__).resolve().parent
EV=ROOT/'low-level-as2piper-head-orin-deploy/experiments/contact_force_ablation/evaluations'
old=ROOT/'low-level-as2piper-head-orin-nocontact/experiments/contact_force_ablation/evaluations'
summary={};rewardrows=[]
for terrain in ['flat','rough']:
    folder=EV/f'B70_iter10000_{terrain}_20261116'
    data=json.loads((folder/'result.json').read_text());samples=np.load(folder/'samples.npz');physics=samples['physics']
    baseline=json.loads((old/f'B_iter10000_{terrain}_20261112/result.json').read_text())
    groups={}
    assert np.isfinite(physics).all()
    for j,(name,m) in enumerate(data['metrics'].items()):
        a=physics[:,j*8:(j+1)*8];forces=a[:,:,9:13];contact=forces>2
        groups[name]={'metrics':m,'contact_norm_gt2_fraction':dict(zip(['FL','FR','RL','RR'],contact.mean((0,1)).tolist())),
            'all_four_contact_fraction':float(contact.all(-1).mean()),'three_or_fewer_fraction':float((contact.sum(-1)<=3).mean()),
            'roll_abs_mean_deg':m['RewardRaw/leg/roll']*180/np.pi,'per_env_vx_mean_m_s':a[:,:,3].mean(0).tolist(),
            'per_env_ee_mean_m':a[:,:,6].mean(0).tolist(),'previous200N_10k_reference':baseline['metrics'][name]}
        for k,v in m.items():
            if k.startswith('RewardWeighted/'):
                term=k.replace('RewardWeighted/','');rewardrows.append({'terrain':terrain,'condition':name,'term':term,'weighted_mean_before_div100':v,'abs_share':m['RewardAbsShare/'+term],'signed_share':m['RewardSignedShare/'+term]})
    summary[terrain]={'checkpoint_sha256':data['checkpoint_sha256'],'optimizer_steps':data['optimizer_steps'],'finite_physics':True,'groups':groups,
       'protocol':data['protocol'],'note':'Contact fractions use instantaneous net-force norm>2N, not positive Fz or sole clearance. Baseline differs in model, EE frame/workspace/control; not a causal threshold ablation.'}
OUT.joinpath('summary.json').write_text(json.dumps(summary,indent=2))
with OUT.joinpath('reward_contributions.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rewardrows[0].keys());w.writeheader();w.writerows(rewardrows)
for terrain,data in summary.items():
    print(terrain)
    for name,x in data['groups'].items():
        m=x['metrics'];print(name,'vx mean/rmse',round(m['Tracking/vx_mean_m_s'],4),round(m['Tracking/vx_rmse_m_s'],4),'yaw mean/rmse',round(m['Tracking/yaw_mean_rad_s'],4),round(m['Tracking/yaw_rmse_rad_s'],4),'EE cm',round(m['EE/position_mean_m']*100,3),'reset',m['Reset/failures'],'force95/99',round(m['Force/norm_p95_N'],1),round(m['Force/norm_p99_N'],1),'roll deg',round(x['roll_abs_mean_deg'],2),'contact',x['contact_norm_gt2_fraction'])
