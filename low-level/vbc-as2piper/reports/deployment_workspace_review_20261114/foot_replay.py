"""Measure foot sphere clearance in existing recorded zero-velocity EE replay."""
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT/'reports/ee_ground_compare_20261114/data_final'
m = mujoco.MjModel.from_xml_path(str(SOURCE/'render_0.xml'))
d = mujoco.MjData(m)
meta = json.loads((SOURCE/'metadata.json').read_text())
z = np.load(SOURCE/'states.npz')
qids = [int(m.joint(n).qposadr[0]) for n in meta['dof_names']]
feet = [m.body(leg+'_foot').id for leg in ['FL','FR','RL','RR']]
clearance = np.zeros((len(z['time']),4,4))
com = np.zeros((len(z['time']),4,3))
foot_xy = np.zeros((len(z['time']),4,4,2))
for i in range(len(z['time'])):
    for j in range(4):
        root = z['root'][i,j]
        d.qpos[:3] = root[:3]; d.qpos[3:7] = root[[6,3,4,5]]; d.qpos[qids] = z['q'][i,j]
        mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d)
        clearance[i,j] = d.xpos[feet,2]-.028
        foot_xy[i,j] = d.xpos[feet,:2]
        com[i,j] = d.subtree_com[m.body('base_link').id]
result = {}
for j,it in enumerate(meta['iterations']):
    phases = {}
    for label,mask in [('all_after2s',z['time']>=2),('ground12_60s',(z['time']>=12)&(z['time']<=60)),('initial0_3s',z['time']<=3)]:
        entries = {}
        for k,leg in enumerate(['FL','FR','RL','RR']):
            x = clearance[mask,j,k]
            longest = current = 0
            for value in x:
                current = current+1 if value>.005 else 0; longest=max(longest,current)
            entries[leg] = {'above5mm_fraction':float((x>.005).mean()),'above10mm_fraction':float((x>.01).mean()),
                'mean_mm':float(x.mean()*1000),'p95_mm':float(np.quantile(x,.95)*1000),'max_mm':float(x.max()*1000),
                'longest_above5mm_s':longest/25}
        phases[label]=entries
    result[str(it)] = phases
np.savez_compressed(OUT/'foot_replay.npz',time=z['time'],clearance_m=clearance,com=com,foot_xy=foot_xy,feet=['FL','FR','RL','RR'],iterations=meta['iterations'])
(OUT/'foot_replay.json').write_text(json.dumps({'source':str(SOURCE),'method':'Exact URDF FK of recorded states, collision sphere bottom=zcenter-0.028; flat groundz0. Geometry only, not force measurement. One video replica per checkpoint.','results':result},indent=2)+'\n')
for it,r in result.items():print(it,{k:v for k,v in r['ground12_60s'].items()},flush=True)
