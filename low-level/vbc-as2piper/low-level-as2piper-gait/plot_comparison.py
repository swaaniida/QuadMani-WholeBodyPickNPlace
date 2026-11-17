"""Standalone measured ablation figure."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_study_report import OUT, PRESETS

d=json.loads((OUT/'comparison_detailed.json').read_text())
names=[p for p in PRESETS if p in d]
fig,axes=plt.subplots(2,2,figsize=(12,8))
panels=[('stand','mean_vx_m_s','Stop command: forward speed (m/s)',0,None),
        ('forward','mean_vx_m_s','Forward command: speed (m/s)',.25,(.15,.35)),
        ('turn','mean_yaw_rad_s','Turn command: yaw rate (rad/s)',.6,(.35,.85)),
        ('forward','stance_slip_mean_m_s','Forward: estimated stance slip (m/s)',None,None)]
for ax,(group,key,title,target,band) in zip(axes.flat,panels):
    values=[d[name][group][key] for name in names]
    if band:ax.axvspan(*band,color='green',alpha=.1)
    ax.barh(names,values,color=['#888888' if n=='original' else '#267eb5' for n in names])
    if target is not None:ax.axvline(target,color='black',ls='--',label='Command')
    ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.2)
    for i,v in enumerate(values):ax.text(v,i,' %.3f'%v,va='center')
fig.suptitle('Paired reward recovery study: same model313, Adam, seed7, 120 updates / 1024 envs')
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
