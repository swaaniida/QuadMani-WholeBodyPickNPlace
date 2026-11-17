"""Plot measured command tracking and falls under the full environment."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent/'experiments/gait_full_env_v1'
GROUPS=['stand','forward','turn','forward_turn']
fig,axes=plt.subplots(1,3,figsize=(12,3.6))
for preset in ['original','speed','stable','progress','progress_gain2','gaussian_gain2']:
    points=[]
    for pilot in [True,False]:
        folder=OUT/(preset+'_early' if pilot else preset)
        for f in folder.glob('eval_*/gait_summary.json'):
            d=json.loads(f.read_text());samples=np.load(f.parent/'policy_evaluation.npz')['samples']
            vx=np.mean([d[k]['vx_abs_error_m_s'] for k in GROUPS]);yaw=np.mean([d[k]['yaw_abs_error_rad_s'] for k in GROUPS])
            falls=sum(d[k]['fall_resets_total'] for k in GROUPS)/(samples.shape[1]*samples.shape[0]*.02/60)
            points.append((d.get('completed_updates',int(f.parent.name.split('_')[-1])+1),vx,yaw,falls))
    points.sort()
    if points:
        values=np.array(points)
        for i,ax in enumerate(axes):ax.plot(values[:,0],values[:,i+1],marker='o',label=preset)
for ax,title in zip(axes,['Mean vx MAE (m/s)','Mean yaw MAE (rad/s)','Falls per robot-minute']):
    ax.set(title=title,xlabel='Completed PPO updates');ax.grid(alpha=.3)
if any(ax.lines for ax in axes):axes[0].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150)
