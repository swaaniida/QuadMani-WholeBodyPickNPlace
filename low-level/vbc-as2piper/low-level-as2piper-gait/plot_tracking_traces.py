"""Actual multi-environment velocity traces; show spread rather than return."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/gait_reward_proxy_v2'
fig,axes=plt.subplots(2,4,figsize=(14,6),sharex=True)
for case,color in [('stable_proxy','#3674b1'),('progress_gain2_proxy','#d7791c')]:
 folder=OUT/case/'eval_200';meta=json.loads((folder/'policy_evaluation.json').read_text());x=np.load(folder/'policy_evaluation.npz')['samples'];n=meta['envs_per_command'];time=np.arange(len(x))*.02
 for j,name in enumerate(['stand','forward','turn','forward_turn']):
  for row,col in [(0,0),(1,1)]:
   v=x[:,j*n:(j+1)*n,col];low,mid,high=np.quantile(v,[.25,.5,.75],axis=1)
   axes[row,j].plot(time,mid,color=color,label=case,lw=1)
   axes[row,j].fill_between(time,low,high,color=color,alpha=.14)
for j,name in enumerate(['stand','forward','turn','forward_turn']):
 axes[0,j].axhline([0,.25,0,.25][j],ls='--',color='black',lw=1)
 axes[1,j].axhline([0,0,.6,.6][j],ls='--',color='black',lw=1)
 axes[0,j].set_title(name);axes[0,j].set_ylim(-.2,.7);axes[1,j].set_ylim(-1,1)
 for row in [0,1]:axes[row,j].grid(alpha=.2);axes[row,j].axvline(2,color='gray',ls=':',lw=.7)
 axes[1,j].set_xlabel('Simulation time (s)')
axes[0,0].set_ylabel('Body vx (m/s)');axes[1,0].set_ylabel('Body yaw rate (rad/s)')
axes[0,0].legend(fontsize=7)
fig.suptitle('201 completed updates, same rough terrain/full DR; median and interquartile range across 16 envs')
fig.tight_layout();fig.savefig(OUT/'tracking_traces_201.png',dpi=160)
