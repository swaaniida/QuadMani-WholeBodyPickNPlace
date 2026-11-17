"""Plot measured policy motion and foot contact; no scripted reference motion."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('evaluation');a=p.parse_args();out=Path(a.evaluation)
    x=np.load(str(out/'policy_evaluation.npz'))['samples'];d=json.loads((out/'policy_evaluation.json').read_text())
    t=np.arange(len(x))*.02;mask=(t>=2)&(t<=8)
    fig,axes=plt.subplots(2,2,figsize=(12,6),sharex=True)
    ax=axes[0,0];v=x[:,8:16,0];ax.plot(t,v.mean(1),label='Measured forward speed');ax.fill_between(t,v.mean(1)-v.std(1),v.mean(1)+v.std(1),alpha=.2)
    ax.axhline(d['forward']['command_vx_m_s'],color='black',ls='--',label='Command');ax.set_ylabel('Forward velocity (m/s)');ax.legend()
    ax=axes[0,1];v=x[:,16:24,1];ax.plot(t,v.mean(1),label='Measured turn rate');ax.fill_between(t,v.mean(1)-v.std(1),v.mean(1)+v.std(1),alpha=.2)
    ax.axhline(d['turn']['command_yaw_rad_s'],color='black',ls='--',label='Command');ax.set_ylabel('Yaw velocity (rad/s)');ax.legend()
    for i,name in enumerate(['FL','FR','RL','RR']):
        axes[1,0].plot(t,x[:,8,5+i],label=name,lw=1)
        axes[1,1].plot(t,x[:,8,9+i]*1000,label=name,lw=1)
    axes[1,0].set_ylabel('Foot normal force (N), forward env 8');axes[1,1].set_ylabel('Foot clearance (mm), forward env 8')
    axes[1,0].legend(ncol=4);axes[1,1].legend(ncol=4)
    for ax in axes.flat:ax.set_xlim(2,8);ax.grid(alpha=.25);ax.set_xlabel('Time (s)')
    fig.suptitle('Actual PhysX policy evaluation: '+Path(d['checkpoint']).name)
    fig.tight_layout();fig.savefig(str(out/'walking_evidence.png'),dpi=150);plt.close(fig)


if __name__=='__main__':main()
