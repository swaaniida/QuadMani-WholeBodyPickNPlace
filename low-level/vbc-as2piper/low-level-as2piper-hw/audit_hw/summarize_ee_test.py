"""Summarize raw applied-target errors; reconstruct true age after resets."""
import json,os,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
OUT=Path(os.environ.get('EE_AUDIT_DIR',str(ROOT/'audit_hw/ppo_armature20')))


def stats(a,mask):
    v=a[mask]
    if not len(v):return dict(count=0)
    return dict(count=len(v),position_mean_m=float(v[:,0].mean()),position_p50_m=float(np.quantile(v[:,0],.5)),
        position_p95_m=float(np.quantile(v[:,0],.95)),orientation_mean_deg=float(v[:,1].mean()),orientation_p95_deg=float(np.quantile(v[:,1],.95)),
        position_lt_2cm_fraction=float((v[:,0]<.02).mean()),position_lt_5cm_fraction=float((v[:,0]<.05).mean()),
        pose_lt_5cm_10deg_fraction=float(((v[:,0]<.05)&(v[:,1]<10)).mean()),contact_gt_1N_fraction=float((v[:,7]>1).mean()),
        force_max_p95_Nm=float(np.quantile(v[:,8],.95)))


def main():
    rows=[json.loads(l) for l in (OUT/'iterations.jsonl').read_text().splitlines()]
    data=np.load(str(OUT/'tracking_samples.npz'))['samples']
    # Runner randomizes episode_length_buf at startup. Its first episode age
    # is artificial; raw done sequences reconstruct actual elapsed test time.
    age=np.zeros(data.shape[:2],dtype=np.float32);counter=np.zeros(data.shape[1],dtype=np.float32)
    for t in range(len(data)):
        counter+=.02;age[t]=counter;counter[data[t,:,6]>.5]=0
    up=data[:,:,3]>.5;random=data[:,:,4]>.5;hold=data[:,:,5]>.5;late=age>.5
    result=dict(iterations_completed=len(rows),control_steps=len(data),sampled_envs=data.shape[1],
        groups={label:stats(data,mask) for label,mask in [('all',np.ones_like(up)),('upright_late',up&late),('upright_random_late',up&late&random),
             ('upright_random_hold',up&late&random&hold),('fallen',~up),('post_reset_first_0p1s',age<=.10001)]},
        random_exposure_fraction=float(random.mean()),upright_fraction=float(up.mean()),
        age_note='Age reconstructed from captured reset events; original startup episode_length_buf was randomized by the PPO runner.')
    for window,subset in [('all_iterations',rows),('last_20_iterations',rows[-20:])]:
        reset={k:sum(r['resets'][k] for r in subset) for k in rows[0]['resets']}
        sampling={k:sum(r['sampling'][k] for r in subset) for k in rows[0]['sampling']}
        if sampling['requested']:sampling['fallback_fraction']=sampling['fallback']/sampling['requested']
        result[window]=dict(reset_counts=reset,sampling=sampling)
    log=ROOT/'audit_forward/ee_100_stdout.log'
    if log.exists():
        text=re.sub(r'\x1b\[[0-9;]*m','',log.read_text())
        lengths=[float(x) for x in re.findall(r'Mean episode length:\s+([\d.]+)',text)]
        if lengths:result['last_logged_mean_episode_s']=lengths[-1]*.02
    fixed=ROOT/'audit_forward/ee_fixed_base_20261029/summary.json'
    if fixed.exists():result['fixed_base']=json.loads(fixed.read_text())
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for label,selection in [('All sampled states',np.ones_like(up)),('Upright >0.5s',up&late),('Upright random >0.5s',up&late&random)]:
        for ax,column,factor in [(axes[0,0],0,100),(axes[0,1],1,1)]:
            y=[]
            for row in rows:
                start=row['iteration']*24;end=start+24;part=data[start:end];mask=selection[start:end]
                y.append(float(part[:,:,column][mask].mean())*factor if mask.any() else np.nan)
            ax.plot([r['iteration'] for r in rows],y,label=label)
    axes[0,0].set_ylabel('Applied-target position error [cm]');axes[0,1].set_ylabel('Orientation error [deg]')
    axes[0,0].axhline(5,color='gray',ls='--');axes[0,1].axhline(10,color='gray',ls='--')
    for key in ['roll_exclusive','pitch_exclusive','height_exclusive','timeout_exclusive']:
        axes[1,0].plot([r['iteration'] for r in rows],[r['resets'][key] for r in rows],label=key)
    axes[1,0].set_ylabel('Episode ends per iteration')
    axes[1,1].plot([r['iteration'] for r in rows],[r['sampling']['requested'] for r in rows],label='Goal requests')
    axes[1,1].plot([r['iteration'] for r in rows],[r['sampling']['fallback'] for r in rows],label='Fallback holds')
    axes[1,1].set_ylabel('Resampling events')
    for ax in axes.flat:ax.set_xlabel('PPO iteration');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Actual PhysX errors before reset; hardware-calibrated flat-ground task\nError curves use the recorded env subset and true age reconstructed from reset events.',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(str(OUT/'training_diagnostics.png'),dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
