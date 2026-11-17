"""Audit actual reward functions: small commands, signs, peak and continuity."""
import isaacgym
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from legged_gym.envs.manip_loco.as2piper_gait_study import AS2PiPERGaitStudy,AS2PiPERGaitStudyCfg,configure
ROOT=Path(__file__).resolve().parent
OLD='progress_gain2_proxy_yaw4';NEW='progress_gain2_proxy_smooth_yaw4'

def score(preset,axis,command,velocity):
    velocity=torch.as_tensor(np.atleast_1d(velocity),dtype=torch.float32)
    commands=torch.zeros(len(velocity),3);commands[:,axis]=command
    base=torch.zeros_like(commands);base[:,axis]=velocity
    e=SimpleNamespace(cfg=configure(AS2PiPERGaitStudyCfg(),preset),commands=commands,base_lin_vel=base,base_ang_vel=base)
    function=AS2PiPERGaitStudy._speed_tracking if axis==0 else AS2PiPERGaitStudy._yaw_tracking
    return function(e)[0].numpy()

report={'verified':[],'low_command_jump':{}}
for axis,commands,span in [(0,[-.634,-.25,-.05,-.00101,-.00099,0.,.00099,.00101,.05,.25,.634],1.),(2,[-1.,-.6,-.1,-.00101,0.,.00101,.1,.6,1.],2.)]:
    for command in commands:
        v=np.unique(np.r_[np.linspace(-span,span,2001),command])
        value=score(NEW,axis,command,v);at_goal=float(score(NEW,axis,command,command)[0])
        assert value.max()<=1.000001 and at_goal>=value.max()-1e-6,(axis,command)
        np.testing.assert_allclose(value,score(NEW,axis,-command,-v),atol=2e-6)
    physical=.4 if axis==0 else .5
    zero=float(score(NEW,axis,0.,physical)[0]);tiny=float(score(NEW,axis,1e-8,physical)[0])
    assert abs(zero-tiny)<1e-6
    oldjump=float(np.diff([score(OLD,axis,c,physical)[0] for c in [.000999,.001001]])[0])
    newjump=float(np.diff([score(NEW,axis,c,physical)[0] for c in [.000999,.001001]])[0])
    assert oldjump>.1 and abs(newjump)<1e-3
    report['low_command_jump'][str(axis)]={'old_jump_across_0_001':oldjump,'new_jump_across_0_001':newjump,'zero_command_score':zero,'score_at_command_1e_8':tiny}
    report['verified'].append({'axis':axis,'command_sweep':commands,'peak_at_target':True,'sign_symmetry':True,'continuous_near_zero':True})
report['formula']='mix=0.5*min(abs(command)/sigma,1); planar uses command norm. Return(1-mix)*Gaussian+mix*clipped directional progress. Exact zero -> Gaussian, unchanged max1. No command range/sampling/environment change.'
path=ROOT/'experiments/gait_reward_proxy_v2/smooth_tracking_probe.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(10,6))
for row,axis in enumerate([0,2]):
    physical=.4 if axis==0 else .5
    velocity=np.linspace(-1.,1.,801);commands=np.linspace(-.02,.02,401)
    for preset,label,color in [(OLD,'constant mix','#b55d25'),(NEW,'smooth mix','#2877af')]:
        axes[row,0].plot(velocity,score(preset,axis,.01,velocity),label=label,color=color)
        values=[score(preset,axis,c,physical)[0] for c in commands]
        axes[row,1].plot(commands,values,label=label,color=color)
    axes[row,0].set(xlabel='Actual velocity',ylabel='Normalized reward',title=('Planar' if axis==0 else 'Yaw')+' command=0.01')
    axes[row,1].set(xlabel='Command near zero',ylabel='Normalized reward',title='Fixed wrong velocity='+str(physical))
for ax in axes.flat:ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(path.with_suffix('.png'),dpi=160)
