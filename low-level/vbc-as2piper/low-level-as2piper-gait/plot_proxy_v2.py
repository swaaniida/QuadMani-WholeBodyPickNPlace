"""Show actual motion and falls; a stationary low-fall policy is not success."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/gait_reward_proxy_v2'
data=json.loads((OUT/'comparison.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13,4))
for case in ['original_vbc_port','stable_net_force','stable_proxy','progress_gain2_proxy','stable_proxy_ground','progress_gain2_proxy_yaw4','progress_gain2_proxy_ground_floor','progress_gain2_proxy_smooth_yaw4','progress_gain2_proxy_smooth_yaw4_contactmatch','progress_gain2_proxy_smooth_yaw4_footground','progress_gain2_proxy_smooth_yaw4_contactmatch_deadband']:
 points=[]
 for label,d in data.items():
  if label.rsplit('_',1)[0] not in (case,case+'_resume1'):continue
  if d.get('envs_per_command')!=16:continue
  points.append((d['completed_updates'],d['forward']['mean_vx_m_s'],d['turn']['mean_yaw_rad_s'],sum(d[k]['fall_resets_total'] for k in ['stand','forward','turn','forward_turn']),'resume1' in label or case in ('progress_gain2_proxy_yaw4','progress_gain2_proxy_ground_floor')))
 points.sort()
 if not points:continue
 line=None
 for axis,column in zip(axes,[1,2,3]):
  line,=axis.plot([p[0] for p in points],[p[column] for p in points],marker='o',label=case)
  for p in points:
   if p[4]:axis.scatter(p[0],p[column],marker='s',s=70,facecolors='none',edgecolors=line.get_color())
axes[0].axhline(.25,color='black',ls='--',lw=1,label='command')
axes[1].axhline(.6,color='black',ls='--',lw=1)
for ax,title in zip(axes,['Forward measured vx (m/s)','Turn measured yaw rate (rad/s)','Fall resets: 64 envs x 20s']):
 ax.set(title=title,xlabel='Completed PPO updates');ax.grid(alpha=.25)
axes[0].legend(fontsize=7)
fig.suptitle('Same rough terrain/full DR; outlined square = resumed training')
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160)
