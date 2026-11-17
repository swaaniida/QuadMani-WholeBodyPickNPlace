"""Recover physical metrics from raw samples after diagnostics-only save failure."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--first-seconds',type=float);a=p.parse_args()
folder=Path(a.folder);x=np.load(folder/'policy_evaluation.npz')['samples']
if a.first_seconds:x=x[:int(a.first_seconds/.02)]
g=x.shape[1]//4;result={'note':'Physical raw samples only. Reward diagnostics/trajectory save did not complete; not a complete evaluation artifact.','seconds':len(x)*.02,'num_envs':x.shape[1]}
for i,name in enumerate(['stand','forward','turn','forward_turn']):
 v=x[100:,i*g:(i+1)*g];all_steps=x[:,i*g:(i+1)*g]
 failures=(all_steps[:,:,2]>.5)&(all_steps[:,:,3]<.5)
 result[name]=dict(mean_vx_m_s=float(v[:,:,0].mean()),mean_yaw_rad_s=float(v[:,:,1].mean()),
      fall_resets_total=int(failures.sum()),fall_resets_first_2s=int(failures[:100].sum()),
      ee_position_mean_m=float(v[:,:,33].mean()),no_fall_environment_fraction=float((~failures.any(0)).mean()))
(folder/'raw_metrics_recovered.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
