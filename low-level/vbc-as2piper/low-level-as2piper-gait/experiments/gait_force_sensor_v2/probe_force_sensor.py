"""Calibrate simulator foot-force reporting against actual engine gravity load."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import isaacgym
from isaacgym import gymapi,gymutil,gymtorch
from isaacgym.torch_utils import quat_apply
import torch,numpy as np
from diagnostic_loader import load
AS2PiPERGaitStudy,AS2PiPERGaitStudyCfg,AS2PiPERGaitStudyCfgPPO,configure=load()
from legged_gym.utils.helpers import class_to_dict,set_seed
from legged_gym.utils.hw_runner import HardwareRunner
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['default','solver_world'],required=True);p.add_argument('--output',required=True);p.add_argument('--rough',action='store_true');p.add_argument('--sensor-link',choices=['foot','calf','hip'],default='foot');a=p.parse_args()
set_seed(20261104);cfg=configure(AS2PiPERGaitStudyCfg(),'stable_sensor' if a.mode=='solver_world' else 'stable',controlled=not a.rough)
cfg.study.sensor_link=a.sensor_link
cfg.env.num_envs=4;cfg.init_state.rand_yaw_range=0.;cfg.init_state.origin_perturb_range=0.;cfg.init_state.init_vel_perturb_range=0.
params=gymapi.SimParams();params.use_gpu_pipeline=True;params.physx.use_gpu=True;gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
e=AS2PiPERGaitStudy(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True)
def reset(ids):
 e.dof_pos[ids]=e.default_dof_pos;e.dof_vel[ids]=0
 e.gym.set_dof_state_tensor(e.sim,gymtorch.unwrap_tensor(e.dof_state))
e._reset_dofs=reset
e._resample_commands=lambda ids:e.commands.__setitem__(ids,0.)
r=HardwareRunner(e,class_to_dict(AS2PiPERGaitStudyCfgPPO()),None,'cuda:0')
r.load(str(ROOT/'logs/as2piper-low/gait_fullenv_stable_s7_v1/model_200.pt'));set_seed(20261104)
e.commands[:]=0;e.compute_observations()
masses=[]
for env in e.envs:
 actor=e.gym.get_actor_handle(env,0)
 masses.append(sum(float(prop.mass) for prop in e.gym.get_actor_rigid_body_properties(env,actor)))
rows=[]
with torch.no_grad():
 for step in range(250):
  e.goal_timer.zero_();action=r.alg.actor_critic.act_inference(e.obs_buf,hist_encoding=True);e.step(action)
  sensor=e.force_sensor_tensor.clone();force=sensor[:,:,:3]
  if a.mode=='default':force=quat_apply(e.rigid_body_state[:,e.feet_indices,3:7].reshape(-1,4),force.reshape(-1,3)).reshape(4,4,3)
  rows.append(dict(sensor_force_world=force.cpu().numpy(),sensor_raw=sensor.cpu().numpy(),net_contact=e.contact_forces[:,e.feet_indices].cpu().numpy(),root=e.root_states[:,:13].cpu().numpy(),angles=e._get_body_orientation().cpu().numpy()))
arrays={k:np.stack([row[k] for row in rows]) for k in rows[0]};weight=np.array(masses)*9.81
sensor_ratio=arrays['sensor_force_world'][100:,:,:,2].sum(-1).mean(0)/weight
net_ratio=arrays['net_contact'][100:,:,:,2].sum(-1).mean(0)/weight
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'forces.npz',**arrays,masses=np.array(masses))
result=dict(mode=a.mode,sensor_link=a.sensor_link,rough=a.rough,engine_actor_mass_kg=masses,sensor_Fz_to_weight=sensor_ratio.tolist(),net_Fz_to_weight=net_ratio.tolist(),mean_abs_root_vertical_velocity=np.abs(arrays['root'][100:,:,9]).mean(0).tolist(),root_z_std=arrays['root'][100:,:,2].std(0).tolist(),max_abs_roll_pitch=np.abs(arrays['angles'][100:]).max((0,2)).tolist(),note='No training. Canonical reset, fixed zero base command, learned standing policy. Last3s of5s averaged.')
(out/'calibration.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));e.gym.destroy_sim(e.sim)
