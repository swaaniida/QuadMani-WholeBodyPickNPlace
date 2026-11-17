"""Fixed-base CPU PhysX gain/effort sensitivity, no learning or robot commands.

Effort profiles other than legacy are diagnostic assumptions, NOT motor ratings.
Full current-camera asset, gravity, same 50Hz target hold / 200Hz physics timing.
"""
import isaacgym
from isaacgym import gymapi,gymtorch
import json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np
import torch
O=Path(__file__).parent;ROOT=O.resolve().parents[2]
spec=importlib.util.spec_from_file_location('piper_control',ROOT/'legged_gym/utils/piper_control.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);ArmCommandFilter=module.ArmCommandFilter
poses=json.loads((O/'static_requirements.json').read_text())['rows']
assert all(p['arm_ik'] and not p['contacts'] for p in poses)
gains=[(100,10),(200,20),(280,28),(400,40),(520,52),(800,80),(400,10)]
profiles={'legacy100':[100]*6,'diagnostic32_8':[32]*3+[8]*3,'diagnostic16_4':[16]*3+[4]*3,'diagnostic8_2':[8]*3+[2]*3}
cases=[dict(pose=p['name'],pose_idx=i,kp=kp,kd=kd,effort_name=name,effort=limit) for i,p in enumerate(poses) for kp,kd in gains for name,limit in profiles.items()]
N=len(cases);g=gymapi.acquire_gym();par=gymapi.SimParams();par.dt=.005;par.substeps=1;par.up_axis=gymapi.UP_AXIS_Z;par.gravity=gymapi.Vec3(0,0,-9.81);par.use_gpu_pipeline=False;par.physx.use_gpu=False;par.physx.num_threads=2;par.physx.solver_type=1;par.physx.num_position_iterations=4;par.physx.num_velocity_iterations=0
sim=g.create_sim(0,-1,gymapi.SIM_PHYSX,par);assert sim is not None
asset_path=ROOT/'resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf'
opts=gymapi.AssetOptions();opts.fix_base_link=True;opts.collapse_fixed_joints=True;opts.flip_visual_attachments=False;opts.default_dof_drive_mode=gymapi.DOF_MODE_POS;opts.use_mesh_materials=True
asset=g.load_asset(sim,str(asset_path.parent),asset_path.name,opts)
dofnames=g.get_asset_dof_names(asset);ndof=len(dofnames);assert ndof==20
baseprops=g.get_asset_dof_properties(asset);envs=[];actors=[];q0=[];tcps=[]
for i,c in enumerate(cases):
 p=poses[c['pose_idx']];env=g.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,2),int(np.ceil(np.sqrt(N))))
 tf=gymapi.Transform();tf.p=gymapi.Vec3(0,0,1.0);tf.r=gymapi.Quat.from_euler_zyx(0,np.deg2rad(p['body_pitch_deg']),0)
 actor=g.create_actor(env,asset,tf,'robot',i,1,0) # self collisions disabled ONLY for isolated response bench
 prop=baseprops.copy();prop['driveMode'][:]=gymapi.DOF_MODE_POS;prop['stiffness'][:]=400;prop['damping'][:]=40
 prop['stiffness'][12:18]=c['kp'];prop['damping'][12:18]=c['kd'];prop['effort'][12:18]=c['effort'];prop['velocity'][12:18]=3.
 g.set_actor_dof_properties(env,actor,prop);g.enable_actor_dof_force_sensors(env,actor)
 q=np.r_[p['leg_q'],p['arm_q'],.049,-.049].astype(np.float32);state=np.zeros(20,dtype=gymapi.DofState.dtype);state['pos']=q;g.set_actor_dof_states(env,actor,state,gymapi.STATE_ALL);g.set_actor_dof_position_targets(env,actor,q)
 envs.append(env);actors.append(actor);q0.append(q);tcps.append(g.find_actor_rigid_body_handle(env,actor,'piper_umi_nominal_tcp'))
assert len(set(tcps))==1 and tcps[0]>=0
g.prepare_sim(sim)
state=gymtorch.wrap_tensor(g.acquire_dof_state_tensor(sim)).view(N,ndof,2)
force=gymtorch.wrap_tensor(g.acquire_dof_force_tensor(sim)).view(N,ndof)
body=gymtorch.wrap_tensor(g.acquire_rigid_body_state_tensor(sim)).view(N,-1,13)
q0=np.array(q0);initial=torch.tensor(q0[:,12:18]);lo=torch.tensor(baseprops['lower'][12:18].copy());hi=torch.tensor(baseprops['upper'][12:18].copy());filt=ArmCommandFilter(initial,lo,hi,3.,.02)
# TCP FK at target q, via exact current MJCF, provides a position reference independent of drive error.
# A separate CPU geometry worker computes these references after this simulation; store root/target state here.
records=[];start=time.time();delta=np.array([.06,.10,-.08,.06,.04,-.06],dtype=np.float32)
for step in range(1400):
 t=step*.005
 if step%4==0:
  if t<1.5:factor=0.
  elif t<3.5:factor=1.
  elif t<5.5:factor=float(np.sin(2*np.pi*.5*(t-3.5)))
  else:factor=0.
  desired=torch.tensor(q0[:,12:18]+factor*delta)
  armcmd=filt.step(desired).numpy();commands=q0.copy();commands[:,12:18]=armcmd
 g.set_dof_position_target_tensor(sim,gymtorch.unwrap_tensor(torch.from_numpy(commands)))
 g.simulate(sim);g.fetch_results(sim,True);g.refresh_dof_state_tensor(sim);g.refresh_dof_force_tensor(sim);g.refresh_rigid_body_state_tensor(sim)
 if step%4==0:
  # q6, dq6, generalized force6, actual TCPxyz+quat7, actual command6
  value=np.concatenate([state[:,12:18,0].numpy(),state[:,12:18,1].numpy(),force[:,12:18].numpy(),body[:,tcps[0],:7].numpy(),commands[:,12:18]],axis=1)
  assert np.isfinite(value).all();records.append(value.copy())
 if step%200==0:print('step',step,'actors',N,'seconds',round(time.time()-start,1),flush=True)
np.savez_compressed(O/'physx_signals.npz',values=np.asarray(records),time=(np.arange(len(records))*4+1)*.005,initial_q=q0,cases_json=json.dumps(cases),columns='q6,dq6,generalized_force6,tcp_xyz_quat7,command6')
readbacks=[g.get_actor_dof_properties(e,a) for e,a in zip(envs,actors)]
for c,p in zip(cases,readbacks):
 assert np.all(p['stiffness'][12:18]==c['kp']) and np.all(p['damping'][12:18]==c['kd']) and np.allclose(p['effort'][12:18],c['effort'])
meta=dict(envs=N,seconds=7,physics_hz=200,target_update_hz=50,fixed_base=True,self_collisions=False,effort_profiles_are_assumptions=True,cases=cases,urdf_sha256=hashlib.sha256(asset_path.read_bytes()).hexdigest(),optimizer_updates=0,elapsed_s=time.time()-start,force_note='Raw PhysX DOF generalized-force sensor; drive-torque interpretation must be checked against gain law and effort clamp before reporting as motor torque.')
(O/'bench_metadata.json').write_text(json.dumps(meta,indent=2));print('DONE',N,len(records),flush=True);g.destroy_sim(sim)
