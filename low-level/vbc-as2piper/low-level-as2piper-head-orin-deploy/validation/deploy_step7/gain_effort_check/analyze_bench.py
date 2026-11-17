import json,csv,time
import numpy as np
import mujoco
import geometry_tools as g
O=g.OUT;z=np.load(O/'physx_signals.npz');v=z['values'];t=z['time'];cases=json.loads(str(z['cases_json']));poses=json.loads((O/'static_requirements.json').read_text())['rows'];m,d=g.m,g.d
m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
start=time.time();rows=[];actual_fk_residual=[];traces={}
for i,c in enumerate(cases):
 p=poses[c['pose_idx']];theta=np.deg2rad(p['body_pitch_deg']);d.qpos[:3]=[0,0,1.];d.qpos[3:7]=[np.cos(theta/2),0,np.sin(theta/2),0];d.qpos[g.lq]=p['leg_q']
 q=v[:,i,:6];dq=v[:,i,6:12];sensor=v[:,i,12:18];actual=v[:,i,18:21];cmd=v[:,i,25:31];ddq=np.gradient(dq,t,axis=0)
 expected=[];fkactual=[];required=[]
 for k in range(len(t)):
  d.qpos[g.aq]=cmd[k];g.fk();expected.append(d.xpos[g.tcp].copy())
  d.qpos[g.aq]=q[k];d.qvel[:]=0;d.qacc[:]=0;d.qvel[g.av]=dq[k];d.qacc[g.av]=ddq[k];mujoco.mj_inverse(m,d)
  fkactual.append(d.xpos[g.tcp].copy());required.append(d.qfrc_inverse[g.av].copy())
 expected=np.array(expected);fkactual=np.array(fkactual);required=np.array(required)
 # Constant environment translation is inferred using FK at observed q, not desired q.
 offset=actual[0]-fkactual[0];residual=np.linalg.norm(actual-(fkactual+offset),axis=1);actual_fk_residual.append(float(residual.max()))
 error=np.linalg.norm(actual-(expected+offset),axis=1);qerr=np.rad2deg(q-cmd)
 hold=(t>.8)&(t<1.45);step=(t>1.5)&(t<3.5);sine=(t>3.5)&(t<5.5);final=t>6.
 limit=np.array(c['effort']);pd=c['kp']*(cmd-q)-c['kd']*dq
 row=dict(**c,hold_tcp_error_cm=float(error[hold].mean()*100),step_tcp_error_cm=float(error[step].mean()*100),sine_tcp_error_cm=float(error[sine].mean()*100),all_tcp_p95_cm=float(np.percentile(error,95)*100),
 hold_joint_rms_deg=float(np.sqrt(np.mean(qerr[hold]**2))),sine_joint_rms_deg=float(np.sqrt(np.mean(qerr[sine]**2))),final_tcp_std_mm=float(np.linalg.norm(np.std(actual[final],axis=0))*1000),
 max_joint_velocity_rad_s=float(np.abs(dq).max()),model_ID_tau_abs_p99_Nm=np.percentile(abs(required),99,axis=0).tolist(),model_ID_tau_abs_max_Nm=abs(required).max(0).tolist(),
 generalized_force_abs_p99_Nm=np.percentile(abs(sensor),99,axis=0).tolist(),hold_model_ID_tau_Nm=required[hold].mean(0).tolist(),hold_generalized_force_Nm=sensor[hold].mean(0).tolist(),
 explicit_PD_demand_abs_p99_Nm=np.percentile(abs(pd),99,axis=0).tolist(),explicit_PD_demand_exceeds_assumed_effort_fraction=float(np.mean(abs(pd)>limit)),FK_actual_max_residual_mm=float(residual.max()*1000))
 rows.append(row)
 if c['pose']=='far_floor' and c['kp'] in (100,200,280,400,520,800) and c['kd']!=10:traces[str(i)]=dict(time=t.tolist(),tcp_error_cm=(error*100).tolist(),joint2_q=q[:,1].tolist(),joint2_goal=cmd[:,1].tolist(),**{k:c[k] for k in ['kp','kd','effort_name']})
 if i%28==0:print('cases analyzed',i,'seconds',round(time.time()-start,1),flush=True)
result=dict(cases=rows,count=len(rows),max_actual_FK_residual_mm=max(actual_fk_residual)*1000,
 scope='Fixed-base 7-second CPU PhysX sensitivity only; diagnostic effort caps are assumptions; inverse-dynamics torque is a MuJoCo model estimate from 50Hz kinematic samples, not measured motor torque.',
 sensor_warning='PhysX DOF generalized force is near zero at loaded static holds; it is NOT the motor/position-drive torque. Do not use it to certify effort limits.',optimizer_updates=0)
(O/'bench_results.json').write_text(json.dumps(result,indent=2));(O/'bench_traces.json').write_text(json.dumps(traces))
keys=['pose','kp','kd','effort_name','hold_tcp_error_cm','step_tcp_error_cm','sine_tcp_error_cm','all_tcp_p95_cm','hold_joint_rms_deg','max_joint_velocity_rad_s','final_tcp_std_mm','FK_actual_max_residual_mm']
with (O/'bench_results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
print('DONE',len(rows),'max FK discrepancy mm',result['max_actual_FK_residual_mm'],flush=True)
