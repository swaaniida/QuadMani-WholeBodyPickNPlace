"""Worst goal: test static compensation by placing front feet 3/5cm forward."""
from pathlib import Path
# Reuse support LP/kinematics, without executing the grid scan.
source=(Path(__file__).resolve().parent/'scan.py').read_text();exec(compile(source.split('put(home);baseline=support();jobs=[]')[0],str(Path(__file__)), 'exec'))
summary=json.loads((O/'summary.json').read_text());rec=summary['worst_goal'];target=np.array(rec['target']);phi,yaw,delta=rec['phi'],rec['yaw'],rec['delta'];rot=Rotation.from_euler('ZYX',[yaw+delta[2],2.455216137744035-phi+delta[1],delta[0]]).as_matrix()
q=np.array(summary['worst_pose']['qpos']);angles=Rotation.from_quat(q[3:7][[1,2,3,0]]).as_euler('xyz');seed=np.r_[q[2],angles[:2],q[g.lq],q[g.aq]];feet0=g.foot_targets.copy();results=[]
for step in [.03,.05]:
 goals=feet0.copy();goals[:2,0]+=step
 def err(x):
  put(x);return np.r_[(d.xpos[g.feet]-goals).ravel(),d.xpos[g.tcp]-target,.25*Rotation.from_matrix(rot@d.xmat[g.tcp].reshape(3,3).T).as_rotvec()]
 fit=least_squares(err,np.clip(seed,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=500,ftol=1e-11,xtol=1e-11,gtol=1e-11)
 e=err(fit.x);mujoco.mj_forward(m,d);hits=g.collisions();pe=float(np.linalg.norm(e[12:15]));ae=float(np.rad2deg(np.linalg.norm(e[15:])/.25));fe=float(np.max(np.abs(e[:12])))
 out=dict(front_foot_shift_m=step,target=target.tolist(),pose_pass=pe<.003 and ae<2 and fe<.002 and not hits,position_error_m=pe,orientation_error_deg=ae,foot_error_m=fe,hits=hits,qpos=d.qpos.tolist(),body_h_roll_pitch=fit.x[:3].tolist(),**support());results.append(out)
 print(step,out['pose_pass'],'frontmargin',out['front_edge_x_margin_m']*1000,'rear',out.get('rear_total_min_N'),flush=True)
(O/'stance_check.json').write_text(json.dumps(results,indent=2))
