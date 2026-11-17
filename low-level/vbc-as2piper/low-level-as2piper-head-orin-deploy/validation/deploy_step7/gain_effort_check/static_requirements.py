import json
import numpy as np
import mujoco
import geometry_tools as g
out=g.OUT;m,d=g.m,g.d;rows=[]
for name,h,tilt,target,beta in [
 ('home',.3244069024246581,0,None,None),
 ('mid',.30,15,[.65,0,.30],160),
 ('floor',.30,25,[.90,0,.02],155),
 ('far_floor',.27,30,[.975,0,.02],150),
 ('upper',.3244069024246581,-15,[.60,0,.80],100)]:
 okleg=g.leg_ik(h,np.deg2rad(tilt));d.qpos[g.aq]=g.neutral;g.fk()
 if target is None:ok=True;pe=ae=0.
 else:ok,pe,ae=g.arm_ik(np.array(target),np.deg2rad(beta),[g.neutral,np.array([0,1.8,-1.4,0,-.5,0])])
 mujoco.mj_forward(m,d)
 rows.append(dict(name=name,body_height_m=h,body_pitch_deg=tilt,leg_ik=bool(okleg),arm_ik=bool(ok),position_error_m=float(pe),orientation_error_deg=float(np.rad2deg(ae)),arm_q=d.qpos[g.aq].tolist(),leg_q=d.qpos[g.lq].tolist(),gravity_tau_Nm=d.qfrc_bias[g.av].tolist(),actual_tcp=d.xpos[g.tcp].tolist(),contacts=g.collisions()))
(out/'static_requirements.json').write_text(json.dumps(dict(mass_kg=float(m.body_mass.sum()),urdf=str(g.URDF),rows=rows,note='Current camera geometry, no object payload; gravity-only arm torque, no friction/thermal/dynamic rating.'),indent=2))
print(json.dumps(rows,indent=2))
