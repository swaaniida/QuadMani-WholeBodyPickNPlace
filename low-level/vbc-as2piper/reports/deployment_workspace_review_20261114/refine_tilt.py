import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
source=(OUT/'workspace_scan.py').read_text().split('rows=[];postures=[];start=time.time()')[0]
ns={'__file__':str(OUT/'workspace_scan.py'),'__name__':'refine_tools'};exec(compile(source,str(OUT/'workspace_scan.py'),'exec'),ns)
m,d=ns['m'],ns['d'];aq,lq=ns['aq'],ns['lq'];neutral=ns['neutral'];postures={};rows=[]
for h in [.3244069024246581,.30,.27]:
    for t in np.arange(0,35.01,2.5):
        if ns['leg_ik'](h,np.deg2rad(t)):postures[h,float(t)]=(d.qpos[:7].copy(),d.qpos[lq].copy())
for h in [.3244069024246581,.30,.27]:
    for x in [.60,.75,.80,.85,.90,.925,.95]:
        target=np.array([x,0,.02]);virtual_phi=np.arctan2(.02-.4563069024246581,x-.22435)
        for mode in ['current_actual_center_law','virtual_center_old_orientation_law','tool160','tool175']:
            found=None
            for t in np.arange(0,35.01,2.5):
                if (h,float(t)) not in postures:continue
                root,legs=postures[h,float(t)];d.qpos[:7]=root;d.qpos[lq]=legs;d.qpos[aq]=neutral;ns['fk']()
                arm=d.xpos[m.body('piper_base_link').id].copy();vec=target-arm;physical_phi=np.arctan2(vec[2],np.linalg.norm(vec[:2]))
                beta={'current_actual_center_law':2.455216137744035-physical_phi,'virtual_center_old_orientation_law':2.455216137744035-virtual_phi,'tool160':np.deg2rad(160),'tool175':np.deg2rad(175)}[mode]
                ok,pe,ae=ns['arm_ik'](target,beta,[neutral,np.array([0,1.8,-1.4,0,-.5,0])])
                if not ok or ns['collisions']():continue
                com=d.subtree_com[m.body('base_link').id];hull=ns['hull'];margin=float(-np.max(hull.equations[:,:2]@com[:2]+hull.equations[:,2]))
                reserve=float(np.rad2deg(np.minimum(d.qpos[aq]-ns['alo'],ns['ahi']-d.qpos[aq]).min()))
                if margin<.02 or reserve<5:continue
                found={'tilt_deg':float(t),'radius_actual':float(np.linalg.norm(vec)),'phi_actual':float(physical_phi),'radius_virtual':float(np.linalg.norm(target-[.22435,0,.4563069024246581])),'phi_virtual':float(virtual_phi),'beta_deg':float(np.rad2deg(beta)),'margin_m':margin,'arm_limit_reserve_deg':reserve,'arm_q':d.qpos[aq].tolist(),'leg_q':legs.tolist(),'position_error_m':float(pe)}
                break
            rows.append({'base_z_m':h,'x_ground_m':x,'mode':mode,'first_sampled_valid':found})
    print('height',h,'done',flush=True)
(OUT/'refined_tilt.json').write_text(json.dumps({'resolution_deg':2.5,'target_z_m':.02,'arm_joint_limit_allowance_deg':5,'com_margin_allowance_m':.02,'rows':rows,'scope':'Offline finite-grid kinematic test, fixed feet and current camera URDF; dynamic/torque/real-hardware validity not claimed.'},indent=2)+'\n')
print('DONE',len(rows),flush=True)
