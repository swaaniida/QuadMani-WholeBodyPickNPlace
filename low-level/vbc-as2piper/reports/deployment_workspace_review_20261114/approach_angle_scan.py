import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
code=(OUT/'workspace_scan.py').read_text().split('rows=[];postures=[];start=time.time()')[0]
ns={'__file__':str(OUT/'workspace_scan.py'),'__name__':'approach_scan'};exec(compile(code,str(OUT/'workspace_scan.py'),'exec'),ns)
m,d=ns['m'],ns['d'];aq,lq=ns['aq'],ns['lq'];neutral=ns['neutral'];rows=[]
for h,t in [(.30,25),(.27,30),(.27,35)]:
    assert ns['leg_ik'](h,np.deg2rad(t));legs=d.qpos[lq].copy();root=d.qpos[:7].copy()
    for beta in [140,150,155,160,175,180]:
        last=neutral.copy();valid=[]
        for x in np.arange(.65,1.1001,.0125):
            d.qpos[:7]=root;d.qpos[lq]=legs
            ok,pe,ae=ns['arm_ik'](np.array([x,0,.02]),np.deg2rad(beta),[last,neutral,np.array([0,1.8,-1.4,0,-.5,0])])
            if not ok:continue
            last=d.qpos[aq].copy()
            if ns['collisions']():continue
            com=d.subtree_com[m.body('base_link').id];hull=ns['hull'];margin=float(-np.max(hull.equations[:,:2]@com[:2]+hull.equations[:,2]))
            reserve=float(np.rad2deg(np.minimum(d.qpos[aq]-ns['alo'],ns['ahi']-d.qpos[aq]).min()))
            if margin<.02 or reserve<5:continue
            valid.append({'x_m':float(x),'arm_q':d.qpos[aq].tolist(),'margin_m':margin,'joint_reserve_deg':reserve,'position_error_m':float(pe)})
        rows.append({'base_z_m':h,'tilt_deg':t,'tool_pitch_deg':beta,'tool_tilt_from_down_deg':180-beta,'max_x_m':max((v['x_m'] for v in valid),default=None),'farthest':max(valid,key=lambda v:v['x_m']) if valid else None})
    print(h,t,[(r['tool_pitch_deg'],r['max_x_m']) for r in rows if r['base_z_m']==h and r['tilt_deg']==t],flush=True)
(OUT/'approach_angle_scan.json').write_text(json.dumps({'rows':rows,'step_x_m':.0125,'target_z_m':.02,'note':'Static sampled geometry with current D435i payload,5deg joint-limit allowance,20mm CoM allowance. Grasp-object geometry and dynamics not validated.'},indent=2)+'\n')
