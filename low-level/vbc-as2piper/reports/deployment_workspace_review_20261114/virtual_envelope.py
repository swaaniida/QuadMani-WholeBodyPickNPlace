"""Screen proposed B1-style terrain-height-invariant target sphere offline."""
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
code=(OUT/'workspace_scan.py').read_text().split('rows=[];postures=[];start=time.time()')[0]
code=code.replace('def arm_ik(target,beta,starts):','def arm_ik(target,beta,starts,yaw=0.):')
code=code.replace('tq=np.array([np.cos(beta/2),0,np.sin(beta/2),0]);',
    'tq=np.array([np.cos(yaw/2)*np.cos(beta/2),-np.sin(yaw/2)*np.sin(beta/2),np.cos(yaw/2)*np.sin(beta/2),np.sin(yaw/2)*np.cos(beta/2)]);')
ns={'__file__':str(OUT/'workspace_scan.py'),'__name__':'geometry_tools'}
exec(compile(code,str(OUT/'workspace_scan.py'),'exec'),ns)
m,d,fk=ns['m'],ns['d'],ns['fk'];lq,aq=ns['lq'],ns['aq'];alo,ahi=ns['alo'],ns['ahi'];neutral=ns['neutral']
postures=[]
for h,theta in [(.3244069024246581,0),(.3244069024246581,-15),(.30,15),(.30,25),(.27,30),(.27,35)]:
    if ns['leg_ik'](h,np.deg2rad(theta)):postures.append({'height':h,'tilt':theta,'root':d.qpos[:7].copy(),'legs':d.qpos[lq].copy(),'last':neutral.copy()})
center=np.array([.22435,0,.4563069024246581]);samples=[]
for radius in [.40,.45,.50,.55,.60,.65,.70,.75,.80,.85]:
    for phi in [-1.15,-1.,-.85,-.70,-.55,-.40,-.25,0,.25,.50,.65]:
        for yaw in [-.60,-.35,0,.35,.60]:
            target=center+radius*np.array([np.cos(phi)*np.cos(yaw),np.cos(phi)*np.sin(yaw),np.sin(phi)])
            row={'r':radius,'phi':phi,'yaw':yaw,'target':target.tolist(),'valid':False,'below_ground':bool(target[2]<.01)}
            if row['below_ground']:samples.append(row);continue
            beta_top=2.455216137744035-phi
            beta_floor=3.05-.25*np.clip((target[0]-.55)/.40,0,1)
            blend=np.clip((.20-target[2])/.10,0,1)
            beta=(1-blend)*beta_top+blend*beta_floor
            row['tool_pitch_deg']=float(np.rad2deg(beta))
            for posture in postures:
                d.qpos[:7]=posture['root'];d.qpos[lq]=posture['legs']
                ok,pe,ae=ns['arm_ik'](target,beta,[posture['last'],neutral,np.array([0,1.8,-1.4,0,-.5,0])],yaw)
                if not ok:continue
                posture['last']=d.qpos[aq].copy();hits=ns['collisions']()
                com=d.subtree_com[m.body('base_link').id];hull=ns['hull']
                margin=float(-np.max(hull.equations[:,:2]@com[:2]+hull.equations[:,2]))
                reserve=float(np.rad2deg(np.minimum(d.qpos[aq]-alo,ahi-d.qpos[aq]).min()))
                if not hits and margin>=.02:
                    row.update(valid=True,body_z=posture['height'],body_tilt_deg=posture['tilt'],margin_m=margin,arm_joint_limit_reserve_deg=reserve,position_error_m=pe,orientation_error_deg=float(np.rad2deg(ae)))
                    break
            samples.append(row)
    print('radius',radius,'valid',sum(r['valid'] for r in samples),'checked',len(samples),flush=True)
    (OUT/'virtual_envelope_progress.json').write_text(json.dumps({'center':center.tolist(),'samples':samples},indent=2)+'\n')
summary={'tested':len(samples),'below_ground':sum(r['below_ground'] for r in samples),'valid':sum(r['valid'] for r in samples),'above_ground_invalid':sum(not r['valid'] and not r['below_ground'] for r in samples)}
(OUT/'virtual_envelope.json').write_text(json.dumps({'center':center.tolist(),'bounds':{'r':[.4,.85],'phi':[-1.15,.65],'yaw':[-.6,.6]},'postures':[{k:p[k] for k in ['height','tilt']} for p in postures],'summary':summary,'samples':samples,'scope':'Finite-grid sampled pose screen, not continuous/path/dynamic feasibility or real hardware validation. No production config changes.'},indent=2)+'\n')
print(summary,flush=True)
