"""Offline spherical range design, current URDF; no training or online gate."""
import json, time, hashlib
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import geometry_tools as g

O=Path(__file__).resolve().parent
m,d=g.m,g.d
bid=m.body('base_link').id
start=time.time()
center=np.array([.28,0.,.50])
lo=np.r_[.25,np.deg2rad(-25),g.llo+1e-5,g.alo+1e-5]
hi=np.r_[.39,np.deg2rad(35),g.lhi-1e-5,g.ahi-1e-5]
home=np.r_[.3244069024,0,[0,.8,-1.6]*4,g.neutral]
cached=home.copy()
records=[]

def put(x):
    d.qpos[:3]=[0,0,x[0]]
    d.qpos[3:7]=[np.cos(x[1]/2),0,np.sin(x[1]/2),0]
    d.qpos[g.lq]=x[2:14]; d.qpos[g.aq]=x[14:]
    g.fk()

def pitch(target,phi):
    x=center[0]+np.linalg.norm((target-center)[:2])
    blend=np.clip((.25-target[2])/.15,0,1)
    return (1-blend)*(2.407684686936221-phi)+blend*np.deg2rad(175-25*np.clip((x-.55)/.425,0,1))

def solve(r,phi,yaw,delta):
    global cached
    target=center+r*np.array([np.cos(phi)*np.cos(yaw),np.cos(phi)*np.sin(yaw),np.sin(phi)])
    rot=Rotation.from_euler('ZYX',[yaw+delta[2],pitch(target,phi)+delta[1],delta[0]]).as_matrix()
    def err(x):
        put(x)
        return np.r_[(d.xpos[g.feet]-g.foot_targets).ravel(),d.xpos[g.tcp]-target,
                     .25*Rotation.from_matrix(rot@d.xmat[g.tcp].reshape(3,3).T).as_rotvec()]
    best=None
    seeds=[cached,home]
    # Known crouched ground witnesses plus alternate arm configuration.
    seeds.append(np.r_[.27,np.deg2rad(30),[0,.8,-1.6]*4,[0,.5,-.6,0,.5,0]])
    for seed in seeds:
        fit=least_squares(err,np.clip(seed,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=90,
                          ftol=1e-7,xtol=1e-7,gtol=1e-7)
        e=err(fit.x);pe=float(np.linalg.norm(e[12:15]));ae=float(np.linalg.norm(e[15:])/.25)
        foot=float(np.max(np.abs(e[:12])))
        mujoco.mj_forward(m,d)
        hits=g.collisions()
        margin=float(-np.max(g.hull.equations[:,:2]@d.subtree_com[bid,:2]+g.hull.equations[:,2]))
        ok=pe<.003 and ae<np.deg2rad(2) and foot<.002 and not hits and margin>=.02
        rec=dict(r=float(r),phi=float(phi),yaw=float(yaw),delta=list(delta),target=target.tolist(),
                 passed=ok,position_error_m=pe,orientation_error_deg=float(np.rad2deg(ae)),
                 foot_error_m=foot,com_margin_m=margin,hits=hits,qpos=d.qpos.tolist())
        score=pe+.25*ae+foot+(.5 if hits else 0)+max(0,.02-margin)
        if best is None or score<best[0]:best=(score,rec)
        if ok:
            cached=fit.x.copy()
            return rec
    return best[1]

if __name__=='__main__':
    import sys
    mode=sys.argv[1] if len(sys.argv)>1 else 'grid'
    jobs=[]
    if mode=='grid':
        for phi in np.linspace(-np.arcsin(.48/.65),np.arcsin(.5/.65),7):
            for r in [.35,.50,.65]:
                for yaw in [-.3,0,.3]:jobs.append((r,phi,yaw,[0,0,0]))
    elif mode=='random':
        rng=np.random.default_rng(20261115)
        for _ in range(256):jobs.append((rng.uniform(.35,.65),rng.uniform(-np.arcsin(.48/.65),np.arcsin(.5/.65)),rng.uniform(-.3,.3),rng.uniform(-.08,.08,3).tolist()))
    else:
        data=json.loads((O/'grid.json').read_text())
        for rec in data['records']:
            if not rec['passed']:continue
            # Stress all orientation corners at spherical outer/inner boundaries.
            if rec['r'] not in (.35,.65):continue
            for dr in [-.08,.08]:
                for dp in [-.08,.08]:
                    for dy in [-.08,.08]:jobs.append((rec['r'],rec['phi'],rec['yaw'],[dr,dp,dy]))
    for i,job in enumerate(jobs):
        rec=solve(*job);records.append(rec)
        print(i+1,len(jobs),rec['passed'],round(rec['position_error_m']*1000,2),round(rec['orientation_error_deg'],2),'time',round(time.time()-start,1),flush=True)
        (O/(mode+'.json')).write_text(json.dumps(dict(center=center.tolist(),urdf_sha256=hashlib.sha256(g.URDF.read_bytes()).hexdigest(),records=records),indent=2))
