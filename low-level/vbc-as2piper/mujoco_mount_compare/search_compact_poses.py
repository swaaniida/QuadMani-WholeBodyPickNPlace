#!/usr/bin/env python3
"""Search compact arm poses with a fixed dorsal wrist camera; no policy changes."""
import json
from pathlib import Path
import numpy as np
import mujoco
from run_locomani_demo import setup, camera_metrics
from run_verified_demo import build_controller, pd_control

HERE = Path(__file__).resolve().parent
OUT = HERE/'results/compact_pose_search'
MC = json.loads((HERE/'models/mesh_seated/mount_config.json').read_text())
LC = json.loads((HERE/'locomani_config.json').read_text())


def properties(m,d):
    ids=np.array([i for i in range(1,m.nbody) if m.body(i).name!='target_object'])
    mass=m.body_mass[ids]; com=(mass[:,None]*d.xipos[ids]).sum(0)/mass.sum()
    R=d.ximat[ids].reshape(-1,3,3)
    inert=np.einsum('nik,nk,njk->nij',R,m.body_inertia[ids],R)
    r=d.xipos[ids]-com
    tensors=inert+mass[:,None,None]*(np.sum(r*r,axis=1)[:,None,None]*np.eye(3)-r[:,:,None]*r[:,None,:])
    # Express the centroidal locked inertia in AS2 axes, not world axes.
    base=m.body('base_link').id;B=d.xmat[base].reshape(3,3)
    arm=np.array([m.body(int(i)).name.startswith('piper_') for i in ids])
    ac=(mass[arm,None]*d.xipos[ids[arm]]).sum(0)/mass[arm].sum()
    return dict(inertia_diag_kg_m2=np.diag(B.T@tensors.sum(0)@B).tolist(),
                arm_inertia_about_whole_com_diag_kg_m2=np.diag(B.T@tensors[arm].sum(0)@B).tolist(),
                com_relative_base_m=(B.T@(com-d.xpos[base])).tolist(),
                arm_com_relative_base_m=(B.T@(ac-d.xpos[base])).tolist())


def visibility(m,d):
    cam=m.camera('wrist_camera').id;origin=d.cam_xpos[cam]
    R=d.cam_xmat[cam].reshape(3,3);delta=d.xpos[m.body('target_object').id]-origin
    local=R.T@delta;depth=-local[2];half=np.deg2rad(m.cam_fovy[cam]/2)
    in_fov=depth>0 and abs(local[1])<depth*np.tan(half)*.9 and abs(local[0])<depth*np.tan(half)*16/9*.9
    if not in_fov:return False
    groups=np.array([1,0,1,0,0,0],dtype=np.uint8);hit=np.array([-1],dtype=np.int32)
    mujoco.mj_ray(m,d,origin,delta/np.linalg.norm(delta),groups,1,-1,hit)
    return int(hit[0])==m.geom('target_object_geom').id


def joints(m):
    return [int(m.joint('piper_joint%d'%i).qposadr) for i in range(1,7)]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    m=mujoco.MjModel.from_xml_path(str(HERE/'models/locomani/plate_only.xml'));d=mujoco.MjData(m)
    setup(m,d,MC,LC);initial=d.qpos.copy();qids=joints(m)
    rng=np.random.default_rng(31);pool=[]
    for k in range(35000):
        q=np.array([0,rng.uniform(.25,2.89),rng.uniform(-2.61,-.35),0,rng.uniform(-.97,.97),0])
        d.qpos[:]=initial;d.qpos[qids]=q;mujoco.mj_forward(m,d)
        if any(c.dist<0 and m.geom(int(c.geom1)).name!='ground' and m.geom(int(c.geom2)).name!='ground' for c in d.contact):continue
        if not visibility(m,d):continue
        p=properties(m,d)
        # Reject poses whose gravity load already exceeds the current motor model.
        dofs=[int(m.joint('piper_joint%d'%i).dofadr) for i in range(1,7)]
        if np.max(abs(d.qfrc_bias[dofs]))>8:continue
        if abs(p['com_relative_base_m'][0])>.06:continue
        pool.append((sum(p['inertia_diag_kg_m2']),q.tolist(),p))
    pool.sort(key=lambda x:x[0]);chosen=[]
    for score,q,p in pool:
        if all(np.linalg.norm(np.array(q)-np.array(x[1]))>.4 for x in chosen):chosen.append((score,q,p))
        if len(chosen)==8:break
    print('eligible',len(pool),'shortlist',[(round(x[0],4),np.round(x[1],3).tolist()) for x in chosen],flush=True)
    # Additional conservative middle-range folds to test the yaw/roll trade-off.
    chosen.extend((0,[0,q2,q3,0,q5,0],{}) for q2 in (.8,1.1,1.4)
                  for q3 in (-.55,-.7) for q5 in (0,.4))
    result=[]
    for variant in ('plate_only','profiles_plate'):
        m=mujoco.MjModel.from_xml_path(str(HERE/'models/locomani'/(variant+'.xml')))
        cases=[('baseline',LC['travel_arm_rad'])]+[('candidate_%d'%i,x[1]) for i,x in enumerate(chosen,1)]
        for name,q in cases:
            d=mujoco.MjData(m);cfg=dict(LC,travel_arm_rad=q);setup(m,d,MC,cfg)
            control=build_controller(m);targets={}
            for leg in ('FL','FR','RL','RR'):
                targets.update({leg+'_hip_joint':0,leg+'_thigh_joint':.8,leg+'_calf_joint':-1.6})
            targets.update({'piper_joint%d'%i:v for i,v in enumerate(q,1)})
            targets.update(piper_gripper_joint1=.049,piper_gripper_joint2=-.049)
            maxself=0;maxu=0;maxpitch=0
            while d.time<4:
                pd_control(m,d,targets,control);mujoco.mj_step(m,d)
                maxu=max(maxu,float(np.max(abs(d.ctrl[12:18]))))
                selfc=sum(m.geom(int(c.geom1)).name not in ('ground','target_object_geom') and m.geom(int(c.geom2)).name not in ('ground','target_object_geom') for c in d.contact)
                maxself=max(maxself,selfc)
            mujoco.mj_forward(m,d)
            item=dict(variant=variant,name=name,q_target_rad=q,q_actual_rad=d.qpos[joints(m)].tolist(),
                      max_self_contacts=maxself,max_arm_command_Nm=maxu,object_center_visible=visibility(m,d),
                      warnings=int(sum(w.number for w in d.warning)),**properties(m,d))
            item['camera']=camera_metrics(m,d)
            item['qpos']=d.qpos.tolist()
            result.append(item)
            print(variant,name,'I',np.round(item['inertia_diag_kg_m2'],3),'armz',round(item['arm_com_relative_base_m'][2],3),'visible',item['object_center_visible'],'self',maxself,flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(seed=31,samples=35000,eligible=len(pool),evaluations=result),indent=2))


if __name__=='__main__':main()
