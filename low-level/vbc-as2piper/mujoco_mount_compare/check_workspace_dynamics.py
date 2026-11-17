#!/usr/bin/env python3
"""Three-second gravity/contact standing checks. This is not a learned policy."""
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from mount_workspace_study import OUT,URDFS,setup,mass_properties
from verify_mounts import stl_triangles,surface


def main():
    results=[];interfaces=[]
    for case in json.loads((OUT/'results.json').read_text()):
        m=mujoco.MjModel.from_xml_path(case['model']);d,arm,feet,_=setup(m,case['name'])
        d.qpos[2]+=.002;target=d.qpos.copy();b1=case['name']=='b1z1'
        if not b1:
            basez=d.xpos[m.body('base_link').id,2]
            gid=m.geom('c_mount_plate_0').id;center=d.geom_xpos[gid]-[0,0,basez];half=m.geom_size[gid]
            platebottom=center[2]-half[2];platetop=center[2]+half[2]
            rect=[center[0]-half[0],center[0]+half[0],center[1]-half[1],center[1]+half[1]]
            shell=stl_triangles(URDFS/'../meshes/as2/base_link.STL')
            report=dict(name=case['name'],plate_as2=surface(shell,rect,platebottom),
                plate_to_piper_gap_mm=1000*(d.xpos[m.body('piper_base_link').id,2]-basez-platetop),
                structural_fastening_verified=False)
            if case['variant']=='profiles_plate':
                supports=[]
                for side in ('left','right'):
                    p=m.geom('c_al_profile_'+side+'_0').id;pc=d.geom_xpos[p]-[0,0,basez];ph=m.geom_size[p]
                    supports.append(surface(shell,[pc[0]-ph[0],pc[0]+ph[0],pc[1]-ph[1],pc[1]+ph[1]],pc[2]-ph[2]))
                    assert abs(pc[2]+ph[2]-platebottom)<1e-8
                report['profile_as2']=supports
            ref=ET.parse(URDFS/('as2piper_umi_v7_mesh_seated_'+case['variant']+'.urdf')).getroot()
            candidate=ET.parse(case['urdf']).getroot()
            preserved=[e for e in ref if 'gripper' in e.get('name','') or 'umi' in e.get('name','')]
            assert all(ET.tostring(e)==ET.tostring(candidate.find(e.tag+"[@name='"+e.get('name')+"']")) for e in preserved)
            report['umi_links_joints_unchanged']=True
            assert abs(report['plate_to_piper_gap_mm'])<1e-6
            assert report['plate_as2']['mesh_penetration_mm']<.01
            interfaces.append(report)
        effort={j.get('name'):float(j.find('limit').get('effort')) for j in ET.parse(case['urdf']).getroot().findall('joint') if j.find('limit') is not None}
        address=[]
        for a in range(m.nu):
            j=int(m.actuator_trnid[a,0]);n=m.joint(j).name
            isarm=n in arm;finger=('gripper' in n.lower())
            kp,kd=(500,5) if finger else ((40,3) if isarm else (100,4))
            cap=.8*effort[n] if b1 else (35 if finger else (9 if isarm else (30 if 'hip' in n else 45)))
            m.actuator_ctrlrange[a]=[-cap,cap]
            address.append((a,int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j]),kp,kd,cap))
        logs=[];peak=0.
        for step in range(round(3/m.opt.timestep)):
            mujoco.mj_forward(m,d)
            # Gravity feed-forward on actuated joints only. No base forces.
            for a,q,v,kp,kd,cap in address:
                d.ctrl[a]=np.clip(kp*(target[q]-d.qpos[q])-kd*d.qvel[v]+d.qfrc_bias[v],-cap,cap)
            peak=max(peak,float(np.max(np.abs(d.ctrl))))
            mujoco.mj_step(m,d)
            if step%25==0:
                mujoco.mj_forward(m,d)
                rpy=Rotation.from_quat(d.qpos[[4,5,6,3]]).as_euler('xyz')
                forces=np.zeros(4)
                for k in range(d.ncon):
                    c=d.contact[k];w=np.zeros(6);mujoco.mj_contactForce(m,d,k,w)
                    for i,f in enumerate(feet):
                        if (m.geom_bodyid[c.geom1]==f and c.geom2==m.geom('ground').id) or (m.geom_bodyid[c.geom2]==f and c.geom1==m.geom('ground').id):forces[i]+=max(0,w[0])
                logs.append([d.time,d.qpos[2],rpy[0],rpy[1],*forces])
        data=np.array(logs);late=data[data[:,0]>2]
        entry=dict(name=case['name'],duration_s=float(d.time),controller='Joint PD + actuated gravity feedforward; no base forces; not trained locomotion',
            upright=bool(np.isfinite(d.qpos).all() and np.max(np.abs(data[:,2:4]))<.25 and d.qpos[2]>.15),
            max_abs_roll_pitch_deg=np.rad2deg(np.max(np.abs(data[:,2:4]),axis=0)).tolist(),
            final_base_z_m=float(d.qpos[2]),late_mean_foot_normal_N=late[:,4:].mean(0).tolist(),
            late_min_foot_normal_N=late[:,4:].min(0).tolist(),max_abs_control_Nm_or_N=peak,
            final_mass_properties=mass_properties(m,d))
        results.append(entry);np.savetxt(str(OUT/(case['name']+'_standing_dynamics.csv')),data,delimiter=',',header='time_s,base_z_m,roll_rad,pitch_rad,FL_normal_N,FR_normal_N,RL_normal_N,RR_normal_N',comments='')
        print(case['name'],entry['upright'],entry['max_abs_roll_pitch_deg'],flush=True)
    (OUT/'standing_dynamics.json').write_text(json.dumps(results,indent=2))
    (OUT/'interface_validation.json').write_text(json.dumps(interfaces,indent=2))


if __name__=='__main__':main()
