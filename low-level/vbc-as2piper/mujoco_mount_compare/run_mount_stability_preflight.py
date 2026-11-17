#!/usr/bin/env python3
"""Preflight static-support/impulse response; NOT an official walking-controller test."""
import json
import os
from pathlib import Path
import math
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = HERE / 'models/mount_workspace'
OUT = HERE / 'results/mount_stability_preflight_20261109'
CASES = {
    'as2_only': MODELS / 'as2_only_actual_link_inertia.xml',
    'as2_piper_center': MODELS / 'as2_piper_center_actual_link_inertia.xml',
    'as2_piper_head_selected': MODELS / 'as2_piper_head_actual_link_inertia.xml',
}
DT = 0.002
SETTLE_S = 1.0
TOTAL_S = 4.0
PULSE_START = 1.0
PULSE_DURATION = 0.05
PULSE_FORCE_N = 30.0


def joint_kind(name):
    return 'arm' if name.startswith('piper_') else 'leg'


def initial_state(model):
    d = mujoco.MjData(model)
    d.qpos[:] = model.qpos0
    d.qpos[:7] = [0, 0, .35, 1, 0, 0, 0]
    for leg in ('FL', 'FR', 'RL', 'RR'):
        for suffix, value in (('hip', 0.), ('thigh', .8), ('calf', -1.6)):
            d.qpos[int(model.joint(leg + '_' + suffix + '_joint').qposadr)] = value
    for i, value in enumerate((0., .76, -.89, 0., .45, 0.), 1):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'piper_joint{}'.format(i))
        if j >= 0:
            d.qpos[model.jnt_qposadr[j]] = value
    for n, value in (('piper_gripper_joint1', .049), ('piper_gripper_joint2', -.049)):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j >= 0:
            d.qpos[model.jnt_qposadr[j]] = value
    mujoco.mj_forward(model, d)
    lows=[]
    for leg in ('FL','FR','RL','RR'):
        bid=int(model.body(leg+'_foot').id)
        for g in range(model.ngeom):
            if model.geom_bodyid[g]==bid and model.geom_contype[g] and model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE:
                lows.append(float(d.geom_xpos[g,2]-model.geom_size[g,0]))
    d.qpos[2] += .002 - min(lows)
    mujoco.mj_forward(model,d)
    return d


def controls(model, data):
    entries=[]
    for a in range(model.nu):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,a)
        jname=name[2:] if name and name.startswith('u_') else None
        if not jname: continue
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,jname)
        if jid<0: continue
        kind=joint_kind(jname); kp,kd=(90.,3.5) if kind=='leg' else (35.,1.5)
        entries.append((a,int(model.jnt_qposadr[jid]),int(model.jnt_dofadr[jid]),kp,kd))
    target=data.qpos.copy()
    return entries,target


def body_angles(data, bid):
    w,x,y,z=data.xquat[bid]
    roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch=math.asin(max(-1.,min(1.,2*(w*y-z*x))))
    return roll,pitch


def contact_feet(model,data):
    names=set()
    for c in data.contact:
        for g in (int(c.geom1),int(c.geom2)):
            b=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
            if b.endswith('_foot') and float(c.dist)<=0.001:
                names.add(b)
    return len(names)


def com(model,data):
    mass=model.body_mass[1:]
    return (mass[:,None]*data.xipos[1:]).sum(axis=0)/mass.sum()


def run_trial(name,path,direction):
    model=mujoco.MjModel.from_xml_path(str(path)); model.opt.timestep=DT
    template=initial_state(model); entries,target=controls(model,template)
    start_root=template.qpos[:3].copy(); start_com=com(model,template)
    d=template
    steps=int(TOTAL_S/DT); pulse0=int(PULSE_START/DT); pulse1=int((PULSE_START+PULSE_DURATION)/DT)
    bid=int(model.body('base_link').id); base0=d.xpos[bid].copy()
    max_roll=max_pitch=0.; min_height=float('inf'); min_feet=4; sat=0; total=0
    contact_proxy=[]; angle_series=[]; base_xyz=[]; com_xyz=[]; times=[]
    for k in range(steps):
        t=k*DT
        d.xfrc_applied[:]=0
        if direction is not None and pulse0<=k<pulse1:
            d.xfrc_applied[bid,:3]=np.asarray(direction,dtype=float)*PULSE_FORCE_N
        for a,qp,qv,kp,kd in entries:
            u=kp*(target[qp]-d.qpos[qp])-kd*d.qvel[qv]
            lo,hi=model.actuator_ctrlrange[a]
            clipped=max(float(lo),min(float(hi),float(u)))
            sat += int(abs(clipped-u)>1e-6); total += 1
            d.ctrl[a]=clipped
        mujoco.mj_step(model,d)
        r,p=body_angles(d,bid); max_roll=max(max_roll,abs(math.degrees(r))); max_pitch=max(max_pitch,abs(math.degrees(p)))
        min_height=min(min_height,float(d.xpos[bid,2])); feet=contact_feet(model,d)
        # The initial pose is intentionally 2 mm above the plane to avoid initial mesh penetration.
        # Ignore this contact-free settling interval in the support-contact minimum.
        if t >= 0.1: min_feet=min(min_feet,feet)
        if k%10==0:
            times.append(t);angle_series.append([math.degrees(r),math.degrees(p)]);base_xyz.append(d.xpos[bid].copy());com_xyz.append(com(model,d));contact_proxy.append(feet)
    final_angles=angle_series[-1]
    final_base=np.asarray(base_xyz[-1]); final_com=np.asarray(com_xyz[-1])
    impulse=(PULSE_FORCE_N*PULSE_DURATION) if direction is not None else 0.
    fall=bool(min_height < .16 or max_roll>35 or max_pitch>35)
    OUT.mkdir(parents=True,exist_ok=True)
    stem=name+'_'+('no_impulse' if direction is None else ('x+' if direction[0]>0 else 'y+' if direction[1]>0 else 'y-' if direction[1]<0 else 'x-'))
    np.savez_compressed(OUT/(stem+'_trace.npz'),time_s=np.asarray(times),roll_pitch_deg=np.asarray(angle_series),base_xyz_m=np.asarray(base_xyz),com_xyz_m=np.asarray(com_xyz),foot_contact_count=np.asarray(contact_proxy))
    return {
        'variant':name,'model':str(path),'direction_world':direction,'impulse_Ns':impulse,'duration_s':TOTAL_S,
        'initial_base_xyz_m':start_root.tolist(),'initial_com_xyz_m':start_com.tolist(),
        'max_abs_roll_deg':max_roll,'max_abs_pitch_deg':max_pitch,'minimum_base_z_m':min_height,
        'minimum_feet_with_contact':min_feet,'final_roll_pitch_deg':final_angles,
        'final_base_displacement_m':float(np.linalg.norm(final_base-base0)),
        'final_com_displacement_m':float(np.linalg.norm(final_com-start_com)),
        'actuator_saturation_fraction':float(sat/max(1,total)),'fall_by_screening_threshold':fall,
        'controller':'joint-position PD hold for preflight only; no walking gait or velocity tracking',
    }


def main():
    os.umask(0o002)
    tests=[('no_impulse',None),('forward',[1,0,0]),('backward',[-1,0,0]),('left',[0,1,0]),('right',[0,-1,0])]
    results=[]
    for name,path in CASES.items():
        for label,direction in tests:
            r=run_trial(name,path,direction);r['condition']=label;results.append(r)
            print(name,label,'impulse',r['impulse_Ns'],'max RP',round(r['max_abs_roll_deg'],2),round(r['max_abs_pitch_deg'],2),'feet min',r['minimum_feet_with_contact'],'fall',r['fall_by_screening_threshold'],flush=True)
    summary={'title':'Static stance and horizontal impulse preflight','warning':'This is not a walking stability test and not the official AS2 controller. It checks model integrity and relative standing response only. Do not use it to certify walking.',
             'conditions':{'all_models_same':True,'arm_pose_rad':[0,.76,-.89,0,.45,0],'leg_pose_rad_per_leg':[0,.8,-1.6],'impulse_force_N':PULSE_FORCE_N,'impulse_duration_s':PULSE_DURATION,'impulse_Ns':PULSE_FORCE_N*PULSE_DURATION,'directions':['+X','-X','+Y','-Y'],'timestep_s':DT,'total_s':TOTAL_S,'minimum_contact_count_excludes_initial_settling_s':0.1},'results':results}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Standing/impulse preflight','',summary['warning'],'','30 N × 50 ms = 1.5 N·s horizontal impulse; identical initial joint pose and PD hold on all three models. These are preliminary static support tests, not walking.','','| Variant | Condition | Max roll (deg) | Max pitch (deg) | Min feet contact | Fall screen |','|---|---|---:|---:|---:|---:|']
    for r in results:lines.append('| {} | {} | {:.2f} | {:.2f} | {} | {} |'.format(r['variant'],r['condition'],r['max_abs_roll_deg'],r['max_abs_pitch_deg'],r['minimum_feet_with_contact'],'FAIL' if r['fall_by_screening_threshold'] else 'no'))
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
