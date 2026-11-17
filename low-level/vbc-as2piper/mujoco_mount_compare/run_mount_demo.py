#!/usr/bin/env python3
"""Gravity/contact/PD demonstration of two AS2+PiPER mounting configurations.

Not a trained locomotion controller. For high fidelity gait comparison,
replace the demonstration PD targets with an identical learned WBC policy.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent

try:
    import mujoco
except ImportError:
    raise SystemExit('MuJoCo not installed. In AstraGo Python 3.8 venv: python -m pip install "mujoco==3.1.6" "numpy<1.25"')


def joint_type(name):
    if name.startswith('piper_gripper_'): return 'gripper'
    if name.startswith('piper_'): return 'arm'
    return 'leg'


def safe_joint_pose(model,data,name,val):
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
    if jid<0:raise RuntimeError('Missing actuator joint '+name)
    lo,hi=model.jnt_range[jid]
    if model.jnt_limited[jid] and not (float(lo)-1e-5<=val<=float(hi)+1e-5):
        raise ValueError('{} pose {} outside joint limits [{},{}]'.format(name,val,lo,hi))
    data.qpos[model.jnt_qposadr[jid]]=val


def setup(model,data,config):
    data.qpos[:]=model.qpos0
    # Natural gravity and four-foot contact; initial height is adjusted to match foot sphere bottoms.
    for leg in ('FL','FR','RL','RR'):
        safe_joint_pose(model,data,leg+'_hip_joint',0)
        safe_joint_pose(model,data,leg+'_thigh_joint',float(config['initial_thigh_rad']))
        safe_joint_pose(model,data,leg+'_calf_joint',float(config['initial_calf_rad']))
    for k,val in enumerate(config['initial_piper_joint_rad'],1):
        safe_joint_pose(model,data,'piper_joint{}'.format(k),float(val))
    op=float(config['initial_gripper_open_m'])
    for name,target in (('piper_gripper_joint1',op),('piper_gripper_joint2',-op)):
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if jid>=0:
            lo,hi=model.jnt_range[jid]
            safe_joint_pose(model,data,name,min(max(target,float(lo)+1e-6),float(hi)-1e-6))
    data.qpos[:3]=[0,0,0.35]
    data.qpos[3:7]=[1,0,0,0]
    mujoco.mj_forward(model,data)
    lowest=[]
    for k in ('FL_foot','FR_foot','RL_foot','RR_foot'):
        bodyid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,k)
        if bodyid<0:raise RuntimeError('Cannot find foot body '+k)
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid]==bodyid and model.geom_contype[gid]:
                if int(model.geom_type[gid])==int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    lowest.append(float(data.geom_xpos[gid,2])-float(model.geom_size[gid,0]))
    if len(lowest)<4:
        raise RuntimeError('Expected collidable sphere geom on each foot, got {}'.format(len(lowest)))
    data.qpos[2]+=0.002-min(lowest)
    mujoco.mj_forward(model,data)


def nominal_targets(config,elapsed,mode):
    import math
    t=float(elapsed)
    targets={}
    def smooth(z):
        z=max(0.0,min(1.0,z))
        return z*z*(3-2*z)
    crotch=smooth((t-1.0)/2.5) if mode=='crouch' else 0.0
    for leg in ('FL','FR','RL','RR'):
        targets[leg+'_hip_joint']=0.0
        targets[leg+'_thigh_joint']=float(config['initial_thigh_rad'])+0.22*crotch
        targets[leg+'_calf_joint']=float(config['initial_calf_rad'])-0.44*crotch
        if mode=='step':
            # Demonstration of alternating joint targets, NOT a balance-controlled gait.
            phase=0 if leg in ('FL','RR') else math.pi
            blend=smooth((t-1.0)/1.0)
            cyc=2*math.pi*0.65*t+phase
            targets[leg+'_thigh_joint']+=0.08*math.sin(cyc)*blend
            targets[leg+'_calf_joint']+=-0.14*max(0,math.sin(cyc))*blend
    for i,val in enumerate(config['initial_piper_joint_rad'],1):
        targets['piper_joint{}'.format(i)]=float(val)
    if mode=='arm':
        blend=smooth((t-1.0)/1.0)
        targets['piper_joint1']+=0.35*math.sin(t*1.0)*blend
        targets['piper_joint4']+=0.2*math.sin(t*0.75)*blend
    opening=float(config['initial_gripper_open_m'])
    targets['piper_gripper_joint1']=opening
    targets['piper_gripper_joint2']=-opening
    return targets


def pd_control(model,data,targets,control_map):
    total_torque=0.0
    total_gripper_force=0.0
    for name,(act,qp,qv,limits,gains) in control_map.items():
        desired=targets[name]
        desired=min(max(desired,limits[0]+1e-6),limits[1]-1e-6)
        kp,kd=gains
        u=kp*(desired-float(data.qpos[qp]))-kd*float(data.qvel[qv])
        lower,upper=model.actuator_ctrlrange[act]
        u=max(float(lower),min(float(upper),u))
        data.ctrl[act]=u
        if joint_type(name)=='gripper': total_gripper_force+=abs(u)
        else: total_torque+=abs(u)
    return (total_torque,total_gripper_force)


def build_controller(model):
    out={}
    for a in range(model.nu):
        act_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,a)
        if not act_name or not act_name.startswith('u_'):
            raise RuntimeError('Unexpected actuator {}'.format(act_name))
        jname=act_name[2:]
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,jname)
        if jid<0:raise RuntimeError('Motor {} missing joint'.format(act_name))
        kind=joint_type(jname)
        gains={'leg':(90.,3.5),'arm':(35.,1.5),'gripper':(1800.,35.)}[kind]
        out[jname]=(a,int(model.jnt_qposadr[jid]),int(model.jnt_dofadr[jid]),tuple(model.jnt_range[jid]),gains)
    if len(out)!=20:raise RuntimeError('Expected 20 motors; found {}'.format(len(out)))
    return out


def pose_euler(q):
    # wxyz orientation -> roll, pitch, yaw in radians
    w,x,y,z=[float(v) for v in q]
    roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    sinp=max(-1.,min(1.,2*(w*y-z*x)))
    pitch=math.asin(sinp)
    yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    return roll,pitch,yaw


def capture(model,data,effort):
    rootid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'base_link')
    r,p,y=pose_euler(data.xquat[rootid])
    mass=0;com=[0,0,0]
    # Aggregate link inertias, excluding the static ground body (body 0).
    for i in range(1,model.nbody):
        m=float(model.body_mass[i]);mass+=m
        for axis in range(3):com[axis]+=m*float(data.xipos[i,axis])
    if mass>0:com=[x/mass for x in com]
    return {'time_s':float(data.time),'base_z_m':float(data.xpos[rootid,2]),
            'roll_deg':math.degrees(r),'pitch_deg':math.degrees(p),
            'com_x_m':com[0],'com_y_m':com[1],'com_z_m':com[2],
            'contacts':int(data.ncon),'sum_abs_joint_torque_Nm':effort[0],
            'sum_abs_gripper_force_N':effort[1]}


def run(args):
    cfg=json.loads((HERE/'mount_config.json').read_text(encoding='utf-8'))
    modelpath=HERE/'models'/(args.variant+'.xml')
    if not modelpath.is_file():
        raise SystemExit('Generate XML first: python mujoco_mount_compare/build_mount_scenes.py')
    m=mujoco.MjModel.from_xml_path(str(modelpath))
    d=mujoco.MjData(m)
    if int(m.nq)!=27 or int(m.nv)!=26 or int(m.nu)!=20:
        raise RuntimeError('Invalid free-base 20-DOF robot: nq={}, nv={}, nu={}. Verify the MJCF.'.format(m.nq,m.nv,m.nu))
    setup(m,d,cfg)
    controller=build_controller(m)
    print('Model: {} / gravity: {} / body mass: {:.3f}kg / nq:{} nv:{} motors:{}'.format(
         args.variant,m.opt.gravity,list(m.body_mass[1:]),m.nq,m.nv,m.nu)) if args.verbose else None
    print('Running: {} / {} / duration {}s / gui {}'.format(args.variant,args.mode,args.seconds,args.gui))
    out=HERE/'results';out.mkdir(parents=True,exist_ok=True)
    outfile=out/'{}_{}.csv'.format(args.variant,args.mode)
    dt=float(m.opt.timestep)
    render=None
    recorder=None
    renderer=None
    camera=None
    if args.video:
        try:
            import imageio.v2 as imageio
            renderer=mujoco.Renderer(m,height=360,width=640)
            camera=mujoco.MjvCamera()
            camera.lookat[:]=[0.05,0,0.22]
            camera.distance=1.65
            camera.azimuth=145
            camera.elevation=-17
            videopath=HERE/'results'/'{}_{}.mp4'.format(args.variant,args.mode)
            videopath.parent.mkdir(parents=True,exist_ok=True)
            recorder=imageio.get_writer(str(videopath),fps=24,codec='libx264',quality=8)
        except Exception as exc:
            raise RuntimeError('Video rendering needs an EGL or OSMesa GL context and imageio[ffmpeg]. Set MUJOCO_GL=egl before invoking; details: {}'.format(exc))
    if args.gui:
        try:
            from mujoco import viewer
            render=viewer.launch_passive(m,d)
        except Exception as exc:
            raise RuntimeError('Interactive viewer needs desktop OpenGL; use --no-gui on SSH, or run locally. {}'.format(exc))
    stride=max(1,int(0.025/dt))
    start=time.monotonic()
    with outfile.open('w',newline='',encoding='utf-8') as handle:
        writer=None
        step=0
        while d.time<args.seconds and (render is None or render.is_running()):
            targets=nominal_targets(cfg,d.time,args.mode)
            cmd=pd_control(m,d,targets,controller)
            mujoco.mj_step(m,d)
            if step%stride==0:
                row=capture(m,d,cmd)
                if writer is None:
                    writer=csv.DictWriter(handle,fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if not all(math.isfinite(float(v)) for v in row.values()):
                    raise RuntimeError('Dynamics diverged; check collision/inertia/initial posture!')
            if renderer is not None and step%max(1,int((1/24.)/dt))==0:
                renderer.update_scene(d,camera=camera)
                recorder.append_data(renderer.render())
            if render is not None and step%5==0:
                render.sync()
                wait=d.time-(time.monotonic()-start)
                if wait>0:time.sleep(min(.02,wait))
            step+=1
    if render is not None:render.close()
    if recorder is not None:
        recorder.close()
        renderer.close()
        print('Saved video:',videopath)
    print('Saved:',outfile)
    print('NOTE: stand/crouch/arm = PD target demonstrations; step = scripted motion, NOT validated walking.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['profiles_plate','plate_only'],required=True)
    p.add_argument('--mode',choices=['stand','crouch','arm','step'],default='stand')
    p.add_argument('--seconds',type=float,default=8.0)
    p.add_argument('--gui',action='store_true',help='Interactive MuJoCo window (local desktop only)')
    p.add_argument('--video',action='store_true',help='Offscreen MP4 (requires MUJOCO_GL=egl or osmesa and imageio[ffmpeg])')
    p.add_argument('--verbose',action='store_true')
    args=p.parse_args()
    run(args)

if __name__=='__main__':main()
