#!/usr/bin/env python3
"""Plot the saved CPU study and render its actual MuJoCo configurations."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import mujoco
from mount_workspace_study import OUT,setup

LABELS=['Plate: current','Plate: +25 mm','Profiles: current','Profiles: +25 mm',
        'Profiles: extended plate (+74 mm)','B1 + Z1 reference']
COLORS=['#3878b8','#15456f','#e6a23b','#b56a10','#32955d','#ae5596']


def main():
    cases=json.loads((OUT/'results.json').read_text())
    fig=plt.figure(figsize=(15,9))
    for index,(c,label,color) in enumerate(zip(cases,LABELS,COLORS)):
        ax=fig.add_subplot(2,3,index+1,projection='3d')
        p=np.load(str(OUT/(c['name']+'_workspace.npz')))['head'][::4]
        ax.scatter(p[:,0],p[:,1],p[:,2],s=.7,alpha=.2,color=color)
        ax.scatter([0],[0],[0],marker='+',color='red',s=50)
        ax.set_title(label,fontsize=10);ax.set_xlim(-1.1,.85);ax.set_ylim(-1,1);ax.set_zlim(-.6,1)
        ax.set_xlabel('Head x [m]');ax.set_ylabel('Head y [m]');ax.set_zlabel('Head z [m]')
        ax.view_init(elev=20,azim=-55)
    fig.suptitle('3D TCP workspace in standing head coordinates; actual collision proxies; empty open gripper')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(str(OUT/'workspace_3d.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
    for ax,c,label,color in zip(axes.flat,cases,LABELS,COLORS):
        p=np.load(str(OUT/(c['name']+'_workspace.npz')))['head']
        ax.scatter(p[:,0],p[:,2],s=.6,alpha=.15,color=color,rasterized=True)
        ax.axvline(0,color='black',lw=.8);floor=-(c['standing_qpos'][2]+c['head_origin_in_base_m'][2])
        ax.axhline(floor,color='black',ls='--',lw=.8,label='Standing ground')
        ax.scatter([0],[0],s=35,marker='+',color='red',label='Head proxy')
        ax.set_title(label);ax.set_xlim(-1.1,.85);ax.set_ylim(-.6,1.)
        ax.set_xlabel('TCP x ahead of standing head [m]');ax.set_ylabel('TCP z relative to head [m]')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Head-relative workspace: 16,000 joint samples per model; self-collision filtered\nSide projection over all lateral positions; not a filled reachable region',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(str(OUT/'workspace_side.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
    for ax,c,label,color in zip(axes.flat,cases,LABELS,COLORS):
        p=np.load(str(OUT/(c['name']+'_workspace.npz')))['head']
        ax.scatter(p[:,0],p[:,1],s=.6,alpha=.15,color=color,rasterized=True)
        ax.axvline(0,color='black',lw=.8);ax.scatter([0],[0],marker='+',color='red')
        ax.set_title(label);ax.set_xlim(-1.1,.85);ax.set_ylim(-1.,1.);ax.grid(alpha=.2)
        ax.set_xlabel('TCP x ahead of head [m]');ax.set_ylabel('TCP y from head [m]')
    fig.suptitle('Head-relative workspace: top projection over all heights',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(str(OUT/'workspace_top.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    labels=['Plate\ncurrent','Plate\nforward','Profiles\ncurrent','Profiles\nforward','Profiles\nextended']
    for ax,key,index,unit in [(axes[0],'com_world_m',0,'CoM x [mm]'),(axes[1],'com_world_m',2,'CoM height [mm]'),(axes[2],'inertia_diag_kg_m2',1,'Pitch inertia about CoM [kg m²]')]:
        y=[c['standing'][key][index]*(1000 if key=='com_world_m' else 1) for c in cases[:5]]
        ax.bar(labels,y,color=COLORS[:5]);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',labelsize=8)
    fig.suptitle('Same AS2 C arm pose and grounded stance; no payload/camera mass')
    fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(str(OUT/'mount_tradeoff.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,key,title in [(axes[0],'ground_reach_kinematic','Kinematic + collision + support screening'),(axes[1],'ground_reach','Additional illustrative torque screening*')]:
        for i,(c,label,color) in enumerate(zip(cases,LABELS,COLORS)):
            y=[c[key].get(str(x),{}).get('angle_deg',np.nan) for x in (.1,.2,.3)]
            ax.plot([.1,.2,.3],y,'o-',label=label,color=color)
            for x,v in zip((.1,.2,.3),y):
                if np.isnan(v):ax.scatter([x+(i-2.5)*.002],[43],marker='x',color=color,s=20)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Target ahead of standing head [m]');ax.set_ylim(-1,46);ax.set_xticks([.1,.2,.3]);ax.grid(alpha=.25)
    axes[0].set_ylabel('Smallest successful sampled forward body tilt [deg]')
    axes[0].legend(fontsize=8,loc='upper left')
    fig.text(.5,.015,'x at 43° = no solution found through 40°; angle step 2.5°. *AS2 <9 Nm/joint (assumption), B1 <80% URDF effort.\nFour feet fixed; rear hip pivot; empty open gripper down-facing. Search results are not global minima or dynamic validation.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.12,1,1]);fig.savefig(str(OUT/'ground_reach.png'),dpi=150);plt.close(fig)
    # Actual meshes and calculated joint configurations; no image generation.
    canvas=Image.new('RGB',(1280,6*390),'white');draw=ImageDraw.Draw(canvas)
    validation=[];frame_configs=[]
    for row,(c,label) in enumerate(zip(cases,LABELS)):
        m=mujoco.MjModel.from_xml_path(c['model']);d,names,feet,radii=setup(m,c['name'])
        m.vis.headlight.ambient[:]=[.6,.6,.6]
        m.vis.headlight.diffuse[:]=[.8,.8,.8]
        rend=mujoco.Renderer(m,height=360,width=640)
        cam=mujoco.MjvCamera();cam.azimuth=120;cam.elevation=-18
        cam.distance=2.3 if c['name']=='b1z1' else 1.6;cam.lookat[:]=[.15,0,.28]
        scenes=[('Standing C' if c['name']!='b1z1' else 'Standing original',c['standing_qpos'])]
        target=c['ground_reach_kinematic'].get('0.1')
        scenes.append(('Head +100 mm: %.1f deg (kinematic)'%target['angle_deg'],target['qpos']) if target else ('No feasible pose found',c['standing_qpos']))
        for col,(caption,q) in enumerate(scenes):
            d.qpos[:]=q;mujoco.mj_forward(m,d)
            rend.update_scene(d,camera=cam);frame=Image.fromarray(rend.render().copy())
            file=OUT/(c['name']+('_standing.png' if col==0 else '_reach.png'));frame.save(str(file))
            canvas.paste(frame,(col*640,row*390+30));draw.text((col*640+10,row*390+8),label+' | '+caption,fill='black')
        rend.close()
        # Frame proposal is a sidecar only; training files are not touched.
        d.qpos[:]=c['standing_qpos'];mujoco.mj_forward(m,d)
        base=m.body('link00' if c['name']=='b1z1' else 'piper_base_link').id
        tcp=m.body('ee_gripper_link' if c['name']=='b1z1' else 'piper_umi_nominal_tcp').id
        root=m.body('base' if c['name']=='b1z1' else 'base_link').id
        T=np.eye(4);T[:3,:3]=d.xmat[base].reshape(3,3);T[:3,3]=d.xpos[base]-d.xpos[root]
        local=np.load(str(OUT/(c['name']+'_workspace.npz')))['arm']
        frame_configs.append(dict(name=c['name'],command_frame=c['workspace']['arm_frame'],
            T_body_arm_at_standing=T.tolist(),initial_tcp_arm_m=(T[:3,:3].T@(d.xpos[tcp]-d.xpos[base])).tolist(),
            sampled_tcp_arm_bounds_m=np.stack([local.min(0),local.max(0)]).tolist(),
            warning='Bounds are not a collision-free command box. Use reachable joint samples or IK-filtered goals. Transform full body roll/pitch/yaw.'))
        validation.append(dict(name=c['name'],nq=m.nq,nv=m.nv,nu=m.nu,joint_count=m.njnt-1))
    canvas.save(str(OUT/'robot_pose_comparison.jpg'),quality=93)
    (OUT/'model_validation.json').write_text(json.dumps(validation,indent=2))
    (OUT/'arm_frame_proposal.json').write_text(json.dumps(frame_configs,indent=2))
    print('Saved workspace, tradeoff, reach plots, mesh renders and frame proposal in',OUT)


if __name__=='__main__':main()
