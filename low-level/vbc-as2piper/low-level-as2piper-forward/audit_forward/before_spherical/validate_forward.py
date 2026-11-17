#!/usr/bin/env python3
"""Independent dense path, FK, mount geometry and source-preservation checks."""
import hashlib,importlib.util,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mujoco_mount_compare'))
from mount_workspace_study import setup,self_collision
from verify_mounts import stl_triangles,surface
spec=importlib.util.spec_from_file_location('forward_fk',ROOT/'legged_gym/utils/urdf_chain.py')
fk_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fk_module)


def main():
    audit=ROOT/'audit_forward';audit.mkdir(exist_ok=True)
    source=ROOT.parent/'low-level-as2piper'
    hashes=json.loads((audit/'source_before.json').read_text())
    changed=[str(p) for p,h in hashes.items() if hashlib.sha256((source/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    asset=ROOT/'resources/robots/as2piper/urdf/as2piper_plate_forward_training.urdf'
    b=np.load(str(ROOT/'resources/commands/arm_commands.npz'));meta=json.loads((ROOT/'resources/commands/arm_commands.json').read_text())
    fk=fk_module.URDFChain(asset,'piper_base_link','piper_umi_nominal_tcp',meta['joint_names'])
    m=mujoco.MjModel.from_xml_path(str(ROOT/'resources/commands/plate_forward.xml'));d,names,feet,radii=setup(m,'plate_only_forward')
    ids=[int(m.joint(n).qposadr) for n in names];neutral=d.qpos.copy();base=m.body('piper_base_link').id;tcp=m.body('piper_umi_nominal_tcp').id
    paths=b['home'][None,None,:]+(b['joints'][:,None,:]-b['home'][None,None,:])*np.linspace(0,1,129)[None,:,None]
    q=torch.tensor(paths.reshape(-1,6),dtype=torch.float)
    with torch.no_grad():p,rotation=fk(q)
    maxerr=0.;maxrot=0.;collisions=0;peak=0.
    for i,x in enumerate(q.numpy()):
        d.qpos[:]=neutral;d.qpos[ids]=x;mujoco.mj_forward(m,d)
        actual=d.xpos[tcp]-d.xpos[base]
        maxerr=max(maxerr,float(np.linalg.norm(actual-p[i].numpy())))
        maxrot=max(maxrot,float(1-abs(np.dot(d.xquat[tcp][[1,2,3,0]],rotation[i].numpy()))))
        collisions+=int(self_collision(m,d))
        peak=max(peak,float(np.max(np.abs(d.qfrc_bias[[int(m.joint(n).dofadr) for n in names]]))))
    assert maxerr<1e-5 and maxrot<1e-5 and collisions==0,(maxerr,maxrot,collisions)
    assert peak<9.
    d.qpos[:]=neutral;mujoco.mj_forward(m,d)
    plate=m.geom('c_mount_plate_0').id;half=m.geom_size[plate];pc=d.geom_xpos[plate]-neutral[:3]
    as2=stl_triangles(ROOT/'resources/robots/as2piper/meshes/as2/base_link.STL')
    geometry=surface(as2,[pc[0]-half[0],pc[0]+half[0],pc[1]-half[1],pc[1]+half[1]],pc[2]-half[2])
    assert geometry['mesh_penetration_mm']<.01
    assert abs(d.xpos[base,0]-.005)<1e-8
    assert abs(d.xpos[base,2]-(d.geom_xpos[plate,2]+half[2]))<1e-8
    before=ET.parse(source/'resources/robots/as2piper/urdf/as2piper_umi_v7_mesh_seated_plate_only.urdf').getroot()
    after=ET.parse(asset).getroot()
    for name in ['mount_plate','piper_gripper_base','piper_gripper_link1','piper_gripper_link2']:
        assert ET.tostring(before.find("link[@name='"+name+"']"))==ET.tostring(after.find("link[@name='"+name+"']"))
    reward='legged_gym/envs/rewards/maniploco_rewards.py'
    assert (ROOT/reward).read_bytes()==(source/reward).read_bytes()
    assert all(p.read_bytes()==(source/'rsl_rl_local'/p.relative_to(ROOT/'rsl_rl_local')).read_bytes() for p in (ROOT/'rsl_rl_local').rglob('*.py'))
    result=dict(source_files_unchanged=len(hashes),reward_and_ppo_identical=True,plate_and_umi_links_identical=True,
        model=dict(nq=m.nq,nv=m.nv,nu=m.nu),paths=len(b['joints']),samples_per_path=129,
        self_collision_count=collisions,max_fk_position_error_m=maxerr,max_quaternion_dot_error=maxrot,
        max_arm_gravity_torque_Nm=peak,plate_as2_geometry=geometry,
        plate_piper_gap_m=float(d.xpos[base,2]-(d.geom_xpos[plate,2]+half[2])),
        active_goal_endpoint_bounds_arm_m=meta['tcp_endpoint_bounds_arm_m'])
    (audit/'validation.json').write_text(json.dumps(result,indent=2))
    fig=plt.figure(figsize=(12,5));ax=fig.add_subplot(121,projection='3d');v=p.numpy()[::8]
    ax.scatter(v[:,0],v[:,1],v[:,2],s=1,alpha=.15);ax.scatter(*fk(torch.tensor(b['home'][None,:],dtype=torch.float))[0][0].numpy(),color='red',label='Reset/home')
    ax.set_xlabel('Arm-base x [m]');ax.set_ylabel('Arm-base y [m]');ax.set_zlabel('Arm-base z [m]');ax.legend();ax.set_title('Coupled FK command paths')
    ax=fig.add_subplot(122);ax.scatter(v[:,0]+.005-.3347053,v[:,2]+.0820285+.3244069,s=1,alpha=.15)
    ax.axvline(0,color='red',ls='--',label='Head x');ax.axhline(0,color='black');ax.set_xlabel('Ahead of head [m]');ax.set_ylabel('TCP world height in neutral stance [m]');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Plate unchanged; arm base x=+5 mm; 256 home-to-endpoint paths (129 samples/path)')
    fig.tight_layout();fig.savefig(str(audit/'command_workspace.png'),dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
