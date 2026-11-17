#!/usr/bin/env python3
"""Build collision-screened, coupled EE position/orientation commands on CPU."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mujoco_mount_compare'))
from mount_workspace_study import StudyConverter,setup,self_collision,mass_properties
from prepare_asset import prepare


def main():
    out=ROOT/'resources/commands';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'resources/robots/as2piper/urdf/as2piper_umi_v7_mesh_seated_plate_only.urdf'
    urdf=prepare(source);xml=out/'plate_forward.xml';StudyConverter(urdf,xml).build()
    m=mujoco.MjModel.from_xml_path(str(xml));d,names,feet,radii=setup(m,'plate_only_forward')
    q0=d.qpos.copy();ids=[int(m.joint(n).qposadr) for n in names];home=q0[ids].copy()
    joint_ids=[m.joint(n).id for n in names];limits=m.jnt_range[joint_ids].copy()
    base=m.body('piper_base_link').id;tcp=m.body('piper_umi_nominal_tcp').id
    rng=np.random.default_rng(20261029);bank=[];positions=[];rotations=[];peak_torques=[]
    offsets=np.array([.45,.55,.65,.35,.45,.35])
    lows=np.maximum(home-offsets,limits[:,0]+.04);highs=np.minimum(home+offsets,limits[:,1]-.04)
    trials=0
    for _ in range(12000):
        trials+=1;goal=rng.uniform(lows,highs);valid=True;peak=0.
        for s in np.linspace(0,1,41):
            d.qpos[:]=q0;d.qpos[ids]=home+(goal-home)*s;mujoco.mj_forward(m,d)
            p=d.xpos[tcp]-d.xpos[base]
            # Forward manipulation start region, with head/body clearance.
            if not (.34<=p[0]<=.58 and abs(p[1])<=.22 and .06<=p[2]<=.40) or self_collision(m,d):valid=False;break
            tau=d.qfrc_bias[[int(m.joint(n).dofadr) for n in names]]
            peak=max(peak,float(np.max(np.abs(tau))))
            # Existing demo screening assumption, not a motor specification.
            if peak>9.:valid=False;break
        if not valid:continue
        bank.append(goal);positions.append(p.copy());rotations.append(d.xquat[tcp][[1,2,3,0]].copy());peak_torques.append(peak)
        if len(bank)>=256:break
    assert len(bank)>=64, 'Too few feasible paths: '+str(len(bank))
    bank=np.array(bank);p=np.array(positions)
    np.savez_compressed(str(out/'arm_commands.npz'),home=home,joints=bank,positions=p,quaternions_xyzw=np.array(rotations))
    metadata=dict(asset_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),joint_names=names,count=len(bank),attempted=trials,
        reference_frame='piper_base_link; full body rotation, not yaw-only',mount_body_xyz_m=[.005,0,.08202847821788371],
        home_joint_rad=home.tolist(),joint_sampling_bounds_rad=[lows.tolist(),highs.tolist()],
        tcp_endpoint_bounds_arm_m=[p.min(0).tolist(),p.max(0).tolist()],
        screening_envelope_arm_m=dict(x=[.34,.58],y=[-.22,.22],z=[.06,.40]),
        trajectory='home -> sampled endpoint -> home; joint interpolation, FK positions and orientations coupled',
        path_samples=41,max_arm_gravity_torque_Nm=max(peak_torques),screening_torque_limit_assumed_Nm=9.,
        collision_tolerance_m=.0005,nominal_base_z_m=float(q0[2]),neutral_foot_radius_m=radii,
        limitations=['Nominal grounded leg pose only; rough terrain/moving legs can collide.',
                    '9 Nm is an inherited simulation screening assumption, not a verified hardware rating.',
                    'Finite path samples are not continuous collision or dynamic feasibility proof.',
                    'No grasped object/camera/cable mass or visibility guarantee.'])
    (out/'arm_commands.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
