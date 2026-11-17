"""Derive simulation starting parameters from mass, geometry and static loads."""
import hashlib,json,sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
import mujoco

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mujoco_mount_compare'))
from mount_workspace_study import setup,mass_properties


def measure(path,kind):
    m=mujoco.MjModel.from_xml_path(str(path));d,_,feet,_=setup(m,kind)
    props=mass_properties(m,d);mass=props['mass_kg'];com=np.array(props['com_world_m'])
    p=d.xpos[feet];A=np.stack([np.ones(4),p[:,0]-com[0],p[:,1]-com[1]])
    load=lsq_linear(A,np.array([mass*9.81,0,0]),bounds=(0,np.inf)).x
    assert np.linalg.norm(A@load-[mass*9.81,0,0])<1e-5
    tau=d.qfrc_bias.copy()
    for bid,f in zip(feet,load):
        J=np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,J,None,bid)
        tau-=J.T@np.array([0,0,f])
    dense=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,dense,d.qM)
    idx=[int(m.joint(leg+'_'+j+'_joint').dofadr) for leg in ['FL','FR','RL','RR'] for j in ['hip','thigh','calf']]
    return dict(**props,stance_height_m=float(d.qpos[2]),foot_positions=p.tolist(),
        static_foot_load_N=load.tolist(),static_leg_torque_Nm=tau[idx].tolist(),
        leg_inertia_diagonal_kg_m2=dense.diagonal()[idx].tolist(),
        asset=str(path),asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    a=measure(ROOT/'resources/commands/plate_forward.xml','plate_only_forward')
    b=measure(ROOT.parent/'mujoco_mount_compare/models/mount_workspace/b1z1.xml','b1z1')
    mass=a['mass_kg']/b['mass_kg'];length=a['stance_height_m']/b['stance_height_m'];time=np.sqrt(length)
    frequency=2/time
    static=np.abs(a['static_leg_torque_Nm']).reshape(4,3).max(0)
    preset=Path('/data/team/users/suhwan/robot-sources/unitree_ros/robots/as2_description/as2.xml')
    preset_xml=ET.parse(preset).getroot()
    armature=np.array([float(preset_xml.find("./default/default[@class='"+j+"_joint']/joint").get('armature')) for j in ['hip','thigh','calf']])
    inertia=np.array(a['leg_inertia_diagonal_kg_m2']).reshape(4,3).max(0)+armature
    # Uniform gravity similarity is only a candidate: AS2 is not a scaled B1.
    kp_similarity=80*mass*length
    kp_static=static/.12  # design allowance: <=0.12 rad nominal load sag
    kp_bandwidth=inertia*(2*np.pi*2*frequency)**2  # 2x gait frequency
    kp=np.maximum(np.maximum(kp_similarity,kp_static),kp_bandwidth)
    kd=2*.7*np.sqrt(kp*inertia)  # swing-joint inertia; not full stance decoupling
    effort=np.array([60.,60.,90.])
    scale=np.minimum(np.minimum([.4,.45,.45],.2*effort/kp),.30)
    p=np.array(a['foot_positions']);half=(p.max(0)-p.min(0))/2
    # Full-system COM shift <=5% of support half-width, expressed at base link.
    com_xy=.05*half[:2]*a['mass_kg']/8.8
    values=dict(leg_kp=np.round(kp,3).tolist(),leg_kd=np.round(kd,3).tolist(),leg_armature=armature.tolist(),
        leg_action_scale=np.round(scale,4).tolist(),gait_hz=float(frequency),
        max_linear_velocity_m_s=float(.8*time),max_yaw_velocity_rad_s=1.,
        added_base_mass_kg=[0.,min(15*mass,.10*a['mass_kg'])],
        base_com_half_range_m=[float(com_xy[0]),float(com_xy[1]),float(min(com_xy))],
        max_push_velocity_m_s=float(.5*time),
        # Existing feet-height reward compares the norm of the two front heights.
        feet_height_target_m=float(np.sqrt(2)*(.028+.10*a['stance_height_m'])),
        # Allow a 1.5 dynamic factor with two legs supporting the entire robot.
        max_contact_force_N=float(1.5*a['mass_kg']*9.81/2),gait_vel_sigma=float(.5*length),
        gait_force_sigma=float(.5*mass*mass))
    out=dict(as2=a,b1=b,official_simulation_preset=str(preset),official_preset_sha256=hashlib.sha256(preset.read_bytes()).hexdigest(),
        ratios=dict(mass=mass,length=length,time=time,torque=mass*length,
        angular_damping=mass*length**1.5),derived=values,
        design_choices=dict(max_static_joint_sag_rad=.12,swing_damping_ratio=.7,
            swing_bandwidth_to_gait=2,exploration_torque_budget_fraction=.2,
            total_added_mass_fraction=.1,full_com_support_half_width_fraction=.05),
        limitations=['Gravity dynamic similarity is approximate for different morphologies.',
            'Static support assumes four vertical nonnegative forces on a plane.',
            'Inertia diagonal damping neglects coupling; armature is an official simulation preset, not motor identification.',
            'URDF effort is a simulation limit, not a measured continuous motor rating.',
            'Camera, cables and exact mount hardware masses remain unverified.'])
    target=ROOT/'resources/commands/hardware.json';target.write_text(json.dumps(out,indent=2));target.chmod(0o664)
    print(json.dumps(out['ratios'],indent=2));print(json.dumps(values,indent=2))


if __name__=='__main__':main()
