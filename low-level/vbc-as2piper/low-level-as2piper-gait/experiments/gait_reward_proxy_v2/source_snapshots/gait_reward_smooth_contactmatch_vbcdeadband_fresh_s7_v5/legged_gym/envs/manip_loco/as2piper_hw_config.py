"""AS2 hardware-derived flat-ground WBC training, independent of the B1 task."""
import json
from pathlib import Path
from .as2piper_config import AS2PiPERPlateCfg,AS2PiPERPlateCfgPPO

HW=json.loads((Path(__file__).resolve().parents[3]/'resources/commands/hardware.json').read_text())['derived']


class AS2PiPERHardwareCfg(AS2PiPERPlateCfg):
    class env(AS2PiPERPlateCfg.env):
        frequencies=HW['gait_hz']
        command_warmup_iterations=1000
        stop_update_goal=False
        pitch_control=False
    class init_state(AS2PiPERPlateCfg.init_state):
        init_vel_perturb_range=.02
        leg_reset_fraction=.05
    class control(AS2PiPERPlateCfg.control):
        leg_armature=HW['leg_armature']
        stiffness={**{j+'_joint':v for j,v in zip(('hip','thigh','calf'),HW['leg_kp'])},'piper_joint':5.}
        damping={**{j+'_joint':v for j,v in zip(('hip','thigh','calf'),HW['leg_kd'])},'piper_joint':.5}
        action_scale=HW['leg_action_scale']*4+[2.1,.6,.6,0.,0.,0.]
    class goal_ee(AS2PiPERPlateCfg.goal_ee):
        class ranges(AS2PiPERPlateCfg.goal_ee.ranges):
            init_pos_end=[.46,.25,0.]
    class commands(AS2PiPERPlateCfg.commands):
        lin_vel_x_clip=.05
        ang_vel_yaw_clip=.1
        class ranges:
            lin_vel_x=[-HW['max_linear_velocity_m_s'],HW['max_linear_velocity_m_s']]
            ang_vel_yaw=[-HW['max_yaw_velocity_rad_s'],HW['max_yaw_velocity_rad_s']]
    class domain_rand(AS2PiPERPlateCfg.domain_rand):
        added_mass_range=HW['added_base_mass_kg']
        added_com_range_x=[-HW['base_com_half_range_m'][0],HW['base_com_half_range_m'][0]]
        added_com_range_y=[-HW['base_com_half_range_m'][1],HW['base_com_half_range_m'][1]]
        added_com_range_z=[-HW['base_com_half_range_m'][2],HW['base_com_half_range_m'][2]]
        leg_motor_strength_range=[.9,1.1]
        friction_range=[.6,1.2]
        max_push_vel_xy=HW['max_push_velocity_m_s']
    class terrain(AS2PiPERPlateCfg.terrain):
        mesh_type='plane'
        measure_heights=False
        curriculum=False
    class rewards(AS2PiPERPlateCfg.rewards):
        max_contact_force=HW['max_contact_force_N']
        feet_height_target=HW['feet_height_target_m']
        gait_force_sigma=HW['gait_force_sigma']
        gait_vel_sigma=HW['gait_vel_sigma']


class AS2PiPERHardwareCfgPPO(AS2PiPERPlateCfgPPO):
    class policy(AS2PiPERPlateCfgPPO.policy):
        init_std=[[.3,.3,.3]*4+[1.]*6]
        # Angle action units: avoid a high exploration floor after convergence.
    class algorithm(AS2PiPERPlateCfgPPO.algorithm):
        min_policy_std=[[.10,.10,.10]*4+[.2]*3+[.05]*3]
    class runner(AS2PiPERPlateCfgPPO.runner):
        experiment_name='as2piper_hw_flat_wbc'

