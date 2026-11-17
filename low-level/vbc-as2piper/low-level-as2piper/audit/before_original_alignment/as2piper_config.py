"""Plate-only hardware adaptation. Reward definitions/scales are inherited unchanged."""
from .b1z1_config import B1Z1RoughCfg, B1Z1RoughCfgPPO


class AS2PiPERPlateCfg(B1Z1RoughCfg):
    class env(B1Z1RoughCfg.env):
        num_envs = 1024
        num_gripper_joints = 2

    class init_state(B1Z1RoughCfg.init_state):
        pos = [0., 0., 0.33]
        default_joint_angles = {
            **{leg+'_'+joint: value for leg in ('FL','FR','RL','RR')
               for joint,value in (('hip_joint',0.),('thigh_joint',.8),('calf_joint',-1.6))},
            **{'piper_joint'+str(i): v for i,v in enumerate([0.,1.1,-.55,0.,0.,0.],1)},
            'piper_gripper_joint1': .049,
            'piper_gripper_joint2': -.049,
        }

    class asset(B1Z1RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/as2piper/urdf/as2piper_plate_training.urdf'
        gripper_name = 'piper_umi_nominal_tcp'
        penalize_contacts_on = ['thigh','base_link','calf']
        collapse_fixed_joints = True
        arm_base_offset = [-.02,0.,.08202847821788371]

    class control(B1Z1RoughCfg.control):
        stiffness = {'joint': 90., 'piper_joint': 35.}
        damping = {'joint': 3.5, 'piper_joint': 1.5}
        arm_position_stiffness = 80.
        arm_position_damping = 8.
        gripper_position_stiffness = 1800.
        gripper_position_damping = 35.
        arm_effort_limit = 9.  # Conservative simulation limit; hardware calibration pending.

    class goal_ee(B1Z1RoughCfg.goal_ee):
        arm_induced_pitch = 2.4
        orientation_roll_offset = 0.
        collision_lower_limits = [-.18,-.13,-.45]
        collision_upper_limits = [.35,.13,-.23]
        underground_limit = -.40
        class sphere_center(B1Z1RoughCfg.goal_ee.sphere_center):
            x_offset = -.02
            z_invariant_offset = .45
        class ranges(B1Z1RoughCfg.goal_ee.ranges):
            init_pos_start = [.35,.5,0.]
            init_pos_end = [.38,.3,0.]
            pos_l = [.25,.48]
            pos_p = [-.65,.8]
            pos_y = [-.8,.8]
            delta_orn_r = [-.15,.15]
            delta_orn_p = [-.15,.15]
            delta_orn_y = [-.15,.15]

    class domain_rand(B1Z1RoughCfg.domain_rand):
        added_mass_range = [0.,3.]
        added_com_range_x = [-.03,.03]
        added_com_range_y = [-.02,.02]
        added_com_range_z = [-.02,.02]

    class rewards(B1Z1RoughCfg.rewards):
        # Same reward function and weights; only the hardware length target changes.
        base_height_target = .30


class AS2PiPERPlateCfgPPO(B1Z1RoughCfgPPO):
    class runner(B1Z1RoughCfgPPO.runner):
        experiment_name = 'as2piper_plate_only'
        resume = False
