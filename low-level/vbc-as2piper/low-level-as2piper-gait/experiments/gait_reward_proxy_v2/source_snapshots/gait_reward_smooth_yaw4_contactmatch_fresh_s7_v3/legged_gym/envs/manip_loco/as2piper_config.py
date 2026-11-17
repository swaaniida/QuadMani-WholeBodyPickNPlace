"""Plate-only hardware adaptation. Reward definitions/scales are inherited unchanged."""
from .b1z1_config import B1Z1RoughCfg, B1Z1RoughCfgPPO


class AS2PiPERPlateCfg(B1Z1RoughCfg):
    class env(B1Z1RoughCfg.env):
        num_gripper_joints = 2
        observe_gait_commands = True

    class init_state(B1Z1RoughCfg.init_state):
        pos = [0.,0.,.3264069024246581]
        default_joint_angles = {
            **{leg+'_'+name+'_joint': value for leg in ('FL','FR','RL','RR')
               for name,value in [('hip',0.),('thigh',.8),('calf',-1.6)]},
            **{'piper_joint'+str(i): v for i,v in enumerate([0.,1.1,-.55,0.,0.,0.],1)},
            'piper_gripper_joint1': .049,
            'piper_gripper_joint2': -.049,
        }

    class asset(B1Z1RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/as2piper/urdf/as2piper_plate_forward_training.urdf'
        gripper_name = 'piper_umi_nominal_tcp'
        penalize_contacts_on = ['thigh','base_link','calf']
        collapse_fixed_joints = True
        arm_base_offset = [.005,0.,.08202847821788371]

    class control(B1Z1RoughCfg.control):
        stiffness = {'joint': 80., 'piper_joint': 5.}
        damping = {'joint': 2., 'piper_joint': .5}
        arm_position_stiffness = 400.
        arm_position_damping = 40.
        gripper_position_stiffness = 1800.
        gripper_position_damping = 35.
        arm_effort_limit = float('inf')  # Use the URDF effort, as in the original.

    class goal_ee(B1Z1RoughCfg.goal_ee):
        orientation_roll_offset = 0.
        command_mode = 'sphere'
        sampling_mode = 'b1_spherical_with_as2_collision_rejection'
        reference_frame = 'arm_origin_yaw_aligned'
        arm_induced_pitch = 2.455216137744035
        collision_lower_limits = [-.185,-.12,-.15]
        collision_upper_limits = [.335,.12,.06]
        underground_limit = -.30
        class sphere_center:
            x_offset = 0.
            y_offset = 0.
            z_invariant_offset = 0.
        class ranges:
            init_pos_start = [.4278984686862198,.421686284190671,0.]
            init_pos_end = [.4278984686862198,.421686284190671,0.]
            pos_l = [.38,.52]
            pos_p = [-.25,.60]
            pos_y = [-.40,.40]
            delta_orn_r = [-.15,.15]
            delta_orn_p = [-.15,.15]
            delta_orn_y = [-.15,.15]
            final_tracking_ee_reward = .55
        # Sampling and spherical interpolation are inherited from B1.

    class rewards(B1Z1RoughCfg.rewards):
        # Geometric height reference, not reward-scale tuning. With the AS2
        # neutral leg pose, all four foot sphere bottoms touch at this height.
        base_height_target = .3244069024246581


class AS2PiPERPlateCfgPPO(B1Z1RoughCfgPPO):
    class runner(B1Z1RoughCfgPPO.runner):
        experiment_name = 'as2piper_plate_forward_spherical'
        resume = False
