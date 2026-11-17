"""Head-mounted PiPER/UMI and back-mounted Orin adaptation of original VBC."""
from .b1z1_config import B1Z1RoughCfg, B1Z1RoughCfgPPO
from pathlib import Path
import json

HW = json.loads((Path(__file__).resolve().parents[3] / "validation/hardware.json").read_text())


class AS2HeadOrinCfg(B1Z1RoughCfg):
    class env(B1Z1RoughCfg.env):
        observe_foot_contacts = False
        observe_gait_commands = True
        num_proprio = 67
        num_observations = num_proprio * (B1Z1RoughCfg.env.history_len + 1) + B1Z1RoughCfg.env.num_priv
        num_gripper_joints = 2
        env_spacing = 3.
        stop_update_goal = False
        pitch_control = False
        # Preserve the effective legacy schedule explicitly: 0 -> 1 policy step.
        # This is not measured AS2/PiPER latency.
        action_delay = 1
        action_delay_warmup_steps = 10000 * 24

    class init_state(B1Z1RoughCfg.init_state):
        pos = [0., 0., 0.3264069024246581]
        default_joint_angles = {
            **{leg + '_' + joint + '_joint': angle
               for leg in ('FL', 'FR', 'RL', 'RR')
               for joint, angle in [('hip', 0.), ('thigh', .8), ('calf', -1.6)]},
            **{'piper_joint' + str(i): angle
               for i, angle in enumerate([0., 1.1, -.55, 0., 0., 0.], 1)},
            'piper_gripper_joint1': .049,
            'piper_gripper_joint2': -.049,
        }

    class asset(B1Z1RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf'
        gripper_name = 'piper_umi_nominal_tcp'
        penalize_contacts_on = ['thigh', 'base_link', 'calf']
        arm_base_offset = [.22435, 0., .1319]

    class control(B1Z1RoughCfg.control):
        stiffness = {**{name + '_joint': gain for name, gain in zip(('hip', 'thigh', 'calf'), HW['kp'])}, 'piper_joint': 5.}
        damping = {**{name + '_joint': gain for name, gain in zip(('hip', 'thigh', 'calf'), HW['kd'])}, 'piper_joint': .5}
        leg_armature = HW['armature']
        # Uncalibrated PhysX gains inherited from VBC, NOT PiPER MIT gains.
        arm_drive_stiffness = 400.
        arm_drive_damping = 40.
        gripper_drive_stiffness = 400.
        gripper_drive_damping = 40.
        # Official position-mode maximum configurable speed, not device readback.
        arm_command_velocity_limit = 3.
        arm_command_acceleration_limit = None  # requires device/firmware readback
        arm_command_delay_steps = 0  # unknown physical latency; no invented value
        arm_effort_limit_verified = False

    class goal_ee(B1Z1RoughCfg.goal_ee):
        # Static nominal flat-ground witness graph; no online IK rejection loop.
        feasible_bank_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/as2piper/workspace/head_orin_feasible.json'
        arm_induced_pitch = 2.455216137744035
        orientation_roll_offset = 0.
        # Conservative body/head + Orin bounds, relative to the arm origin.
        collision_lower_limits = [-.32435, -.112, -.1959]
        # Extend forward face 20 mm: cover the visual head tip with ~11 mm margin.
        collision_upper_limits = [.12165, .112, .0158]
        underground_limit = -.4563069024246581

        class sphere_center:
            # Virtual yaw frame: nominal arm origin, independent of body z/roll/pitch.
            # Nominal base height 0.3244069024246581 + mount height 0.1319.
            height_roll_pitch_invariant = True
            x_offset = .22435
            y_offset = 0.
            z_invariant_offset = .4563069024246581

        class ranges:
            init_pos_start = [.427898468676838, .421686284147847, 0.]
            init_pos_end = [.427898468676838, .421686284147847, 0.]
            # Legacy ranges below are inactive when feasible_bank_file is configured.
            # Bank nodes and screened edges define position + orientation sampling.
            pos_l = [.40, .48]
            pos_p = [.10, .50]
            pos_y = [-.25, .25]
            delta_orn_r = [-.08, .08]
            delta_orn_p = [-.08, .08]
            delta_orn_y = [-.08, .08]
            final_tracking_ee_reward = .55

    class domain_rand(B1Z1RoughCfg.domain_rand):
        # User-selected fixed 400/40 arm drive, matching effective VBC behavior.
        # Leg motor DR is inherited unchanged; retain the arm DR mechanism for tests.
        arm_motor_strength_range = [1., 1.]
        added_mass_range = HW['added_base_mass_kg']
        added_com_range_x = [-HW['base_com_half_range_m'][0], HW['base_com_half_range_m'][0]]
        added_com_range_y = [-HW['base_com_half_range_m'][1], HW['base_com_half_range_m'][1]]
        added_com_range_z = [-HW['base_com_half_range_m'][2], HW['base_com_half_range_m'][2]]
        max_push_vel_xy = HW['max_push_velocity_m_s']
        standing_push_multiplier = 2.5

    class termination(B1Z1RoughCfg.termination):
        height_reference = 'terrain'

    class rewards(B1Z1RoughCfg.rewards):
        base_height_target = .3244069024246581
        # 0.30 m B1 target x approximately 0.60 geometric scale.
        feet_height_target = 0.18

        class scales(B1Z1RoughCfg.rewards.scales):
            # This unused legacy term requires a foot-mounted force sensor.
            foot_contacts_z = 0.0


class AS2HeadOrinCfgPPO(B1Z1RoughCfgPPO):
    class runner(B1Z1RoughCfgPPO.runner):
        experiment_name = 'as2piper_head_orin_nocontact'
        resume = False
