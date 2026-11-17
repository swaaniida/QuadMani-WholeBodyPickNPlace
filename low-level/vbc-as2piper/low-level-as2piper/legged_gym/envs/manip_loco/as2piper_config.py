"""Plate-only hardware adaptation. Reward definitions/scales are inherited unchanged."""
from .b1z1_config import B1Z1RoughCfg, B1Z1RoughCfgPPO


class AS2PiPERPlateCfg(B1Z1RoughCfg):
    class env(B1Z1RoughCfg.env):
        num_gripper_joints = 2

    class init_state(B1Z1RoughCfg.init_state):
        default_joint_angles = {
            **{k:v for k,v in B1Z1RoughCfg.init_state.default_joint_angles.items() if not k.startswith('z1_')},
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
        stiffness = {'joint': 80., 'piper_joint': 5.}
        damping = {'joint': 2., 'piper_joint': .5}
        arm_position_stiffness = 400.
        arm_position_damping = 40.
        gripper_position_stiffness = 1800.
        gripper_position_damping = 35.
        arm_effort_limit = float('inf')  # Use the URDF effort, as in the original.

    class goal_ee(B1Z1RoughCfg.goal_ee):
        orientation_roll_offset = 0.


class AS2PiPERPlateCfgPPO(B1Z1RoughCfgPPO):
    class runner(B1Z1RoughCfgPPO.runner):
        experiment_name = 'as2piper_plate_only'
        resume = False
