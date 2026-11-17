"""Assert that only explicitly necessary hardware/asset fields differ."""
import json
import numpy as np
from pathlib import Path
import isaacgym
from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg,B1Z1RoughCfgPPO
from legged_gym.envs.manip_loco.as2piper_config import AS2PiPERPlateCfg,AS2PiPERPlateCfgPPO
from legged_gym.utils.helpers import class_to_dict


def differences(a,b,prefix=''):
    found={}
    for key in sorted(set(a)|set(b)):
        name=prefix+key
        av,bv=a.get(key),b.get(key)
        if isinstance(av,dict) and isinstance(bv,dict):found.update(differences(av,bv,name+'.'))
        elif not np.array_equal(av,bv):found[name]={'original':av,'as2':bv}
    return found


if __name__=='__main__':
    original=class_to_dict(B1Z1RoughCfg());adapted=class_to_dict(AS2PiPERPlateCfg())
    delta=differences(original,adapted)
    allowed=('asset.','init_state.default_joint_angles.','control.stiffness.','control.damping.',
             'control.arm_position_','control.gripper_position_','control.arm_effort_limit',
             'env.num_gripper_joints','env.observe_gait_commands','goal_ee.','init_state.pos',
             'rewards.base_height_target')
    assert all(any(k.startswith(prefix) for prefix in allowed) for k in delta),delta
    for key in ('domain_rand','terrain','commands','termination','noise','normalization'):
        assert not differences(original[key],adapted[key]),key
    reward_delta=differences(original['rewards'],adapted['rewards'])
    assert set(reward_delta)=={'base_height_target'},reward_delta
    assert adapted['rewards']['base_height_target']==.3244069024246581
    assert adapted['env']['num_envs']==6144
    ppo=differences(class_to_dict(B1Z1RoughCfgPPO()),class_to_dict(AS2PiPERPlateCfgPPO()))
    assert set(ppo)=={'runner.experiment_name'},ppo
    out=Path(__file__).resolve().parent/'audit_forward/settings_diff.json'
    out.write_text(json.dumps({'environment_differences':delta,'ppo_differences':ppo,
                              'reward_formulas_scales_equal':True,'reward_parameter_differences':reward_delta,
                              'num_envs':6144,'max_iterations':45000},indent=2))
    print('Forward hardware-setting contract PASS. Necessary differences:',len(delta))
