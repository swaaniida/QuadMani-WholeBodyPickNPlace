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
             'env.num_gripper_joints','goal_ee.orientation_roll_offset')
    assert all(any(k.startswith(prefix) for prefix in allowed) for k in delta),delta
    for key in ('rewards','domain_rand','terrain','commands','termination','noise','normalization'):
        assert not differences(original[key],adapted[key]),key
    assert adapted['env']['num_envs']==6144
    ppo=differences(class_to_dict(B1Z1RoughCfgPPO()),class_to_dict(AS2PiPERPlateCfgPPO()))
    assert set(ppo)=={'runner.experiment_name'},ppo
    out=Path(__file__).resolve().parent/'audit/original_settings.json'
    out.write_text(json.dumps({'environment_differences':delta,'ppo_differences':ppo,
                              'all_rewards_exactly_equal':True,'num_envs':6144,'max_iterations':45000},indent=2))
    print('Original-setting equality PASS. Necessary hardware differences:',len(delta))
