from legged_gym.envs.manip_loco.as2_head_orin_config import AS2HeadOrinCfg, AS2HeadOrinCfgPPO

class Force40Cfg(AS2HeadOrinCfg):
    class rewards(AS2HeadOrinCfg.rewards):
        max_contact_force = 40.

class Force70Cfg(AS2HeadOrinCfg):
    class rewards(AS2HeadOrinCfg.rewards):
        max_contact_force = 70.

class ForceOffCfg(AS2HeadOrinCfg):
    class rewards(AS2HeadOrinCfg.rewards):
        max_contact_force = 40.
        class scales(AS2HeadOrinCfg.rewards.scales):
            feet_contact_forces = 0.

CASES = {'A': Force40Cfg, 'B': Force70Cfg, 'C': ForceOffCfg}
TASKS = {key: 'as2piper_nocontact_force_' + key.lower() for key in CASES}

def configs(case, seed=1):
    env = CASES[case]()
    train = AS2HeadOrinCfgPPO()
    env.seed = train.seed = seed
    assert env.rewards.feet_height_target == .18
    assert env.rewards.scales.feet_height == 1.
    assert env.env.observe_gait_commands and not env.env.observe_foot_contacts
    return env, train

def register_tasks():
    from legged_gym.utils.task_registry import task_registry
    from .runtime import MeteredEnvironment
    for case in CASES:
        env, train = configs(case)
        task_registry.register(TASKS[case], MeteredEnvironment, env, train, TASKS[case])
