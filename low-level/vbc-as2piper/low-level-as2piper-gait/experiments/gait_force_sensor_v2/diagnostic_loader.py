"""Reproduce rejected diagnostic hooks in memory; production source stays original."""
from pathlib import Path

def load():
    folder=Path(__file__).resolve().parent
    from legged_gym.envs.manip_loco.manip_loco import ManipLoco
    ns={'__name__':'legged_gym.envs.manip_loco._diagnostic_base','__package__':'legged_gym.envs.manip_loco'}
    exec(compile((folder/'manip_loco_diagnostic_hooks.py').read_text(),str(folder/'manip_loco_diagnostic_hooks.py'),'exec'),ns)
    for name in ['step','_create_envs']:setattr(ManipLoco,name,getattr(ns['ManipLoco'],name))
    ns={'__name__':'legged_gym.envs.manip_loco._diagnostic_study','__package__':'legged_gym.envs.manip_loco'}
    exec(compile((folder/'gait_diagnostic_hooks.py').read_text(),str(folder/'gait_diagnostic_hooks.py'),'exec'),ns)
    return tuple(ns[k] for k in ['AS2PiPERGaitStudy','AS2PiPERGaitStudyCfg','AS2PiPERGaitStudyCfgPPO','configure'])
