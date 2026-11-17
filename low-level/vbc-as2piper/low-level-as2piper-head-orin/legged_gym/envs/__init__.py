from .manip_loco.manip_loco import ManipLoco
from .manip_loco.as2_head_orin_config import AS2HeadOrinCfg, AS2HeadOrinCfgPPO
from legged_gym.utils.task_registry import task_registry

task_registry.register('as2piper_head_orin', ManipLoco, AS2HeadOrinCfg(), AS2HeadOrinCfgPPO(), 'as2piper_head_orin')
