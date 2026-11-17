"""Continue a walking policy with random EE goals and hardware-sized DR."""
from .as2piper_gait import AS2PiPERGait,AS2PiPERGaitCfg,AS2PiPERGaitCfgPPO
from .as2piper_hw import AS2PiPERHardware
from .as2piper_hw_config import AS2PiPERHardwareCfg


class AS2PiPERWalkWBCCfg(AS2PiPERGaitCfg):
    class goal_ee(AS2PiPERHardwareCfg.goal_ee):
        pass
    class domain_rand(AS2PiPERHardwareCfg.domain_rand):
        pass


class AS2PiPERWalkWBCCfgPPO(AS2PiPERGaitCfgPPO):
    class runner(AS2PiPERGaitCfgPPO.runner):
        experiment_name='as2piper_walk_wbc'


class AS2PiPERWalkWBCNominalCfg(AS2PiPERWalkWBCCfg):
    # Learn the newly varying arm-state inputs before adding payload/COM shifts.
    class domain_rand(AS2PiPERGaitCfg.domain_rand):
        pass


class AS2PiPERWalkWBC(AS2PiPERGait):
    def _resample_ee_goal(self,ids,is_init=False):
        return AS2PiPERHardware._resample_ee_goal(self,ids,is_init)

    def _step_contact_targets(self):
        super()._step_contact_targets()
        # A stationary command asks for four supporting feet, not a frozen
        # transition between trot phases (which would give contact target .5).
        self.desired_contact_states[~self._get_walking_cmd_mask()]=1.
