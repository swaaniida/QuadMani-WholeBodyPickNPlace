"""Fixed-torso physical tracking check with forbidden online guard calls."""
import json
from pathlib import Path
import isaacgym
from audit_hw import fixed_base_hw as audit
from legged_gym.envs.manip_loco.as2piper_walk_wbc_vbc import AS2PiPERWalkWBCVBC, AS2PiPERWalkWBCVBCCfg


class CountedSampler(AS2PiPERWalkWBCVBC):
    def _resample_ee_goal(self, ids, is_init=False):
        result = super()._resample_ee_goal(ids, is_init)
        if not is_init:
            self.pose_requests += len(ids)
            self.pose_accepted += len(ids)
        return result


class ForbiddenOnlineGuard:
    def __init__(self, *args, **kwargs):
        pass
    def solve(self, *args, **kwargs):
        raise AssertionError('Online iterative candidate IK must not run')
    def collides(self, *args, **kwargs):
        raise AssertionError('Online OBB rejection must not run')


if __name__ == '__main__':
    audit.AS2PiPERForward = CountedSampler
    audit.AS2PiPERPlateCfg = AS2PiPERWalkWBCVBCCfg
    audit.ArmCollisionGuard = ForbiddenOnlineGuard
    audit.OUT = Path(__file__).resolve().parent/'audit_hw/goal_runtime_20261029/fixed_base_screened'
    audit.OUT.mkdir(parents=True, exist_ok=True)
    audit.main()
    path = audit.OUT/'summary.json'
    result = json.loads(path.read_text())
    result['conditions'] = ('Fixed torso, nominal mass/COM, gravity; original VBC online sphere sampler/checker; '
                            'offline-screened ranges; iterative candidate IK/OBB calls forbidden; not locomotion.')
    result['pose_counter_definition'] = 'Generated requests, not online IK/OBB certification.'
    path.write_text(json.dumps(result, indent=2))
