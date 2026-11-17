"""Offline-screened AS2 ranges with the original VBC online sampler/checker."""
from pathlib import Path
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.urdf_chain import URDFChain
from .manip_loco import ManipLoco
from .as2piper_forward import AS2PiPERForward
from .as2piper_walk_wbc import AS2PiPERWalkWBC, AS2PiPERWalkWBCNominalCfg, AS2PiPERWalkWBCCfgPPO


class AS2PiPERWalkWBCVBCCfg(AS2PiPERWalkWBCNominalCfg):
    class goal_ee(AS2PiPERWalkWBCNominalCfg.goal_ee):
        sampling_mode = 'original_vbc_spherical_offline_screened_ranges'
        validation_mode = 'vbc_ee_box_and_floor_only'
        offline_range_report = 'audit_hw/goal_runtime_20261029/range_holdout.json'
        class ranges(AS2PiPERWalkWBCNominalCfg.goal_ee.ranges):
            pos_l = [.46, .49]
            pos_p = [.30, .45]
            pos_y = [-.10, .10]
            delta_orn_r = [-.015, .015]
            delta_orn_p = [-.015, .015]
            delta_orn_y = [-.015, .015]
            init_pos_end = [.475, .375, 0.]


class AS2PiPERWalkWBCVBCCfgPPO(AS2PiPERWalkWBCCfgPPO):
    class runner(AS2PiPERWalkWBCCfgPPO.runner):
        experiment_name = 'as2piper_walk_wbc_vbc_sampler'


class AS2PiPERWalkWBCVBC(AS2PiPERWalkWBC):
    # Same checker implementation as B1: ten EE points against a forbidden
    # box/floor, not repeated candidate IK or arm/body OBB rejection.
    _collision_check = ManipLoco._collision_check

    def _init_buffers(self):
        ManipLoco._init_buffers(self)
        asset = Path(self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
        # Retain AS2 reset-height FK; do not instantiate an online pose guard.
        self.leg_fk = [URDFChain(asset, 'base_link', leg+'_foot',
            [leg+'_'+joint+'_joint' for joint in ('hip', 'thigh', 'calf')], self.device)
            for leg in ('FL', 'FR', 'RL', 'RR')]
        self.pose_fallback_count = 0

    def _resample_ee_goal(self, env_ids, is_init=False):
        if is_init:
            # AS2 reset must immediately publish its FK-matched start goal.
            return AS2PiPERForward._resample_ee_goal(self, env_ids, is_init=True)
        return ManipLoco._resample_ee_goal(self, env_ids, is_init=False)
