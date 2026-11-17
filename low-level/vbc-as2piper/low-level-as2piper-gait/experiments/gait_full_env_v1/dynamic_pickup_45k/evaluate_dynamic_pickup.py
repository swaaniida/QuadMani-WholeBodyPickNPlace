"""Evaluation-only wide-range EE goals for a floor-pick stress-test rollout."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GAIT = HERE.parents[2]
sys.path.insert(0, str(GAIT))

from legged_gym.envs.manip_loco.as2piper_walk_wbc_vbc import AS2PiPERWalkWBCVBCCfg

# Override only this evaluator process. Keep VBC's spherical sampling and
# online collision/floor guard; expand the position envelope down toward floor.
ranges = AS2PiPERWalkWBCVBCCfg.goal_ee.ranges
ranges.pos_l = [0.42, 0.52]
ranges.pos_p = [-0.85, 0.35]
ranges.pos_y = [-0.08, 0.08]
ranges.delta_orn_r = [-0.04, 0.04]
ranges.delta_orn_p = [-0.04, 0.04]
ranges.delta_orn_y = [-0.04, 0.04]
# Permit the sampled TCP to approach the floor in the arm-base frame. This is
# still the existing path/body-box check; only its prior local-z cutoff moves.
AS2PiPERWalkWBCVBCCfg.goal_ee.underground_limit = -0.40

import evaluate_hw

evaluate_hw.main()
