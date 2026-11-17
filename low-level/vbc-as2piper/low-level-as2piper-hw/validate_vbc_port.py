"""Physical fixed-torso EE test of the restored VBC frame and expanded range."""
import json
from pathlib import Path
import isaacgym
import torch
from audit_hw import fixed_base_hw as audit
from validate_vbc_sampler import ForbiddenOnlineGuard
from legged_gym.envs.manip_loco.as2piper_vbc_port import AS2PiPERVBCPort,AS2PiPERVBCPortCfg


class CountedPort(AS2PiPERVBCPort):
    def _resample_ee_goal(self, ids, is_init=False):
        if not hasattr(self,'ee_goal_center_offset'):
            c=self.cfg.goal_ee.sphere_center
            self.ee_goal_center_offset=torch.tensor([c.x_offset,c.y_offset,c.z_invariant_offset],device=self.device).repeat(self.num_envs,1)
        result=super()._resample_ee_goal(ids,is_init)
        if not is_init:
            self.pose_requests+=len(ids);self.pose_accepted+=len(ids)
        return result


if __name__=='__main__':
    audit.AS2PiPERForward=CountedPort;audit.AS2PiPERPlateCfg=AS2PiPERVBCPortCfg
    audit.ArmCollisionGuard=ForbiddenOnlineGuard
    audit.OUT=Path(__file__).resolve().parent/'audit_hw/vbc_port_audit_20261029/fixed_base'
    audit.OUT.mkdir(parents=True,exist_ok=True);audit.main()
    p=audit.OUT/'summary.json';d=json.loads(p.read_text())
    d['conditions']='Fixed torso/nominal/gravity; expanded port_inner range; original VBC fixed-world-Z/yaw frame; online candidate IK/OBB forbidden.'
    d['pose_counter_definition']='Generated requests, not online pose-guard certification.'
    p.write_text(json.dumps(d,indent=2))
