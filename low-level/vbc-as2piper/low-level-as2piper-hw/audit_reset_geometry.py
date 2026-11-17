"""Compare original B1 and AS2 reset geometry without training or simulation."""
import json
from pathlib import Path
import isaacgym
import torch
from isaacgym.torch_utils import sphere2cart
from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg
from legged_gym.envs.manip_loco.as2piper_config import AS2PiPERPlateCfg
from legged_gym.utils.urdf_chain import URDFChain

ROOT = Path(__file__).resolve().parent


def stats(x):
    return dict(mean=float(x.mean()), p05=float(x.quantile(.05)), p50=float(x.median()),
                p95=float(x.quantile(.95)), min=float(x.min()), max=float(x.max()))


def main():
    torch.manual_seed(20261029)
    result = {}
    for label, cfg, asset, base, radius, spread in [
        ('b1_original', B1Z1RoughCfg(), ROOT.parent/'low-level/resources/robots/b1z1/urdf/b1z1_workspace_study.urdf', 'base', .04, .2),
        ('as2_original_reset', AS2PiPERPlateCfg(), ROOT/'resources/robots/as2piper/urdf/as2piper_plate_forward_training.urdf', 'base_link', .028, .2),
        ('as2_restricted_reset', AS2PiPERPlateCfg(), ROOT/'resources/robots/as2piper/urdf/as2piper_plate_forward_training.urdf', 'base_link', .028, .05),
    ]:
        n = 10000
        feet = []
        for leg in ['FL', 'FR', 'RL', 'RR']:
            names = [leg+'_'+j+'_joint' for j in ['hip', 'thigh', 'calf']]
            home = torch.tensor([cfg.init_state.default_joint_angles[j] for j in names])
            q = home*(1-spread+2*spread*torch.rand(n, 3))
            feet.append(URDFChain(asset, base, leg+'_foot', names)(q)[0][:, 2]-radius)
        low = torch.stack(feet, -1).min(-1).values
        initial_height = cfg.init_state.pos[2]
        bottom = initial_height+low
        if label.startswith('b1'):
            names=['z1_waist','z1_shoulder','z1_elbow','z1_wrist_angle','z1_forearm_roll','z1_wrist_rotate','z1_jointGripper']
            tip='ee_gripper_link';center=torch.tensor([.3, 0, .7])
        else:
            names=['piper_joint'+str(i) for i in range(1, 7)]
            tip='piper_umi_nominal_tcp';center=torch.tensor([.005, 0, .3244069024246581+.08202847821788371])
        home = torch.tensor([cfg.init_state.default_joint_angles[j] for j in names])
        q = home.repeat(n, 1)
        if label != 'as2_restricted_reset':q *= .8+.4*torch.rand(n, len(names))
        actual, _ = URDFChain(asset, base, tip, names)(q)
        actual[:, 2] += initial_height
        sphere = torch.tensor(cfg.goal_ee.ranges.init_pos_start)[None, :]
        desired = center+sphere2cart(sphere)[0]
        error = (actual-desired).norm(dim=-1)
        result[label] = dict(samples=n, initial_height_m=initial_height,
            minimum_foot_bottom_m=stats(bottom), penetration_gt_1cm_fraction=float((bottom<-.01).float().mean()),
            fk_snap_height_m=stats(-low+.002), initial_ee_error_m=stats(error))
    out=ROOT/'audit_hw/vbc_port_audit_20261029/reset_geometry.json'
    out.write_text(json.dumps(result, indent=2));print(json.dumps(result, indent=2))


if __name__ == '__main__':main()
