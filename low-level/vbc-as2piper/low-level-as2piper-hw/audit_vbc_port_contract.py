"""Record every config difference and verify restored VBC method contracts."""
import ast,hashlib,json
from pathlib import Path
import isaacgym
import torch
from isaacgym.torch_utils import sphere2cart,quat_apply
from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg,B1Z1RoughCfgPPO
from legged_gym.envs.manip_loco.as2piper_vbc_port import AS2PiPERVBCPort,AS2PiPERVBCPortCfg,AS2PiPERVBCPortCfgPPO
from legged_gym.envs.manip_loco.as2piper_forward import AS2PiPERForward
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.utils.helpers import class_to_dict

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent/'vbc-legacy'
OUT=ROOT/'audit_hw/vbc_port_audit_20261029'


def differences(a,b,prefix=''):
    rows=[]
    for key in sorted(set(a)|set(b)):
        path=prefix+'.'+key if prefix else key;x=a.get(key);y=b.get(key)
        if hasattr(x,'tolist'):x=x.tolist()
        if hasattr(y,'tolist'):y=y.tolist()
        if isinstance(x,dict) and isinstance(y,dict):rows.extend(differences(x,y,path))
        elif x!=y:rows.append(dict(path=path,before=x,after=y))
    return rows


def main():
    cfg=AS2PiPERVBCPortCfg();ppo=AS2PiPERVBCPortCfgPPO()
    b=B1Z1RoughCfg();bp=B1Z1RoughCfgPPO()
    c=class_to_dict(cfg);p=class_to_dict(ppo)
    assert c['rewards']['scales']==class_to_dict(b)['rewards']['scales']
    assert c['rewards']['arm_scales']==class_to_dict(b)['rewards']['arm_scales']
    assert c['terrain']==class_to_dict(b)['terrain']
    assert c['control']['action_scale']==class_to_dict(b)['control']['action_scale']
    assert c['init_state']['init_vel_perturb_range']==.1
    assert all(c['domain_rand'][k] for k in ['randomize_base_mass','randomize_base_com','randomize_gripper_mass','randomize_friction','randomize_motor','push_robots'])
    inherited=['_resample_commands','_reset_dofs','_get_ee_goal_spherical_center','_get_ee_goal_observation','_collision_check','_step_contact_targets']
    assert all(getattr(AS2PiPERVBCPort,k) is getattr(ManipLoco,k) for k in inherited)
    assert 'AS2PiPERHardware' not in [k.__name__ for k in AS2PiPERVBCPort.__mro__]
    def methods(path):
        t=ast.parse(path.read_text())
        return {f.name:ast.dump(f,include_attributes=False) for node in t.body if isinstance(node,ast.ClassDef) and node.name=='ManipLoco' for f in node.body if isinstance(f,ast.FunctionDef)}
    baseline=methods(BASE/'low-level/legged_gym/envs/manip_loco/manip_loco.py');current=methods(ROOT/'legged_gym/envs/manip_loco/manip_loco.py')
    original=['_resample_commands','_get_ee_goal_spherical_center','_collision_check','_step_contact_targets','_resample_ee_goal_sphere_once','_resample_ee_goal_orn_once']
    assert all(baseline[k]==current[k] for k in original)
    # Geometry-only counterexample: moving body height moves the previous goal
    # by the same amount, whereas VBC deliberately leaves goal world-Z fixed.
    fake=type('State',(),{})();fake.num_envs=3;fake.device='cpu'
    fake.root_states=torch.zeros(3,13);fake.root_states[:,2]=torch.tensor([.2744069,.3244069,.3744069]);fake.root_states[:,6]=1
    fake.base_yaw_quat=fake.root_states[:,3:7].clone()
    fake.arm_base_offset=torch.tensor(cfg.asset.arm_base_offset).repeat(3,1)
    gc=cfg.goal_ee.sphere_center
    fake.ee_goal_center_offset=torch.tensor([gc.x_offset,gc.y_offset,gc.z_invariant_offset]).repeat(3,1)
    v=sphere2cart(torch.tensor(cfg.goal_ee.ranges.init_pos_start)[None,:]).repeat(3,1)
    actual=fake.root_states[:,:3]+fake.arm_base_offset+v
    old_goal=AS2PiPERForward._get_ee_goal_spherical_center(fake)+v
    vbc_goal=ManipLoco._get_ee_goal_spherical_center(fake)+v
    frames=dict(root_z=fake.root_states[:,2].tolist(),previous_goal_z=old_goal[:,2].tolist(),
        vbc_goal_z=vbc_goal[:,2].tolist(),previous_ee_z_error=(actual-old_goal)[:,2].tolist(),
        vbc_ee_z_error=(actual-vbc_goal)[:,2].tolist())
    old_run=ROOT/'logs/as2piper-low/as2piper_hw_gaitfix_vbcsampler_scratch_20261029_134701/runtime_snapshot.json'
    old=json.loads(old_run.read_text())
    result=dict(environment_vs_b1=differences(class_to_dict(b),c),ppo_vs_b1=differences(class_to_dict(bp),p),
        environment_vs_current_run=differences(old['environment'],c),
        inherited_methods=inherited,baseline_ast_equal=original,frame_height_counterexample=frames,
        non_original_reward_formula='Contact-shaped force/velocity raw sign correction only; weights unchanged.',
        original_reset_clamp_difference='Shared AS2 base clamps URDF limits and fixes finger targets; multiplicative +/-20% all remaining DOFs retained.')
    (OUT/'config_contract.json').write_text(json.dumps(result,indent=2))
    print('CONTRACT PASS; environment differences',len(result['environment_vs_b1']),'PPO differences',result['ppo_vs_b1'])
    print('FRAME',json.dumps(frames))


if __name__=='__main__':main()
