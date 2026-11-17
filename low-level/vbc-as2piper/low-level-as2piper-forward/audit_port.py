"""Read-only source/reward checks and a review diff for this isolated port."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'low-level'
OUT=ROOT/'audit'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def method(path,name):
    tree=ast.parse(path.read_text())
    return next(ast.dump(x,include_attributes=False) for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==name)


def main():
    OUT.mkdir(exist_ok=True)
    env='legged_gym/envs/manip_loco/manip_loco.py'
    reward='legged_gym/envs/rewards/maniploco_rewards.py'
    checks={'reward_module_byte_identical':sha(SOURCE/reward)==sha(ROOT/reward),
            'reward_aggregation_ast_identical':method(SOURCE/env,'compute_reward')==method(ROOT/env,'compute_reward'),
            'baseline_config_byte_identical':sha(SOURCE/'legged_gym/envs/manip_loco/b1z1_config.py')==sha(ROOT/'legged_gym/envs/manip_loco/b1z1_config.py'),
            'ppo_copy_byte_identical':all(sha(p)==sha(ROOT/'rsl_rl_local'/p.relative_to(ROOT.parent/'third_party/rsl_rl')) for p in (ROOT.parent/'third_party/rsl_rl').rglob('*.py'))}
    assert all(checks.values()),checks
    hashes={str(p.relative_to(SOURCE)):sha(p) for p in (SOURCE/'legged_gym').rglob('*.py')}
    hashes['urdf/as2piper_umi_actuated_v7.urdf']=sha(SOURCE/'resources/robots/as2piper/urdf/as2piper_umi_actuated_v7.urdf')
    (OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    (OUT/'preservation.json').write_text(json.dumps(checks,indent=2))
    patch=[]
    for p in sorted((ROOT/'legged_gym').rglob('*.py')):
        old=SOURCE/p.relative_to(ROOT)
        a=[line+'\n' for line in old.read_text().splitlines()] if old.exists() else []
        b=[line+'\n' for line in p.read_text().splitlines()]
        patch.extend(difflib.unified_diff(a,b,fromfile='a/'+str(p.relative_to(ROOT)) if old.exists() else '/dev/null',tofile='b/'+str(p.relative_to(ROOT))))
    (OUT/'port.patch').write_text(''.join(patch))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
