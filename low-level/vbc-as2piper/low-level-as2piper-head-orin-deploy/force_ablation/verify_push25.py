"""Isolated actual push-function test; never creates a simulation or learner."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,json,torch
r=Path(__file__).resolve().parents[1]
before=ast.parse((r/'force_ablation/before_push25_as2_config.py.txt').read_text())
after=ast.parse((r/'legged_gym/envs/manip_loco/as2_head_orin_config.py').read_text())
old_value=next(n.value for n in ast.walk(before) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='standing_push_multiplier' for t in n.targets))
restored=copy.deepcopy(after);found=0
for n in ast.walk(restored):
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='standing_push_multiplier' for t in n.targets):
        assert ast.literal_eval(n.value)==2.5;n.value=copy.deepcopy(old_value);found+=1
assert found==1 and ast.dump(restored,include_attributes=False)==ast.dump(before,include_attributes=False)
source=ast.parse((r/'legged_gym/envs/manip_loco/manip_loco.py').read_text())
fn=next(f for c in source.body if isinstance(c,ast.ClassDef) and c.name=='ManipLoco' for f in c.body if isinstance(f,ast.FunctionDef) and f.name=='_push_robots')
# Fix sampled velocities to avoid TorchScript random-generator warmup differences.
sampled=torch.tensor([[.1,-.2],[.2,.1],[-.2,.1],[.1,.2]])
namespace={'torch':torch,'torch_rand_float':lambda *args,**kwargs:sampled.clone(),'gymtorch':SimpleNamespace(unwrap_tensor=lambda x:x)}
module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,'actual_push_function','exec'),namespace)
class Gym:
    def set_actor_root_state_tensor(self,*args):pass
outputs=[]
for multiplier in [1.,2.5]:
    state=torch.zeros(4,13)
    shadow=SimpleNamespace(cfg=SimpleNamespace(domain_rand=SimpleNamespace(max_push_vel_xy=.39639,standing_push_multiplier=multiplier)),root_states=state,_root_states=state,commands=torch.tensor([[0.,0.,0.],[.3,0.,0.],[0.,0.,.6],[.3,0.,.6]]),num_envs=4,device='cpu',common_step_counter=0,gym=Gym(),sim=None)
    namespace['_push_robots'](shadow);outputs.append(state[:,7:9].clone())
assert torch.allclose(outputs[1][0],outputs[0][0]*2.5) and torch.equal(outputs[1][1:],outputs[0][1:])
result={'standing_push_multiplier':2.5,'only_config_change':'domain_rand.standing_push_multiplier1->2.5','moving_push_unchanged':True,'test':'actual push function with fixed sampled velocities; no simulation/optimizer','stop_delete_restart_requires_root':True}
(r/'force_ablation/push25_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
