"""CPU strict-load/history/JIT check only. Requires installed training dependencies."""
from pathlib import Path
import argparse,sys,json
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['deploy70','baseline200'],default='deploy70');p.add_argument('--checkpoint',type=Path);args=p.parse_args()
variant='low-level-as2piper-head-orin-'+('deploy' if args.variant=='deploy70' else 'nocontact');runtime=root/variant
sys.path[:0]=[str(runtime),str(runtime/'rsl_rl_local'),str(root/'third_party/isaacgym/python')]
import isaacgym
import torch
from force_ablation.run import build_model,compare_settings
from force_ablation.configs import configs
from legged_gym.utils.helpers import HistoryPolicyExporter
compare_settings(1);cfg,train=configs('B')
if args.checkpoint is None:
 name='as2piper_headpiper_orin_70N_20261115T133137Z' if args.variant=='deploy70' else 'as2piper_headpiper_orin_200N_0.18m'
 checkpoint=runtime/'experiments/contact_force_ablation/runs'/name/('model_20000.pt' if args.variant=='deploy70' else 'model_45000.pt')
else:checkpoint=args.checkpoint.resolve()
model=build_model(train);model.load_state_dict(torch.load(str(checkpoint),map_location='cpu')['model_state_dict'],strict=True);model.eval();obs=torch.zeros(4,755)
with torch.no_grad():
 action=model.act_inference(obs,True);jit=torch.jit.script(HistoryPolicyExporter(model.actor));output=jit(torch.cat([obs[:,:67],obs[:,85:]],-1))
 assert action.shape==(4,18) and torch.isfinite(action).all() and torch.allclose(action,output,atol=1e-6)
print(json.dumps({'checkpoint':str(checkpoint),'threshold':cfg.rewards.max_contact_force,'strict_load':True,'history_inference':True,'JIT_737_parity':True,'optimizer_steps':0},indent=2))
