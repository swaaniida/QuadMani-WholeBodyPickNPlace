"""Initialize newly varying arm inputs without masking observations or gradients."""
import copy,hashlib,json
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parent


def main():
    source=ROOT/'logs/as2piper-low/gait_pilot500_20261029/model_500.pt'
    d=torch.load(source,map_location='cpu');m=d['model_state_dict']
    trace=np.load(str(ROOT/'audit_hw/gait_eval500_history/trajectories.npz'))
    q=trace['q'].reshape(-1,20);root=trace['root'].reshape(-1,7)
    target=trace['target'].reshape(-1,3);R=Rotation.from_quat(root[:,3:7]).as_matrix()
    origin=root[:,:3]+np.einsum('nij,j->ni',R,[.005,0,.08202847821788371])
    local=np.einsum('nji,nj->ni',R,target-origin)
    ref=torch.zeros(71);ref[17:23]=torch.tensor(q[:,12:18].mean(0)-[0,1.1,-.55,0,0,0])
    ref[60:63]=torch.tensor(local.mean(0))
    cols=list(range(17,23))+list(range(35,41))+[60,61,62]
    prefixes=['actor.actor_backbone.0','actor.history_encoder.encoder.0','critic.critic_backbone.0']
    changes={}
    for prefix in prefixes:
        w=m[prefix+'.weight'];b=m[prefix+'.bias'];old=w.clone();old_b=b.clone()
        b+=w[:,cols]@ref[cols];w[:,cols]=0
        # Exact first-layer output preservation at the selected reference input.
        x=torch.randn(128,w.shape[1]);x[:,cols]=ref[cols]
        error=((x@old.T+old_b)-(x@w.T+b)).abs().max().item()
        assert error<2e-5,(prefix,error)
        changes[prefix]=dict(reference_output_max_error=error)
    # Do not let old Adam moments reintroduce the removed random sensitivity.
    for opt_key,names in [('optimizer_state_dict',list(m)),('history_optimizer_state_dict',[k for k in m if k.startswith('actor.history_encoder.')])]:
        opt=d[opt_key];ids=[i for g in opt['param_groups'] for i in g['params']]
        assert len(ids)==len(names)
        for name,i in zip(names,ids):
            if i not in opt['state']:continue
            state=opt['state'][i]
            assert state['exp_avg'].shape==m[name].shape,(name,i)
            if any(name==p+'.weight' for p in prefixes):
                for key in ['exp_avg','exp_avg_sq']:state[key][:,cols]=0
            if any(name==p+'.bias' for p in prefixes):
                for key in ['exp_avg','exp_avg_sq']:state[key].zero_()
    notes=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        initialized_input_columns=cols,reference_proprio=ref.tolist(),changed_layers=changes,
        observations_masked=False,gradient_masked=False,
        meaning='Arm/EE inputs remain observed and trainable. First-layer effect at the gait reference is preserved; responses away from that reference are deliberately reinitialized.')
    d['infos']=notes
    out=ROOT/'logs/as2piper-low/gait_arm_transfer500_20261029';out.mkdir(parents=True,exist_ok=True)
    torch.save(d,out/'model_500.pt');(out/'transfer_initialization.json').write_text(json.dumps(notes,indent=2))
    print(json.dumps(notes['changed_layers'],indent=2))


if __name__=='__main__':main()
