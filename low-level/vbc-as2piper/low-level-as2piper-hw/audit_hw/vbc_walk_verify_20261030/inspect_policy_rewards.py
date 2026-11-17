"""Instrument evaluation reward calls without changing their returned values."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import evaluate_hw
import numpy as np
from legged_gym.envs.manip_loco.as2piper_vbc_port import AS2PiPERVBCPort
original=AS2PiPERVBCPort._prepare_reward_function
records={};frames=[];names=[]
def prepare(self):
    original(self)
    names[:]=self.reward_names
    for i,name in enumerate(self.reward_names):
        fn=self.reward_functions[i]
        def capture(fn=fn,name=name):
            result=fn();val=result[0]
            if not hasattr(val,'clone'):val=self.commands[:,0]*0+val
            records.setdefault(name,[]).append(val.detach().clone()*self.reward_scales[name])
            return result
        self.reward_functions[i]=capture
    old=self.compute_reward
    def compute():
        frames.append(dict(cached_velocity=self.foot_velocities.detach().clone(),fresh_velocity=self.rigid_body_state[:,self.feet_indices,7:10].detach().clone(),cached_position=self.foot_positions.detach().clone(),fresh_position=self.rigid_body_state[:,self.feet_indices,:3].detach().clone(),sensor=self.force_sensor_tensor.detach().clone(),sensor_contact=self.foot_contacts_from_sensor.detach().clone(),net_force=self.contact_forces[:,self.feet_indices].detach().clone(),desired=self.desired_contact_states.detach().clone()))
        old()
    self.compute_reward=compute
AS2PiPERVBCPort._prepare_reward_function=prepare
if __name__=='__main__':
    evaluate_hw.main()
    out=Path(sys.argv[sys.argv.index('--output')+1])
    import torch,json
    data={name:torch.stack(items).cpu().numpy() for name,items in records.items()}
    data.update({key:torch.stack([f[key] for f in frames]).cpu().numpy() for key in frames[0]})
    np.savez_compressed(str(out/'reward_trace.npz'),**data)
    summary={}
    for k,g in enumerate(['stand','forward','turn','mixed']):
        summary[g]={name:float(data[name][100:,k*8:(k+1)*8].mean()) for name in names}
        sensor=data['sensor_contact'][100:,k*8:(k+1)*8];contact=data['net_force'][100:,k*8:(k+1)*8, :,2]>5
        summary[g]['sensor_vs_net_contact_agreement']=float((sensor==contact).mean())
        summary[g]['sensor_contact_fraction']=sensor.mean((0,1)).tolist()
        summary[g]['desired_contact_mean']=data['desired'][100:,k*8:(k+1)*8].mean((0,1)).tolist()
    (out/'reward_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
