"""Audit current reward presets against saved, actually executed configurations."""
import isaacgym
import json
from pathlib import Path
from legged_gym.envs.manip_loco.as2piper_gait_study import AS2PiPERGaitStudyCfg,configure
from legged_gym.utils.helpers import class_to_dict
ROOT=Path(__file__).resolve().parent

def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        result=[]
        for key in sorted(set(a)|set(b)):
            if key not in a or key not in b:result.append((path+'.'+key,a.get(key),b.get(key)))
            else:result.extend(differences(a[key],b[key],path+'.'+key))
        return result
    return [] if a==b else [(path,a,b)]

results={}
for name in ['stable_proxy','progress_gain2_proxy','stable_proxy_ground']:
    cfg=configure(AS2PiPERGaitStudyCfg(),name)
    current=json.loads(json.dumps(class_to_dict(cfg),default=lambda x:x.tolist()))
    executed=json.loads((ROOT/'logs/as2piper-low'/('gait_reward_'+name+'_s7_v2')/'runtime_snapshot.json').read_text())['environment']
    current['env']['num_envs']=6144;current['seed']=7
    rows=differences(current,executed)
    expected={'.env.num_observations':(744,799),'.env.num_proprio':(66,71),'.terrain.num_sub_terrains':(None,200)}
    unexpected=[row for row in rows if row[0] not in expected or tuple(row[1:])!=expected[row[0]]]
    assert not unexpected,(name,unexpected)
    results[name]={'unexpected_configuration_differences':unexpected,'derived_runtime_fields':rows}
a=class_to_dict(configure(AS2PiPERGaitStudyCfg(),'progress_gain2_proxy'))
b=class_to_dict(configure(AS2PiPERGaitStudyCfg(),'progress_gain2_proxy_ground'))
a=json.loads(json.dumps(a,default=lambda x:x.tolist()));b=json.loads(json.dumps(b,default=lambda x:x.tolist()))
rows=differences(a,b)
assert {r[0] for r in rows}=={'.study.preset','.study.terrain_height_mode'},rows
results['combined_candidate_only_height_reference_changes']=rows
c=class_to_dict(configure(AS2PiPERGaitStudyCfg(),'progress_gain2_proxy_yaw4'))
c=json.loads(json.dumps(c,default=lambda x:x.tolist()))
yaw_rows=differences(a,c)
assert {r[0] for r in yaw_rows}=={'.study.preset','.rewards.scales.tracking_ang_vel'},yaw_rows
assert c['rewards']['scales']['tracking_ang_vel']==4.0
results['yaw4_candidate_only_yaw_weight_changes']=yaw_rows
floor_cfg=class_to_dict(configure(AS2PiPERGaitStudyCfg(),'progress_gain2_proxy_ground_floor'))
floor_cfg=json.loads(json.dumps(floor_cfg,default=lambda x:x.tolist()))
floor_rows=differences(a,floor_cfg)
assert {r[0] for r in floor_rows}=={'.study.preset','.study.terrain_height_mode','.study.penalize_excess_swing'},floor_rows
results['ground_floor_candidate_reward_group_changes']=floor_rows
(ROOT/'experiments/gait_reward_proxy_v2/current_config_validation.json').write_text(json.dumps(results,indent=2)+'\n')
print('Executed presets unchanged; planned ground/yaw branches each change only their declared reward setting.')

# Contract: higher swing is allowed on rough ground; stance and plane stay strict.
import torch
from types import SimpleNamespace
from legged_gym.envs.manip_loco.as2piper_gait_study import AS2PiPERGaitStudy
cfg=configure(AS2PiPERGaitStudyCfg(),'progress_gain2_proxy_ground_floor')
e=SimpleNamespace(cfg=cfg,foot_indices=torch.full((3,4),.75),feet_indices=torch.arange(4),rigid_body_state=torch.zeros(3,4,13))
e._get_walking_cmd_mask=lambda:torch.ones(3,dtype=torch.bool)
e._foot_ground_height=lambda mode=None:torch.zeros(3,4)
e.rigid_body_state[:,:,2]=torch.tensor([.028+.04,.028+.08,.028])[:,None]
r,_=AS2PiPERGaitStudy._phase_clearance(e)
assert torch.allclose(r[:2],torch.ones(2)) and r[2]<.1
# During stance a lifted foot must still lose height reward.
e.foot_indices.fill_(.25)
r,_=AS2PiPERGaitStudy._phase_clearance(e)
assert r[0]<.1 and torch.isclose(r[2],torch.tensor(1.))
# Preserve nominal-plane Gaussian, including overshoot cost.
e.foot_indices.fill_(.75);cfg.terrain.mesh_type='plane'
r,_=AS2PiPERGaitStudy._phase_clearance(e)
assert torch.isclose(r[0],torch.tensor(1.)) and r[1]<.1
print('Rough-swing, stance-contact and nominal-plane height reward contracts passed.')

# Smooth pair differs only by the previously declared height reward group.
d=class_to_dict(configure(AS2PiPERGaitStudyCfg(),'progress_gain2_proxy_smooth_yaw4'))
f=class_to_dict(configure(AS2PiPERGaitStudyCfg(),'progress_gain2_proxy_smooth_ground_floor_yaw4'))
d=json.loads(json.dumps(d,default=lambda x:x.tolist()));f=json.loads(json.dumps(f,default=lambda x:x.tolist()))
assert {r[0] for r in differences(c,d)}=={'.study.preset','.study.smooth_progress_mix'}
assert {r[0] for r in differences(d,f)}=={'.study.preset','.study.terrain_height_mode','.study.penalize_excess_swing'}
print('Fresh smooth pair: common yaw4/progress settings; only height reward group differs.')

matched=configure(AS2PiPERGaitStudyCfg(),'progress_gain2_proxy_smooth_yaw4_contactmatch')
matched_dict=json.loads(json.dumps(class_to_dict(matched),default=lambda x:x.tolist()))
assert {r[0] for r in differences(d,matched_dict)}=={'.study.preset','.study.symmetric_contact'}
# Contact contract: matched diagonal pair beats one-foot or no-foot support.
target=torch.tensor([[1.,0.,0.,1.]]).repeat(5,1)
presence=torch.tensor([[1,0,0,1],[0,0,0,1],[0,0,0,0],[1,1,1,1],[0,1,1,0]],dtype=torch.bool)
e=SimpleNamespace(cfg=matched,desired_contact_states=target,foot_contacts_from_sensor=presence)
reward,metric=AS2PiPERGaitStudy._proxy_contact_force(e)
np_expected=torch.tensor([0.,.25,.5,.5,1.])
assert torch.allclose(reward,np_expected) and torch.allclose(metric,-np_expected)
e.desired_contact_states.fill_(1.)
e.foot_contacts_from_sensor=torch.ones(5,4,dtype=torch.bool)
assert torch.allclose(AS2PiPERGaitStudy._proxy_contact_force(e)[0],torch.zeros(5))
print('Fresh contact pair differs only by stance-missing-contact penalty; contact contracts passed.')

foot_cfg=configure(AS2PiPERGaitStudyCfg(),'progress_gain2_proxy_smooth_yaw4_footground')
foot_dict=json.loads(json.dumps(class_to_dict(foot_cfg),default=lambda x:x.tolist()))
assert {r[0] for r in differences(d,foot_dict)}=={'.study.preset','.study.foot_height_reference'}
# Real supporting surface .05m above conservative grid; body objective unchanged.
def foot_case(cfg):
    cfg.terrain.mesh_type='trimesh'
    fake=SimpleNamespace(cfg=cfg,foot_indices=torch.full((1,4),.75),feet_indices=torch.arange(4),rigid_body_state=torch.zeros(1,4,13),_support_ground=torch.full((1,4),.05),_body_ground=torch.tensor([.05]),height_samples=torch.zeros(2,2),root_states=torch.zeros(1,13))
    fake.terrain=SimpleNamespace(cfg=SimpleNamespace(border_size=0.,horizontal_scale=1.,vertical_scale=1.))
    fake.rigid_body_state[:,:,2]=.05+.028+.04;fake.root_states[:,2]=.35
    fake._get_walking_cmd_mask=lambda:torch.ones(1,dtype=torch.bool)
    fake._foot_ground_height=lambda mode=None:AS2PiPERGaitStudy._foot_ground_height(fake,mode)
    return fake
old=foot_case(configure(AS2PiPERGaitStudyCfg(),'progress_gain2_proxy_smooth_yaw4'));new=foot_case(foot_cfg)
assert AS2PiPERGaitStudy._phase_clearance(old)[0].item()<.1
assert torch.isclose(AS2PiPERGaitStudy._phase_clearance(new)[0],torch.ones(1)).all()
assert torch.equal(AS2PiPERGaitStudy._terrain_relative_base_height(old)[0],AS2PiPERGaitStudy._terrain_relative_base_height(new)[0])
print('Foot-only support reference contract passed; body height objective unchanged.')

dead_cfg=configure(AS2PiPERGaitStudyCfg(),'progress_gain2_proxy_smooth_yaw4_footground_deadband')
dead_dict=json.loads(json.dumps(class_to_dict(dead_cfg),default=lambda x:x.tolist()))
assert {r[0] for r in differences(foot_dict,dead_dict)}=={'.study.preset','.study.vbc_linear_deadband'}
raw_command=torch.tensor([[.1,0.,.6],[.1,0.,.6]])
state=torch.tensor([[0.,0.,0.],[.1,0.,0.]])
a=SimpleNamespace(cfg=configure(AS2PiPERGaitStudyCfg(),'progress_gain2_proxy_smooth_yaw4_footground'),commands=raw_command.clone(),base_lin_vel=state)
b=SimpleNamespace(cfg=dead_cfg,commands=raw_command.clone(),base_lin_vel=state)
old_score=AS2PiPERGaitStudy._speed_tracking(a)[0];new_score=AS2PiPERGaitStudy._speed_tracking(b)[0]
from legged_gym.envs.rewards.maniploco_rewards import ManipLoco_rewards
original_score=ManipLoco_rewards._reward_tracking_lin_vel_max(SimpleNamespace(env=b))[0]
assert old_score[0]<old_score[1] and new_score[0]>new_score[1] and original_score[0]>original_score[1]
assert torch.equal(raw_command,b.commands)
a.commands[:,0]=.25;b.commands[:,0]=.25
assert torch.equal(AS2PiPERGaitStudy._speed_tracking(a)[0],AS2PiPERGaitStudy._speed_tracking(b)[0])
print('Original VBC low-linear-command intent restored; sampler/observations unchanged, moderate-command reward identical.')

quiet_cfg=configure(AS2PiPERGaitStudyCfg(),'progress_gain2_proxy_smooth_yaw4_contactmatch_deadband')
quiet_dict=json.loads(json.dumps(class_to_dict(quiet_cfg),default=lambda x:x.tolist()))
assert {r[0] for r in differences(matched_dict,quiet_dict)}=={'.study.preset','.study.vbc_linear_deadband'}
assert quiet_cfg.study.terrain_height_mode=='grid' and not hasattr(quiet_cfg.study,'foot_height_reference')
print('Contactmatch/deadband pair changes only original VBC low-linear reward branch.')
