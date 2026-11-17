"""Validate B1 algorithm preservation and AS2 spherical pose collision rejection."""
import ast,hashlib,importlib,json,sys,types
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mujoco_mount_compare'))
from mount_workspace_study import setup,self_collision
pkg=types.ModuleType('forward_utils');pkg.__path__=[str(ROOT/'legged_gym/utils')];sys.modules['forward_utils']=pkg
Guard=importlib.import_module('forward_utils.arm_collision').ArmCollisionGuard


def method(path,name):
    return next(ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)


def main():
    out=ROOT/'audit_forward';old=ROOT.parent/'low-level-as2piper'
    baseline=old/'legged_gym/envs/manip_loco/manip_loco.py';current=ROOT/'legged_gym/envs/manip_loco/manip_loco.py'
    identical={n:method(baseline,n)==method(current,n) for n in ['_resample_ee_goal_sphere_once','_resample_ee_goal_orn_once','_resample_ee_goal','_update_curr_ee_goal','_collision_check','compute_reward']}
    assert all(identical.values()),identical
    subclass=ast.parse((ROOT/'legged_gym/envs/manip_loco/as2piper_forward.py').read_text())
    assert not {'_resample_ee_goal_sphere_once','_resample_ee_goal_orn_once','_update_curr_ee_goal'}&{n.name for n in ast.walk(subclass) if isinstance(n,ast.FunctionDef)}
    hashes=json.loads((out/'source_before.json').read_text());assert all(hashlib.sha256((old/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    tree=ast.parse((ROOT/'legged_gym/envs/manip_loco/as2piper_config.py').read_text())
    goal=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='goal_ee')
    ranges=next(n for n in goal.body if isinstance(n,ast.ClassDef) and n.name=='ranges')
    values={n.targets[0].id:ast.literal_eval(n.value) for n in ranges.body if isinstance(n,ast.Assign)}
    params={n.targets[0].id:ast.literal_eval(n.value) for n in goal.body if isinstance(n,ast.Assign)}
    asset=ROOT/'resources/robots/as2piper/urdf/as2piper_plate_forward_training.urdf'
    guard=Guard(asset,ROOT/'resources/commands/arm_collision.json','cpu')
    m=mujoco.MjModel.from_xml_path(str(ROOT/'resources/commands/plate_forward.xml'));d,names,_,_=setup(m,'plate_only_forward')
    neutral=d.qpos.copy();ids=[int(m.joint(n).qposadr) for n in names];base=m.body('piper_base_link').id;tcp=m.body('piper_umi_nominal_tcp').id
    rng=np.random.default_rng(20261029);count=1200
    sphere=np.stack([rng.uniform(*values[k],count) for k in ['pos_l','pos_p','pos_y']],axis=-1)
    delta=np.stack([rng.uniform(*values[k],count) for k in ['delta_orn_r','delta_orn_p','delta_orn_y']],axis=-1)
    home=np.array(values['init_pos_start']);q=torch.tensor(neutral[ids],dtype=torch.float).repeat(count,1)
    rejected=torch.zeros(count,dtype=torch.bool);ik_failed=rejected.clone();collided=rejected.clone();points=[]
    for t in (0.,.5,1.):
        s=home+(sphere-home)*t;r,p,y=s.T
        target=np.stack([r*np.cos(p)*np.cos(y),r*np.cos(p)*np.sin(y),r*np.sin(p)],axis=-1)
        rot=Rotation.from_euler('xyz',np.stack([delta[:,0],-p+params['arm_induced_pitch']+delta[:,1],y+delta[:,2]],axis=-1)).as_matrix()
        q,failed=guard.solve(torch.tensor(target,dtype=torch.float),torch.tensor(rot,dtype=torch.float),q)
        collision=guard.collides(q,torch.tensor([.005,0,.0820284782]))
        ik_failed|=failed;collided|=collision;rejected|=failed|collision;points.append(target)
    accepted=(~rejected).numpy();exact_collisions=0;max_error=0.
    for i in np.where(accepted)[0]:
        d.qpos[:]=neutral;d.qpos[ids]=q[i].numpy();mujoco.mj_forward(m,d)
        exact_collisions+=int(self_collision(m,d));max_error=max(max_error,float(np.linalg.norm(d.xpos[tcp]-d.xpos[base]-target[i])))
    assert accepted.sum()>count*.5, 'Excessive rejection; revisit ranges/solver'
    assert exact_collisions==0,exact_collisions
    p,R,_,_,_=guard.fk(torch.tensor(neutral[ids],dtype=torch.float)[None,:])
    d.qpos[:]=neutral;mujoco.mj_forward(m,d)
    home_error=float(np.linalg.norm(p[0].numpy()-(d.xpos[tcp]-d.xpos[base])))
    # A gripper placed at the AS2 head center must fail the OBB SAT test.
    module=importlib.import_module('forward_utils.arm_collision')
    assert bool(module.box_collision(torch.tensor([[[.28,0,.08]]]),torch.eye(3)[None,None],torch.tensor([[.02,.02,.02]]),guard.obstacle_center,guard.obstacle_half)[0])
    result=dict(b1_method_ast_identical=identical,original_files_unchanged=len(hashes),proposal_count=count,
        accepted_by_guard=int(accepted.sum()),rejected_ik=int(ik_failed.sum()),rejected_body_head=int(collided.sum()),
        accepted_mujoco_endpoint_collisions=exact_collisions,max_accepted_position_error_m=max_error,
        fast_fk_home_error_m=home_error,spherical_ranges=values,
        collision_guard='Arm/gripper OBBs versus AS2 body/head AABB; 2mm/5mm padding; 3 arm path samples, original 10 EE samples',
        limitations='Finite nominal tests; arm self/leg/cable/object collision and continuous path clearance are not guaranteed.')
    (out/'validation.json').write_text(json.dumps(result,indent=2))
    fig=plt.figure(figsize=(12,5));ax=fig.add_subplot(121,projection='3d')
    ax.scatter(*points[-1][accepted].T,s=4,alpha=.3,label='Accepted');ax.scatter(*points[-1][~accepted].T,s=6,alpha=.4,color='red',label='Rejected')
    ax.set_xlabel('Arm-origin yaw x [m]');ax.set_ylabel('y [m]');ax.set_zlabel('z [m]');ax.legend();ax.set_title('Independent B1 spherical proposals')
    ax=fig.add_subplot(122);p=points[-1];ax.scatter(p[accepted,0]+.005,p[accepted,2]+.0820285,s=4,alpha=.3)
    ax.scatter(p[~accepted,0]+.005,p[~accepted,2]+.0820285,s=6,alpha=.4,color='red')
    from matplotlib.patches import Rectangle
    c=guard.obstacle_center[0].numpy();h=guard.obstacle_half[0].numpy();ax.add_patch(Rectangle((c[0]-h[0],c[2]-h[2]),2*h[0],2*h[2],color='gray',alpha=.3,label='Body/head keep-out'))
    ax.set_xlabel('AS2 body x [m]');ax.set_ylabel('AS2 body z [m]');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('B1 sampling/interpolation retained; AS2 ranges + collision rejection');fig.tight_layout();fig.savefig(str(out/'command_workspace.png'),dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
