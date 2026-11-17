"""Evaluation-only world-frame EE trajectory; compare four saved policies in one PhysX scene."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from .run import environment, build_model, source_hashes, model_hash
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_rotate_inverse, quat_apply, cart2sphere
import numpy as np
import torch

# Time, world xyz relative to nominal initial AS2 base xy / ground z, world rpy.
# 11--60s concentrates on ground approach, contact goals, sweeping and small lifts.
KNOTS = [
    (0, [.6148, 0, .6335], [0, 2.03353, 0], '초기 자세'),
    (3, [.6148, 0, .6335], [0, 2.03353, 0], '정지 안정화'),
    (6, [.52, 0, .80], [0, 1.70, 0], '상단 접근'),
    (9, [.72, 0, .43], [0, 2.25, 0], '전방 확장'),
    (12, [.60, 0, .16], [0, 2.80, 0], '바닥 접근'),
    (15, [.60, 0, .01], [0, 3.05, 0], '바닥 접촉 목표'),
    (18, [.55, .18, .02], [.12, 2.90, .35], '바닥 왼쪽'),
    (21, [.55, -.18, .02], [-.12, 2.90, -.35], '바닥 오른쪽'),
    (24, [.72, 0, .02], [0, 2.80, 0], '바닥 먼 위치'),
    (27, [.45, 0, .02], [0, 3.0, 0], '바닥 가까운 위치'),
    (30, [.50, .12, .13], [.10, 2.65, .25], '살짝 들어올리기'),
    (33, [.50, .12, .01], [.10, 3.0, .25], '왼쪽 내려놓기'),
    (36, [.68, -.15, .03], [-.10, 2.85, -.25], '바닥 대각선 이동'),
    (39, [.66, -.12, .16], [0, 2.70, -.30], '오른쪽 들어올리기'),
    (42, [.66, -.12, .01], [0, 3.0, -.30], '오른쪽 내려놓기'),
    (45, [.56, 0, .02], [0, 3.0, 0], '바닥 중앙'),
    (48, [.48, -.18, .04], [-.12, 2.85, -.35], '가까운 오른쪽'),
    (51, [.69, .16, .04], [.12, 2.85, .35], '먼 왼쪽'),
    (54, [.58, 0, .01], [0, 3.05, 0], '중앙 접촉 목표'),
    (57, [.58, 0, .01], [0, 3.05, 0], '바닥 유지'),
    (60, [.58, 0, .16], [0, 2.80, 0], '바닥에서 들어올리기'),
    (64, [.6148, 0, .6335], [0, 2.03353, 0], '초기 자세 복귀'),
]


def trajectory(t):
    j = next((j for j in range(1, len(KNOTS)) if t <= KNOTS[j][0]), len(KNOTS)-1)
    lo, hi = KNOTS[j-1], KNOTS[j]
    u = float(np.clip((t-lo[0])/(hi[0]-lo[0]), 0, 1))
    u = u*u*u*(10-15*u+6*u*u)
    return ((1-u)*np.array(lo[1])+u*np.array(hi[1]),
            (1-u)*np.array(lo[2])+u*np.array(hi[2]), j)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    run = Path(a.run).resolve()
    saved = json.loads((run/'manifest.json').read_text())
    assert saved['source_sha256'] == source_hashes()
    env, train = environment('B', 1, 'cuda:0', small=True, terrain='flat')
    try:
        iterations = [10000, 20000, 30000, 45000]
        models = []
        hashes = []
        checkpoints = []
        for it in iterations:
            ckpt = run/('model_%d.pt' % it)
            model = build_model(train, env.device)
            model.load_state_dict(torch.load(str(ckpt), map_location=env.device)['model_state_dict'], strict=True)
            model.eval()
            models.append(model)
            hashes.append(model_hash(model))
            checkpoints.append({'iteration':it, 'path':str(ckpt), 'sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest()})
        # Evaluation overrides only: identical initial posture and no ordinary 10s timeout cuts.
        env.cfg.init_state.origin_perturb_range = 0.
        env.cfg.init_state.rand_yaw_range = 0.
        env.cfg.init_state.init_vel_perturb_range = 0.
        env.cfg.env.episode_length_s = env.max_episode_length_s = 90.
        env.max_episode_length = int(math.ceil(90/env.dt))
        old_reset = env._reset_dofs
        def exact_reset(ids):
            old_reset(ids)
            env.dof_pos[ids] = env.default_dof_pos
            env.dof_vel[ids] = 0.
            env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
        env._reset_dofs = exact_reset
        env._resample_commands = lambda ids: env.commands.__setitem__(ids, 0.)
        env.cfg.env.teleop_mode = True
        env.commands.zero_()
        # reset() performs a physics step. Hold the arm during that one step:
        # cached EE/Jacobian tensors can otherwise still refer to the pre-reset pose.
        old_ik = env._control_ik
        env._control_ik = lambda dpose: torch.zeros(env.num_envs, 6, device=env.device)
        env.reset()
        env._control_ik = old_ik
        env.global_steps = 45000*train.runner.num_steps_per_env  # Same control-delay branch for all.
        initial_q = env.dof_pos.clone()
        initial_root = env.root_states[:, :7].clone()
        initial_root[:, :3] -= env.env_origins
        q_delta = (initial_q-initial_q[0]).abs().max().item()
        root_delta = (initial_root-initial_root[0]).abs().max().item()
        # PhysX's first post-reset step has per-environment solver variation.
        # Preserve and report it, rather than claiming bitwise identical states.
        origin = env.env_origins.clone()
        np.savez_compressed(out/'initial_states.npz',q=initial_q.cpu().numpy(),root_local=initial_root.cpu().numpy())
        clock = {'time':0.}
        def apply_target():
            xyz, rpy, _ = trajectory(clock['time'])
            env.curr_ee_goal_cart_world = origin + torch.tensor(xyz, device=env.device, dtype=torch.float)
            relative = env.curr_ee_goal_cart_world-env._get_ee_goal_spherical_center()
            env.curr_ee_goal_cart[:] = quat_rotate_inverse(env.base_yaw_quat, relative)
            env.curr_ee_goal_sphere[:] = cart2sphere(env.curr_ee_goal_cart)
            r = torch.tensor(rpy, device=env.device, dtype=torch.float).repeat(env.num_envs, 1)
            env.ee_goal_orn_quat = quat_from_euler_xyz(r[:,0],r[:,1],r[:,2])
            env.ee_goal_orn_delta_rpy.zero_()
        env._update_curr_ee_goal = apply_target
        apply_target()
        env.compute_observations()
        ids = [0,8,16,24]
        records = {k:[] for k in ['time','root','q','target','ee','target_quat','ee_quat','velocity','yaw','done','phase','ee_force','finger_force','position_error','orientation_error_deg']}
        finger_ids = [i for i,n in enumerate(env.body_names) if 'gripper' in n and i!=env.gripper_idx]
        errors = []
        reset_events = []
        with torch.no_grad():
            for i in range(int(64/env.dt)):
                clock['time'] = i*env.dt
                apply_target()
                env.commands.zero_()
                # Refresh only the command part; avoid advancing observation history twice.
                arm_base = env.base_pos + quat_apply(env.base_quat, env.arm_base_offset)
                env.obs_buf[:,56:59] = quat_rotate_inverse(env.base_quat, env.curr_ee_goal_cart_world-arm_base)
                actions = torch.cat([model.act_inference(env.obs_buf[j*8:(j+1)*8], True) for j,model in enumerate(models)], dim=0)
                env.step(actions)
                assert env.commands.abs().max()==0
                assert torch.isfinite(env.obs_buf).all()
                pos = (env.curr_ee_goal_cart_world-env.ee_pos).norm(dim=-1)
                dot = (env.ee_goal_orn_quat*env.ee_orn).sum(-1).abs().clamp(0,1)
                angle = 2*torch.acos(dot)*180/np.pi
                errors.append(torch.stack([pos, angle, env.reset_buf.float(), env.time_out_buf.float(),env.ee_pos[:,2]-origin[:,2]],dim=-1).cpu().numpy())
                if env.reset_buf.any():
                    for eid in env.reset_buf.nonzero().flatten().cpu().tolist():
                        reset_events.append({'time':(i+1)*env.dt,'env':eid,'iteration':iterations[eid//8],'timeout':bool(env.time_out_buf[eid])})
                if i%2==1:
                    root = env.root_states[ids,:7].clone()
                    root[:,:3] -= origin[ids]
                    data = {'time':(i+1)*env.dt,'root':root,'q':env.dof_pos[ids],
                            'target':env.curr_ee_goal_cart_world[ids]-origin[ids], 'ee':env.ee_pos[ids]-origin[ids],
                            'target_quat':env.ee_goal_orn_quat[ids], 'ee_quat':env.ee_orn[ids],
                            'velocity':env.base_lin_vel[ids,0], 'yaw':env.base_ang_vel[ids,2],
                            'done':env.reset_buf[ids], 'phase':trajectory(clock['time'])[2],
                            'ee_force':env.contact_forces[ids,env.gripper_idx].norm(dim=-1),
                            'finger_force':env.contact_forces[ids][:,finger_ids].norm(dim=-1).sum(-1) if finger_ids else torch.zeros(4,device=env.device),
                            'position_error':pos[ids], 'orientation_error_deg':angle[ids]}
                    for k,v in data.items():records[k].append(v.cpu().numpy().copy() if isinstance(v,torch.Tensor) else v)
                if i%100==99:
                    env.meter.clear()
                    print('captured %.1f / 64s' % ((i+1)*env.dt), flush=True)
                if i%250==249:
                    partial={k:np.asarray(v) for k,v in records.items()}
                    partial['all_env_errors']=np.asarray(errors)
                    np.savez_compressed(out/'progress_states.npz',**partial)
        assert all(model_hash(m)==h for m,h in zip(models,hashes))
        arrays={k:np.asarray(v) for k,v in records.items()}
        arrays['all_env_errors']=np.asarray(errors)
        np.savez_compressed(out/'states.npz',**arrays)
        np.savez_compressed(out/'initial_states.npz',q=initial_q.cpu().numpy(),root_local=initial_root.cpu().numpy())
        metrics={}
        for j,it in enumerate(iterations):
            x=arrays['all_env_errors'][:,j*8:(j+1)*8]
            low=(arrays['time']>=12)&(arrays['time']<=60)
            metrics[str(it)]={'position_mean_cm':float(x[100:,:,0].mean()*100),'position_p95_cm':float(np.quantile(x[100:,:,0],.95)*100),
                'orientation_mean_deg':float(x[100:,:,1].mean()),'failures':sum(v['iteration']==it and not v['timeout'] for v in reset_events),
                'tcp_min_height_cm':float(x[:,:,4].min()*100),
                'video_replica_ground_segment_position_mean_cm':float(arrays['position_error'][low,j].mean()*100),
                'video_replica_tcp_below_3cm_fraction':float((arrays['ee'][low,j,2]<.03).mean()),
                'video_replica_finger_contact_frames':int((arrays['finger_force'][:,j]>.5).sum())}
        meta={'iterations':iterations,'checkpoints':checkpoints,'knots':KNOTS,'metrics':metrics,'reset_events':reset_events,
              'body_names':env.body_names,'finger_force_bodies':[env.body_names[i] for i in finger_ids],
              'dof_names':env.dof_names,'fps':25,'terrain':'flat','duration_s':64,'seed':1,'video_envs':ids,
              'protocol':'4 policies, 8 nominal replicas each; world-frame identical targets; velocity commands all zero; nominal physics; DR and push off; 90s evaluation timeout; identical reset setpoints, initial jitter removed; post-reset PhysX variation measured; failures retained',
              'position_frame':'ground-fixed, relative to initial AS2 base xy','orientation_frame':'world XYZW quaternion',
              'out_of_training_range':True,'optimizer_updates':0,'source_sha256':source_hashes(),
              'initial_q_max_difference':q_delta,'initial_root_max_difference':root_delta,
              'urdf':str(Path(__file__).resolve().parents[1]/'resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf')}
        (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
        print(json.dumps(metrics,indent=2),flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
