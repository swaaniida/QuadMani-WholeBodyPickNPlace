"""Offline pose/path screening; never called by the training sampler."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import isaacgym
from isaacgym.torch_utils import sphere2cart, quat_from_euler_xyz, quat_rotate_inverse
import torch
from legged_gym.envs.manip_loco.as2piper_hw import RefinedGuard
from legged_gym.utils.urdf_chain import rotate

ROOT = Path(__file__).resolve().parent
CANDIDATES = {
    'original': [[.38, .52], [-.25, .60], [-.40, .40], [-.15, .15]],
    'compact': [[.40, .48], [.10, .50], [-.25, .25], [-.10, .10]],
    'conservative': [[.40, .47], [.20, .50], [-.20, .20], [-.08, .08]],
    'tight': [[.40, .46], [.25, .48], [-.15, .15], [-.05, .05]],
    'front': [[.44, .48], [.35, .50], [-.10, .10], [-.03, .03]],
    'center': [[.42, .46], [.38, .46], [-.10, .10], [-.03, .03]],
    'extended': [[.46, .49], [.30, .45], [-.10, .10], [-.03, .03]],
    'extended_small_orn': [[.46, .49], [.30, .45], [-.10, .10], [-.015, .015]],
    'front_mid': [[.46, .49], [.25, .45], [-.15, .15], [-.015, .015]],
    'front_work': [[.46, .50], [.20, .45], [-.20, .20], [-.015, .015]],
    'port_broad': [[.38, .52], [-.25, .60], [-.40, .40], [-.15, .15]],
    'port_medium': [[.40, .52], [.0, .55], [-.35, .35], [-.10, .10]],
    'port_inner': [[.40, .50], [.10, .55], [-.30, .30], [-.10, .10]],
    'port_front': [[.43, .51], [.05, .55], [-.35, .35], [-.10, .10]],
    'port_lower': [[.42, .50], [-.35, .55], [-.30, .30], [-.10, .10]],
    'port_ground': [[.45, .60], [-.75, .55], [-.35, .35], [-.10, .10]],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--candidates', nargs='+', default=list(CANDIDATES))
    parser.add_argument('--random-paths', type=int, default=256)
    parser.add_argument('--extra-offline-iterations', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--path-samples', type=int, default=9)
    parser.add_argument('--tilt-grid', type=int, default=3)
    parser.add_argument('--tilt-limit', type=float, default=.15)
    parser.add_argument('--frame', choices=['body_mount', 'vbc'], default='body_mount')
    parser.add_argument('--root-heights', type=float, nargs='+', default=[.3244069024246581])
    parser.add_argument('--exact-collision', action='store_true')
    args = parser.parse_args()
    device = args.device
    torch.manual_seed(args.seed)
    asset = ROOT / 'resources/robots/as2piper/urdf/as2piper_plate_forward_training.urdf'
    guard = RefinedGuard(asset, ROOT/'resources/commands/arm_collision.json', device)
    if args.exact_collision:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(ROOT/'resources/commands/plate_forward.xml'))
        data = mujoco.MjData(model)
        arm_addr = [int(model.jnt_qposadr[model.joint('piper_joint'+str(i)).id]) for i in range(1, 7)]
        for leg in ['FL','FR','RL','RR']:
            for joint,value in zip(['hip','thigh','calf'],[0,.8,-1.6]):
                data.qpos[model.joint(leg+'_'+joint+'_joint').qposadr[0]]=value
        data.qpos[model.joint('piper_gripper_joint1').qposadr[0]]=.049
        data.qpos[model.joint('piper_gripper_joint2').qposadr[0]]=-.049
        arm_geoms={i for i in range(model.ngeom) if model.geom_contype[i] and
                   model.body(int(model.geom_bodyid[i])).name.startswith('piper_') and
                   model.body(int(model.geom_bodyid[i])).name not in ['piper_base_link','piper_link1']}
        body_geoms={i for i in range(model.ngeom) if model.geom_contype[i] and
                    model.body(int(model.geom_bodyid[i])).name=='base_link'}
        def exact(q_values):
            output=[]
            for row in q_values.detach().cpu().numpy():
                data.qpos[arm_addr]=row
                mujoco.mj_forward(model,data)
                hit=False
                for contact in data.contact:
                    a,b=int(contact.geom1),int(contact.geom2)
                    if contact.dist < 0 and ((a in arm_geoms and b in body_geoms) or
                                             (b in arm_geoms and a in body_geoms)):
                        hit=True;break
                output.append(hit)
            return torch.tensor(output,device=device,dtype=torch.bool)
    home = torch.tensor([0, 1.1, -.55, 0, 0, 0], device=device)
    home_sphere = torch.tensor([.4278984686862198, .421686284190671, 0], device=device)
    offset = torch.tensor([.005, 0, .08202847821788371], device=device)
    tilt_values = torch.linspace(-args.tilt_limit, args.tilt_limit, args.tilt_grid).tolist() if args.tilt_grid > 1 else [0.]
    tilts = torch.tensor(list(itertools.product(tilt_values, tilt_values, args.root_heights)), device=device)
    results = {}
    for name in args.candidates:
        bounds = CANDIDATES[name]
        low = torch.tensor([b[0] for b in bounds[:3]]+[bounds[3][0]]*3, device=device)
        high = torch.tensor([b[1] for b in bounds[:3]]+[bounds[3][1]]*3, device=device)
        corners = torch.tensor(list(itertools.product([0., 1.], repeat=6)), device=device)
        end = torch.cat((low+(high-low)*corners,
                         low+(high-low)*torch.rand(args.random_paths, 6, device=device)))
        # Half begin at home; half from independently sampled previous targets.
        start = home_sphere.repeat(len(end), 1)
        start[len(corners):] = low[:3]+(high[:3]-low[:3])*torch.rand(args.random_paths, 3, device=device)
        end = end.repeat(len(tilts), 1)
        start = start.repeat(len(tilts), 1)
        body_rp = tilts.repeat_interleave(len(start)//len(tilts), 0)
        n = len(start)
        body = quat_from_euler_xyz(body_rp[:, 0], body_rp[:, 1], torch.zeros(n, device=device))
        q = home.repeat(n, 1)
        rejected = torch.zeros(n, dtype=torch.bool, device=device)
        by_step = []
        max_error = 0.; max_angle = 0.; joint_margin = float('inf'); min_z = float('inf')
        with torch.no_grad():
            for t in torch.linspace(0, 1, args.path_samples).tolist():
                sphere = torch.lerp(start, end[:, :3], t)
                world_vector = sphere2cart(sphere)
                mount = rotate(body, offset.expand(n, 3))
                mount[:, 2] += body_rp[:, 2]
                center = torch.tensor([.005, 0, .3244069024246581+.08202847821788371], device=device)
                world = mount+world_vector if args.frame == 'body_mount' else center+world_vector
                target = quat_rotate_inverse(body, world-mount)
                orn = quat_from_euler_xyz(end[:, 3], -sphere[:, 1]+2.455216137744035+end[:, 4],
                                         sphere[:, 2]+end[:, 5])
                basis = torch.eye(3, device=device).expand(n, 3, 3)
                axes = rotate(orn[:, None, :].expand(-1, 3, -1), basis)
                local_axes = quat_rotate_inverse(body[:, None, :].expand(-1, 3, -1).reshape(-1, 4),
                                                 axes.reshape(-1, 3)).reshape(n, 3, 3).transpose(1, 2)
                q, failed = guard.solve(target, local_axes, q)
                if args.extra_offline_iterations and failed.any():
                    ids = failed.nonzero(as_tuple=False).flatten()
                    q1, f1 = guard.solve(target[ids], local_axes[ids], q[ids], args.extra_offline_iterations)
                    q[ids] = q1; failed[ids] = f1
                proxy_collision = guard.collides(q, offset[None, None, :].expand(n, 1, 3))
                collision = exact(q) if args.exact_collision else proxy_collision
                z = world[:, 2]
                floor = z < .03
                bad = failed | collision | floor
                rejected |= bad
                actual, rotation, _, _, _ = guard.fk(q)
                error = (actual-target).norm(dim=-1)
                angle = torch.rad2deg(torch.acos((((rotation*local_axes).sum((1, 2))-1)/2).clamp(-1, 1)))
                max_error = max(max_error, error.max().item()); max_angle = max(max_angle, angle.max().item())
                joint_margin = min(joint_margin, torch.minimum(q-guard.low, guard.high-q).min().item())
                min_z = min(min_z, z.min().item())
                by_step.append(dict(t=t, ik_failed=int(failed.sum()), body_collision=int(collision.sum()),
                                    proxy_body_collision=int(proxy_collision.sum()),
                                    proxy_false_positive=int((proxy_collision&~collision).sum()),
                                    floor_failed=int(floor.sum()), failed=int(bad.sum())))
        results[name] = dict(bounds=bounds, paths=n, poses=n*args.path_samples, rejected_paths=int(rejected.sum()),
                             rejected_fraction=float(rejected.float().mean()),
                             max_ik_position_error_m=max_error, max_ik_orientation_error_deg=max_angle,
                             min_joint_margin_from_guard_rad=joint_margin, min_tcp_ground_height_m=min_z,
                             steps=by_step,
                             rejected_by_body_tilt=[dict(roll=float(tilts[i, 0]), pitch=float(tilts[i, 1]), root_height=float(tilts[i, 2]),
                                 rejected=int(rejected.reshape(len(tilts), -1)[i].sum())) for i in range(len(tilts))])
        print(name, json.dumps(results[name]), flush=True)
        Path(args.output).write_text(json.dumps(dict(asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
            body_roll_pitch_rad=tilt_values, samples_per_path=args.path_samples, seed=args.seed,
            extra_offline_iterations=args.extra_offline_iterations, candidates=results,
            frame=args.frame, root_heights=args.root_heights,
            exact_collision=args.exact_collision,
            limitations='Sampled IK/body OBB/ground checks, not continuous collision certification; no legs/cables/objects/self-pairs.'), indent=2))


if __name__ == '__main__':
    main()
