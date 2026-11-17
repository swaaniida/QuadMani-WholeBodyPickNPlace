"""Prepare/validate by default. Training requires an explicit --mode train invocation."""
import argparse, hashlib, json, random
from datetime import datetime, timezone
from pathlib import Path
import isaacgym
from isaacgym import gymapi, gymutil
import numpy as np
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict, set_seed
from rsl_rl.modules import ActorCritic
from .configs import configs, TASKS, register_tasks
from .runtime import MeteredEnvironment, AblationRunner

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/contact_force_ablation'

def serial(value):
    return value.tolist() if hasattr(value, 'tolist') else str(value)

def canonical(value):
    return json.dumps(value, sort_keys=True, default=serial, separators=(',', ':'))

def model_hash(model):
    state = model.state_dict() if hasattr(model, 'state_dict') else model
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode()); h.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def build_model(train, device='cpu'):
    return ActorCritic(67, 67, 18, **class_to_dict(train.policy), num_priv=18, num_hist=10, num_prop=67).to(device)

def initialization(seed):
    path = BASE / ('seed_' + str(seed)) / 'initial_policy.pt'
    _, train = configs('A', seed)
    contract = {'seed': seed, 'policy': class_to_dict(train.policy), 'proprio': 67, 'privileged': 18, 'history': 10, 'action': 18}
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        set_seed(seed)
        model = build_model(train)
        payload = {'purpose': 'untrained common initialization; optimizer steps=0', 'contract': canonical(contract),
                   'model_state_dict': model.state_dict(), 'model_sha256': model_hash(model)}
        # Exclusive creation prevents replacing an existing initialization/checkpoint.
        with path.open('xb') as f:
            torch.save(payload, f)
    value = torch.load(str(path), map_location='cpu')
    assert value['contract'] == canonical(contract), 'Existing initialization uses a different seed/policy contract'
    assert value['model_sha256'] == model_hash(value['model_state_dict'])
    return path, value

def config_manifest(case, seed):
    env, train = configs(case, seed)
    return {'condition': case, 'task': TASKS[case], 'seed': seed, 'env': class_to_dict(env), 'train': class_to_dict(train)}

def compare_settings(seed):
    manifests = {c: config_manifest(c, seed) for c in 'ABC'}
    reference = manifests['A']
    for c, current in manifests.items():
        left = json.loads(canonical(reference)); right = json.loads(canonical(current))
        for x in [left, right]:
            x.pop('condition'); x.pop('task')
            x['env']['rewards'].pop('max_contact_force')
            x['env']['rewards']['scales'].pop('feet_contact_forces')
        assert left == right, 'Unintended difference in ' + c
    assert manifests['A']['env']['rewards']['max_contact_force'] == 40
    assert manifests['B']['env']['rewards']['max_contact_force'] == 75
    assert manifests['C']['env']['rewards']['scales']['feet_contact_forces'] == 0
    return manifests

def environment(case, seed, device, small=False, terrain=None):
    cfg, train = configs(case, seed)
    if small:
        cfg.env.num_envs = 16 if terrain is None else 32
        cfg.terrain.num_rows = cfg.terrain.num_cols = 4
        cfg.terrain.border_size = 5
    if terrain is not None:
        if terrain == 'flat': cfg.terrain.height = [0., 0.]
        for flag in ['randomize_friction', 'randomize_base_mass', 'randomize_base_com', 'randomize_motor', 'randomize_gripper_mass', 'push_robots']:
            setattr(cfg.domain_rand, flag, False)
    set_seed(seed)
    params = gymapi.SimParams()
    params.dt = cfg.sim.dt; params.up_axis = gymapi.UP_AXIS_Z; params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = device.startswith('cuda'); params.physx.use_gpu = device.startswith('cuda')
    gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
    return MeteredEnvironment(cfg, params, gymapi.SIM_PHYSX, device, True), train

def initialize_runner(env, train, initial, directory=None):
    runner = AblationRunner(env, class_to_dict(train), log_dir=None if directory is None else str(directory), device=env.device)
    runner.alg.actor_critic.load_state_dict(initial['model_state_dict'], strict=True)
    assert model_hash(runner.alg.actor_critic) == initial['model_sha256']
    assert not runner.alg.optimizer.state and not runner.alg.hist_encoder_optimizer.state
    assert runner.alg.counter == 0 and runner.current_learning_iteration == 0
    env.meter.clear()
    return runner

def source_hashes():
    names = ['legged_gym/envs/manip_loco/manip_loco.py', 'legged_gym/envs/manip_loco/as2_head_orin_config.py',
        'legged_gym/envs/manip_loco/b1z1_config.py', 'legged_gym/envs/rewards/maniploco_rewards.py',
        'rsl_rl_local/rsl_rl/algorithms/ppo.py', 'rsl_rl_local/rsl_rl/storage/rollout_storage.py',
        'rsl_rl_local/rsl_rl/modules/actor_critic.py', 'rsl_rl_local/rsl_rl/runners/on_policy_runner.py',
        'validation/hardware.json', 'resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf',
        'force_ablation/configs.py', 'force_ablation/runtime.py', 'force_ablation/run.py']
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}

def new_directory(kind, case, seed, name=None):
    name = name or (case + '_seed' + str(seed) + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    assert Path(name).name == name and name not in ['.', '..']
    path = BASE / kind / name
    path.mkdir(parents=True, exist_ok=False)
    return path

def validate(args, initial_path, initial):
    path = new_directory('validation', args.condition, args.seed, args.name)
    env, train = environment(args.condition, args.seed, args.device, small=True)
    try:
        runner = initialize_runner(env, train, initial)
        env.meter.verify = True
        targets = torch.tensor([[0., 0., 0.], [.3, 0., 0.], [0., 0., .6], [.3, 0., .6]], device=env.device).repeat_interleave(4, 0)
        env._resample_commands = lambda ids: env.commands.__setitem__(ids, targets[ids])
        env.commands.copy_(targets)
        # Fixed states isolate the causal change in the force reward alone.
        import copy
        from legged_gym.envs.rewards.maniploco_rewards import ManipLoco_rewards
        shadow = copy.copy(env); shadow.contact_forces = env.contact_forces.clone()
        shadow.episode_length_buf = torch.full_like(env.episode_length_buf, 200)
        reward = ManipLoco_rewards(shadow)
        probes = {}
        for force in [0., 40., 150., 200., 250.]:
            shadow.contact_forces[:, env.feet_indices] = torch.tensor([0., 0., force], device=env.device)
            raw, _ = reward._reward_feet_contact_forces()
            expected = 4 * max(force - env.cfg.rewards.max_contact_force, 0)
            assert torch.allclose(raw, torch.full_like(raw, expected))
            probes[str(force)] = {'raw': float(raw.mean()), 'weighted': float(raw.mean() * env.cfg.rewards.scales.feet_contact_forces)}
        for _ in range(120):
            env.commands.copy_(targets)
            env.step(torch.zeros(env.num_envs, 18, device=env.device))
        summaries = {name: env.meter.summary(slice(i * 4, (i + 1) * 4)) for i, name in enumerate(['stand', 'forward', 'turn', 'forward_turn'])}
        for name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
            assert summaries['stand']['RewardWeighted/leg/' + name] == 0
            assert summaries['forward']['RewardWeighted/leg/' + name] < 0
        assert env.obs_buf.shape[-1] == 755
        assert model_hash(runner.alg.actor_critic) == initial['model_sha256']
        if args.condition == 'C':
            assert 'feet_contact_forces' not in env.reward_names
            assert summaries['forward']['RewardWeighted/leg/feet_contact_forces'] == 0
        manifest = config_manifest(args.condition, args.seed)
        manifest.update(initialization=str(initial_path), initial_model_sha256=initial['model_sha256'],
            final_model_sha256=model_hash(runner.alg.actor_critic), optimizer_steps=0,
            validation_overrides={'envs':16, 'terrain_rows_cols':[4,4], 'border_size_m':5, 'steps':120, 'seconds':120*env.dt, 'actions':'zero leg/arm actions; original arm IK; DR otherwise inherited'},
            force_probes=probes, reward_sum_verified=True, metrics=summaries,
            dr_parameters_sha256=hashlib.sha256(torch.cat([env.mass_params_tensor.flatten(),env.friction_coeffs_tensor.flatten(),env.motor_strength.flatten()]).cpu().numpy().tobytes()).hexdigest(),
            source_sha256=source_hashes())
        env.meter.save_samples(path / 'samples.npz')
        (path / 'result.json').write_text(json.dumps(manifest, indent=2, default=serial) + '\n')
        print('Validation saved:', path, flush=True)
    finally:
        env.gym.destroy_sim(env.sim)

def train_run(args, initial_path, initial):
    import wandb
    path = new_directory('runs', args.condition, args.seed, args.name)
    manifest = config_manifest(args.condition, args.seed)
    manifest.update(initialization=str(initial_path), initial_model_sha256=initial['model_sha256'], source_sha256=source_hashes(), wandb_run_id=wandb.util.generate_id())
    (path / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=serial) + '\n')
    wandb.init(project=args.project, name=path.name, id=manifest['wandb_run_id'], resume='never', dir=str(path), config=manifest)
    env, train = environment(args.condition, args.seed, args.device)
    try:
        runner = initialize_runner(env, train, initial, path)
        (path / 'initial_model_sha256.txt').write_text(model_hash(runner.alg.actor_critic) + '\n')
        runner.learn(num_learning_iterations=train.runner.max_iterations, init_at_random_ep_len=True)
    finally:
        env.gym.destroy_sim(env.sim); wandb.finish()

def evaluate(args):
    assert args.checkpoint is not None, '--checkpoint is required'
    checkpoint = Path(args.checkpoint).resolve()
    assert checkpoint.is_file()
    saved_manifest = checkpoint.parent / 'manifest.json'
    if saved_manifest.exists():
        saved = json.loads(saved_manifest.read_text())
        assert saved['condition'] == args.condition, 'Checkpoint condition differs from evaluation condition'
        assert saved['source_sha256'] == source_hashes(), 'Evaluation source differs from recorded training source'
    path = new_directory('evaluations', args.condition, args.seed, args.name)
    env, train = environment(args.condition, args.seed, args.device, small=True, terrain=args.terrain)
    try:
        model = build_model(train, env.device)
        weights = torch.load(str(checkpoint), map_location=env.device)
        model.load_state_dict(weights['model_state_dict'], strict=True); model.eval()
        initial_hash = model_hash(model)
        targets = torch.tensor([[0.,0.,0.],[.4,0.,0.],[0.,0.,.6],[.4,0.,.6]],device=env.device).repeat_interleave(8,0)
        env._resample_commands=lambda ids:env.commands.__setitem__(ids,targets[ids])
        env.reset(); env.global_steps=int(weights.get('iter',45000))*train.runner.num_steps_per_env
        # Prescribed central EE targets make A/B/C comparison commands identical after resets.
        env.cfg.env.teleop_mode=True
        with torch.no_grad():
            for i in range(int(args.seconds/env.dt)):
                t=i*env.dt
                env.curr_ee_goal_sphere[:]=torch.tensor([.44+.02*np.sin(t),.3+.1*np.sin(t*.7),.15*np.sin(t*.5)],device=env.device)
                env.ee_goal_orn_delta_rpy.zero_();env._update_curr_ee_goal();env.commands.copy_(targets)
                env.step(model.act_inference(env.obs_buf,True).clone())
                if i==99: env.meter.clear()
        assert model_hash(model)==initial_hash
        values={name:env.meter.summary(slice(j*8,(j+1)*8)) for j,name in enumerate(['stand','forward','turn','forward_turn'])}
        env.meter.save_samples(path/'samples.npz')
        (path/'result.json').write_text(json.dumps({'condition':args.condition,'seed':args.seed,'terrain':args.terrain,'seconds':args.seconds,'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'source_sha256':source_hashes(),'metrics':values,'optimizer_steps':0,'protocol':'32 env, DR/push off, 2s warmup, fixed velocity and central EE commands; terrain is original generator 4x4; simulation performance only'},indent=2)+'\n')
        print('Evaluation saved:',path,flush=True)
    finally:env.gym.destroy_sim(env.sim)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','validate','train','evaluate'],default='validate')
    parser.add_argument('--condition',choices=list('ABC'),default='A')
    parser.add_argument('--seed',type=int,default=1);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--name');parser.add_argument('--project',default='as2piper-low')
    parser.add_argument('--checkpoint');parser.add_argument('--terrain',choices=['flat','rough'],default='flat');parser.add_argument('--seconds',type=float,default=40)
    args=parser.parse_args();torch.set_num_threads(2)
    register_tasks();manifests=compare_settings(args.seed)
    if args.mode=='evaluate':
        assert args.seconds>2;evaluate(args);return
    path,initial=initialization(args.seed)
    if args.mode=='prepare':
        target=path.parent/'conditions.json'
        value=canonical(manifests)
        if target.exists():assert target.read_text()==value
        else:
            with target.open('x') as f:f.write(value)
        print('Common initialization:',path,'model SHA256:',initial['model_sha256'])
    elif args.mode=='validate':validate(args,path,initial)
    else:train_run(args,path,initial)

if __name__=='__main__':main()
