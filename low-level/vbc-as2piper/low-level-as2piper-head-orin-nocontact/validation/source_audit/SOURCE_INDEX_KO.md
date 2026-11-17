# 원본 VBC 대비 현재 런타임 소스 변경 목록

원본: vbc-legacy/low-level 및 third_party/rsl_rl. 현재: low-level-as2piper-head-orin-nocontact. 학습 상태 점검이 아닌 읽기 전용 소스 대조다.

| 파일 | 구분 | 변경 심볼 |
|---|---|---|
| legged_gym/envs/__init__.py | modified | 모듈 수준 코드 |
| legged_gym/envs/manip_loco/as2_head_orin_config.py | added | AS2HeadOrinCfg, AS2HeadOrinCfg.env, AS2HeadOrinCfg.init_state, AS2HeadOrinCfg.asset, AS2HeadOrinCfg.control, AS2HeadOrinCfg.goal_ee, AS2HeadOrinCfg.goal_ee.sphere_center, AS2HeadOrinCfg.goal_ee.ranges, AS2HeadOrinCfg.domain_rand, AS2HeadOrinCfg.termination, AS2HeadOrinCfg.rewards, AS2HeadOrinCfg.rewards.scales, AS2HeadOrinCfgPPO, AS2HeadOrinCfgPPO.runner |
| legged_gym/envs/manip_loco/manip_loco.py | modified | ManipLoco, ManipLoco.__init__, ManipLoco.step, ManipLoco.post_physics_step, ManipLoco.compute_observations, ManipLoco._termination_ground_height, ManipLoco._reward_foot_ground_heights, ManipLoco.check_termination, ManipLoco.create_sim, ManipLoco.reset_idx, ManipLoco._get_env_origins, ManipLoco._create_envs, ManipLoco._process_rigid_body_props, ManipLoco._init_buffers, ManipLoco._push_robots, ManipLoco._reset_dofs, ManipLoco._get_noise_scale_vec, ManipLoco._collision_check, ManipLoco._update_curr_ee_goal, ManipLoco._get_ee_goal_spherical_center |
| legged_gym/envs/rewards/maniploco_rewards.py | modified | ManipLoco_rewards, ManipLoco_rewards._reward_feet_jerk, ManipLoco_rewards._reward_feet_drag, ManipLoco_rewards._reward_feet_contact_forces, ManipLoco_rewards._reward_base_height, ManipLoco_rewards._reward_tracking_contacts_shaped_force, ManipLoco_rewards._reward_tracking_contacts_shaped_vel, ManipLoco_rewards._reward_feet_height, ManipLoco_rewards._reward_feet_air_time |
| legged_gym/scripts/b1z1_interface.py | modified | ManipLoco_Policy, ManipLoco_Policy.init_env, ManipLoco_Policy.step |
| legged_gym/scripts/play.py | modified | play |
| legged_gym/scripts/train.py | modified | train |
| legged_gym/utils/helpers.py | modified | export_policy_as_jit, HistoryPolicyExporter, HistoryPolicyExporter.__init__, HistoryPolicyExporter.forward |
| rsl_rl_local/rsl_rl/modules/actor_critic.py | modified | ActorCritic, ActorCritic.__init__, ActorCritic.__init__.Actor, ActorCritic.__init__.Actor.forward, ActorCritic.__init__.Critic, ActorCritic.__init__.Critic.forward |
| rsl_rl_local/rsl_rl/runners/on_policy_runner.py | modified | OnPolicyRunner, OnPolicyRunner.__init__, OnPolicyRunner.get_inference_policy |
| force_ablation/__init__.py | added | 모듈 수준 코드 |
| force_ablation/configs.py | added | Force40Cfg, Force200Cfg, Force200Cfg.rewards, ForceOffCfg, ForceOffCfg.rewards, ForceOffCfg.rewards.scales, configs, register_tasks |
| force_ablation/report_validation.py | added | 모듈 수준 코드 |
| force_ablation/run.py | added | serial, canonical, model_hash, build_model, initialization, config_manifest, compare_settings, environment, initialize_runner, source_hashes, new_directory, validate, train_run, evaluate, main |
| force_ablation/runtime.py | added | Meter, Meter.__init__, Meter.clear, Meter.record, Meter.sample, Meter.summary, Meter.save_samples, MeteredEnvironment, MeteredEnvironment.__init__, MeteredEnvironment.compute_reward, AblationRunner, AblationRunner.log |
| run_force_ablation.sh | added | 모듈 수준 코드 |
| run_head.sh | added | 모듈 수준 코드 |

새 config·launcher·ablation 계측/평가 코드를 포함한다. .wandb_deps/.deps/IsaacGym 바이너리, 모델 weight, 로그/영상, 검증 스크립트 이력은 소스 patch 대상에서 제외하며 CHANGES_KO.md 및 각 검증 보고서에 별도 기록한다.

URDF 교체·질량/관성·메시 출처와 미확인 하드웨어 항목은 CHANGES_KO.md의 자산/기구/제어 설명을 함께 참조한다. 원본 B1 URDF와 AS2 URDF는 서로 다른 로봇 자산이므로 이 Python 소스 patch에 기계적 차이를 전부 표현하지 않는다.

기존 각 단계 changes.patch는 해당 시점의 diff이며, 현재 런타임 소스의 통합 diff는 original_vbc_to_current.patch다.
