# W&B 초기화 호환성 수정

기존SDK0.15.12의 internal_api.server_settings_introspection에서 서버가 ServerSettingsType=null을 반환하여 .get 호출이 실패했다. 이 단계는 환경/runner 생성 이전이며 optimizer 업데이트0, checkpoint 없음. 로그인 재시도나 PPO 설정 변경으로 해결할 문제가 아니다.

전역 SDK를 변경하지 않고 프로젝트 .wandb_deps에Python3.8 지원 wandb0.22.3 및의존성을설치했다. run_force_ablation.sh에서 이 경로를PYTHONPATH 최우선으로 사용한다. WANDB_ENTITY 기본quadruped_capstone도명시했다. 이전실행스크립트는run_force_ablation.before_wandb_fix.sh로보관했다.

오프라인SDK backend init/log/finish와Isaac/Torch/공통초기가중치 prepare검사를통과했다. 모델SHA256 fd0f3dfcbf5343c3f3e0a699ca50e71e4389fd069cfc445c8d09a90acec16f0f 유지. 전역SDK·기존PPO/reward/DR/URDF/checkpoint는보존했다.

Codex실행사용자에는W&B API키가없어실서버인증검사는완료하지못했다. 인증파일을직접읽거나옮기지않았다. 기존로그인이설정된root터미널에서online재실행해야한다. 온라인init성공을아직검증했다고주장하지않는다.

실패한원본로그/manifest는experiments/contact_force_ablation/failed_initializations/as2piper_headpiper_orin_200N_0.18m_20261111T130018Z에보존했다. 원래runs경로는재실행할수있게비웠으며로그를삭제하지않았다. 실제학습은재시작하지않았다.
