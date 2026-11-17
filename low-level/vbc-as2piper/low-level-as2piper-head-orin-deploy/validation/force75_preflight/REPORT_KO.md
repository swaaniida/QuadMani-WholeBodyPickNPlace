# 75N 학습 전 점검

## 변경

- `as2_head_orin_config.py`: 기본 threshold75N 명시(기존 상속40N).
- `force_ablation/configs.py`: B200N→75N; A40N/C40N·scale0 보존.
- `force_ablation/run.py`: B 설정 검증 assertion75N. 초기화·학습·저장 로직 변경 없음.
- PhysX net-contact-force XYZ norm의 발별75N 초과분 합 ×−0.001. 시작2초 제외 유지. 전체 Leg Reward `/100` 유지.
- Foot Lift Penalty, terrain 계산, 기존 다른 Reward/PPO/DAgger/DR/Push는 변경하지 않음.

## 점검 결과

- 런타임 Python 전체 AST 구문 검사 통과.
- 6144환경,45000iterations,save_interval200; 755학습/737배포,18action,20DOF,질량23.4303404741327kg.
- privileged/history actor·critic forward, JIT737입력 출력 일치 통과.
- 합성 force probe: 네 발 각각100N→raw100·weighted−0.1, 각각200N→raw500·weighted−0.5. threshold75N 이하0, reset초기 exclusion 확인.
- GPU PhysX16환경·120스텝(2.4초), 정지/전진/회전/전진+회전 명령, zero-action PD. 원본 DR/Push 설정 상속; 검증 편의상 terrain4x4/border5m. 비정상 reset0, 모델 hash 불변, optimizer update0.
- 실제 net force 비영값; 75N force reward 활성; reward 합산 일치. 정지 contact shaping0, 이동 force/vel shaping음수.
- GPU result.json의 env.num_envs는 원래 cfg6144로 기록됨. 실제검증16환경은 validation_overrides에 명시.
- CPU contact-sensitive 검증 실패: 발 net-force 모두0. 독립 CPU 검사에서 접지 높이≈28mm·다른 body force 비영값 확인. CPU/GPU 차이의 세부 원인은 미확정이며 production 설정을 수정하지 않음. GPU 학습 경로에서 재검증 통과.
- 새 workspace·카메라 모델의 장시간 수렴,6144환경 메모리/성능,실기 성능은 이번 zero-action 검사로 검증되지 않음.
- W&B SDK import/로컬 init 코드 경로 검토. 원격 로그인/프로젝트 쓰기는 이번 점검에서 실행하지 않음.
- hardware.json의 질량23.328kg은 이전 asset 기반 gain/DR 설계 참고값; 실제 simulator 질량은 현재URDF23.430kg. 요청 범위에 따라 재조정하지 않음.

## 보존 및 실행

- 변경 전 파일 before/ 및 unified changes.patch 보존. 보호파일 해시 비교 preservation.json.
- 기존 nocontact45k 실험 디렉터리·checkpoint 수정 없음. 새 deploy의 experiments/contact_force_ablation 아래에 독립 저장.
- fresh training은 --mode train --condition B --seed1 사용. resume 경로 없음. W&B project as2piper-low, 새로운75N 이름 사용.
- 상위 restart_B_push25.sh 및 force_ablation/restart_push25.py는 과거 실험 정리/재시작용이므로 새 학습에 사용하지 않음.

증거: static_checks.json, physics_checks.json(CPU), gpu_physics.log, ../../experiments/contact_force_ablation/validation/preflight_75N_gpu_20261115/result.json.
