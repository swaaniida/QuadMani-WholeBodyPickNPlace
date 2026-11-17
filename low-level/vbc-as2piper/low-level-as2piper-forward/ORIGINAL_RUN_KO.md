# 원본 조건 AS2 plate_only 실험

사용자 요청에 따라 AS2 자산 및 필수 하드웨어 연결만 포팅하고 나머지 학습 조건을 원본으로 복구했다. 장시간 학습은 아직 시작하지 않았다.

| 항목 | 적용값 |
|---|---|
| W&B project | `as2piper-low` |
| W&B run name / 로컬 exptid | `plate_only_C_original_20261028` |
| W&B run ID | `as2piper_plate_C_orig_20261028_01` |
| W&B mode | `online` |
| num_envs | 6144 |
| max_iterations | 45000 |
| rollout length | 24 steps/env |
| control decimation | 4 |
| terrain / commands / noise / termination | 원본과 동일 |
| domain randomization | 원본과 동일: 추가 질량 0~15kg, base CoM x/y/z ±0.15m 포함 |
| rewards 전체 | 원본과 동일: base_height_target=0.55m 포함 |
| PPO | 원본과 동일; 로그 experiment namespace만 AS2로 분리 |
| leg PD | 원본 80/2 |
| arm PhysX position drive | 원본 400/40 |
| 초기 root z | 원본 0.5m |
| 초기 다리 관절각 | 원본 hip ±0.2/thigh 0.8/calf -1.5rad |

남긴 하드웨어 차이는 AS2 plate_only URDF, PiPER 초기 자세 C, 관절 이름·2개 gripper·TCP·발 센서·base offset/body index, PiPER TCP orientation frame, 별도 gripper drive다. arm effort는 원본 코드처럼 URDF effort를 사용한다. 이것은 실제 PiPER 하드웨어 effort를 실측 검증했다는 뜻은 아니다.

**실행:**

```bash
cd /data/team/users/suhwan/vbc-as2piper
bash low-level-as2piper/run_as2.sh
```

위 명령은 W&B online으로 6144 env / 45000 iterations를 시작한다. 로그인은 이미 서버에 설정된 W&B 방식을 사용하며 Codex가 인증 파일을 읽거나 변경하지 않았다. project/run은 첫 online 실행 시 생성된다. 같은 run ID가 이미 있으면 기본 `resume=never`로 중단한다. 새 실험은 WANDB_RUN_ID와 WANDB_NAME을 함께 바꾼다. W&B resume를 지정하는 것과 PPO checkpoint를 복원하는 것은 별개다.

원본 조건 복구 후 `WANDB_MODE=disabled AS2_SMOKE=1 ... --num_envs 8 --max_iterations 2`로 100-step rollout/reset와 PPO 2 updates를 다시 통과했다. 증거는 `logs/as2piper-plate-only/original_alignment_smoke/` 및 `audit/original_alignment_smoke.log`다. 이 검증은 online 연결이나 6144 env 장시간 실행을 수행한 것이 아니다.

`check_original_settings.py`가 rewards/domain_rand/terrain/commands/termination/noise/normalization 전체 및 PPO 설정 동등성을 assert한다. 모든 config 차이는 `audit/original_settings.json`에 기록했다. 원본 reward 함수와 scale, PPO source도 보존한다.

원본 비교 조건의 물리적 한계: B1 몸통 목표 0.55m, EE spherical center z=0.7m/radius 최대0.95m, 추가 질량 최대15kg와 CoM ±0.15m는 AS2에 과도할 수 있다. 요청대로 그대로 유지했으므로 성공적인 AS2 수렴을 보장하는 하드웨어 전용 튜닝 설정은 아니다. 실제 장착 구조/카메라 질량/모터 한계의 미확정 사항도 그대로 남는다.
