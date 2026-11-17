# AS2 + PiPER gait 보상 비교 결과

[최신 원인 분리와 수정 근거](CAUSES_FULL_ENV_KO.md). Fresh v3 두 후보를 원래 rough/전체DR에서0회부터 비교 중이다. 기존 warm-start 후보의 성능을 fresh 성공으로 표시하지 않는다.

> 후속 검증: VBC의 rough terrain·전체 DR·6144환경에서 보상만 바꾼 비교를 진행 중이다.
> [최신 접촉·높이 보상 비교](experiments/gait_reward_proxy_v2/RESULTS_KO.md), [첫 full-environment 결과](experiments/gait_full_env_v1/RESULTS_KO.md),
> [공통 조건과 동일 counter 비교](experiments/gait_full_env_v1/PROTOCOL_KO.md)를 우선 확인한다.
> 초기 201회에서 Gaussian 수정안은 균형을 유지하지만 이동을 거의 하지 않아 보행 성공이 아니다.
> 아래 평지 결과를 full-DR 성공으로 해석하지 않는다.

## 판단

수정 보상으로 중간 속도의 평지 보행이 학습되는 것을 확인했다. 동일 초기화·동일 학습량의 기존 보상보다 과속과 회전 중 전진 이탈이 크게 감소했다. 거친 지면과 전체 DR의 안정화는 아직 충족하지 못했으며, 평지 결과를 실물 또는 전체 환경의 안정성으로 확대하지 않는다.

## 코드 보존 및 분리

| 구분 | 경로 / task | 상태 |
|---|---|---|
| 보존한 AS2 VBC 원복판 | `../low-level-as2piper-hw`, `as2piper_plate_vbc_port` | 기존 소스 51개 SHA256 동일. `BASELINE_VBC_PORT.md`로 표시 |
| 신규 gait 복제본 | 현재 디렉터리, `as2piper_plate_gait_stable` | 별도 subclass로 보상 수정 |
| B1 원본 | `/data/team/users/suhwan/vbc-legacy` | 기록된 Python 소스 19개 SHA256 동일 |
| 기존 실행 중 학습 | PID 512627 | 유지. 신규 실험과 별개 |

원복판은 B1 소스 그대로라는 의미가 아니다. AS2 하드웨어 포팅과 contact shaping 부호 보정을 포함하며, 상속된 버그도 보존되어 있다. `experiments/preservation.json`과 `experiments/changes_from_preserved_baseline.patch`에서 확인할 수 있다.

## 확인한 원인과 변경

| 원인 | 변경 | 확인 결과 |
|---|---|---|
| tensor advanced indexing으로 복사된 발 속도가 갱신되지 않음 | PhysX 갱신 후 현재 발 속도를 다시 선택 | 실제 stance 이동이 contact velocity penalty에 반영 |
| feet_jerk 상태의 저장·검사 객체 불일치 | 동일 객체에 이전 wrench 저장 | 기존 0 항목 활성화. 단독 수정만으로 개선되지 않음 |
| 정지 contact 목표 0.5와 부호 보정된 force penalty 불일치 | 정지 시 네 발 목표 1 | 네 발 지지 시 force penalty 제거. 단독 효과는 불충분 |
| 목표보다 빨라도 속도 보상이 포화 | 대칭 planar Gaussian, sigma 0.20m/s | 과속 감소에 가장 큰 조합 효과 |
| 회전 추종 유인이 상대적으로 약함 | yaw sigma 0.35rad/s, weight 0.5→1 | 속도 수정과 함께 회전 명령 추종 개선 |
| 앞발 두 개의 world-z norm을 높이 목표로 사용 | 네 발의 위상별 clearance, 평지 swing 최대 4cm | 발 높이와 위상을 직접 연결 |
| 2Hz gait와 체공시간 0.5초 목표 불일치 | 목표 0.25초, 네 발 적용, credit 상한, weight 2→0.5 | 긴 비행의 유인 완화 |
| home 관절 근접 보상이 강함 | walking_dof weight 1.5→0.5 | 발 디딤과 충돌하는 유인 완화 |

PD, armature, 질량·관성, URDF, PPO/network, 799관측, 정책18출력/시뮬레이션20DOF, EE 목표 생성 범위는 바꾸지 않았다. 온라인 IK/충돌 유효성 검사를 추가하지 않았다. 음수 패널티라는 이유만으로 제거한 것이 아니다.

토크 limit 90% 초과 샘플은 기존 전진 평가에서 0%였고, 최대 사용비율은 71.3%였다. 발 인덱스와 PhysX→MuJoCo 발 높이 차이도 검증했으므로 토크 포화·렌더링 변환을 주원인으로 지지하는 증거는 없었다.

## 원인별 회복 실험

공통 model313, Adam/history optimizer, seed7, 1024환경에서 120회 추가 학습했다. 평지·robot DR off이며 원래 관절 reset 범위, random EE, command 분포와 상자 actor는 유지했다.

| 변경 조합 | 정지 시 전진속도 | 전진 시 속도 | 회전 시 전진속도 |
|---|---:|---:|---:|
| 기존 | 0.542 | 0.537 | 0.574 |
| 발 속도 갱신만 | 0.370 | 0.438 | 0.447 |
| jerk만 | 0.655 | 0.560 | 0.602 |
| 정지 contact만 | 0.708 | 0.595 | 0.680 |
| 세 수정 조합 | 0.382 | 0.443 | 0.456 |
| 위 조합 + 속도/yaw 추종 | 0.246 | 0.281 | 0.292 |
| 위 조합 + 발 높이/체공시간/posture | 0.254 | 0.272 | 0.255 |

속도 단위 m/s, 전진 목표 0.25m/s. 어느 조건도 이 짧은 회복 실험만으로 전체 합격하지 않았다. 기존 정책도 추가 학습 후 접촉 주파수가 3.25Hz에서 2Hz가 되었으므로, 초기의 빠른 발놀림을 특정 보상 버그 하나로만 설명할 수 없다. 조합 내부 weight의 개별 효과까지 분리한 실험은 아니다.

## 처음부터 동일하게 학습한 비교

4096환경·seed7·201회 업데이트. checkpoint 없이 시작했으며, 첫 저장 시 history 외 network tensor 33개의 초기값이 정확히 일치했다. 동일한 평지·robot DR off 조건이다.

평가: 정지/전진/회전/전진+회전 각각 8환경, 10초. 속도는 초기 2초 제외 평균, 넘어짐은 전체 10초에서 집계했다. 목표 전진 0.25m/s, 회전 +0.6rad/s.

| 동작 | 기존 전진속도 | 수정 전진속도 | 기존 yaw | 수정 yaw | 수정 넘어짐 |
|---|---:|---:|---:|---:|---:|
| 정지 | -0.001 | -0.001 | 0.003 | 0.001 | 0 |
| 전진 | 0.497 | 0.286 | 0.016 | 0.010 | 0 |
| 제자리 회전 | 0.438 | 0.039 | 0.240 | 0.559 | 0 |
| 전진+회전 | 0.540 | 0.267 | 0.256 | 0.613 | 0 |

수정안의 전진 속도 MAE는 0.037m/s, 전진+회전은 0.021m/s였다. 전진 접촉 주파수는 2Hz, roll RMS는 대표 환경에서 1.13도, 발 clearance P95는 발별 2.35~5.01cm, EE 위치 평균 오차는 약 1.6~1.8cm였다. 회전에서 좌우/앞뒤 다리 사용이 완벽하게 대칭인 것은 아니며 gait 최적성까지 주장하지 않는다.

다른 seed와 전진 0.35m/s, 반대 회전 -0.6rad/s에서도 같은 평가 기준을 통과했다. 전진 실측은 0.381m/s, 제자리 회전 yaw는 -0.552rad/s, 전진+회전은 0.366m/s / -0.583rad/s였고 넘어짐은 0회였다. 다수 seed의 학습 재현성 검증은 아니다.

선정 체크포인트: `logs/as2piper-low/gait_stable_fresh_s7_v1/model_200.pt`. 파일명과 달리 저장 metadata 기준 201회 업데이트 완료 상태다. 평가 통과 후 학습은 217회에서 정상 종료했으며 최종 model217도 별도로 보존했다. 장시간 수렴 결과는 아니다.

## 영상

`experiments/gait_fresh_v1/eval_200/learned_gait.mp4`

정지/전진/회전/전진+회전의 실제 PhysX 학습 정책 상태를 MuJoCo에서 재생했다. Scripted gait 또는 PD 데모가 아니다. 125프레임, 12.5fps, 10초로 실제 시간과 같으며 느리게 재생한 영상이 아니다. `walking_evidence.png`에 속도·접촉력·발 높이 근거를 함께 저장했다.

## 거친 지면과 DR

기본 공개 task는 원래 rough terrain과 전체 robot DR를 유지한다. 평지 비교를 위해 사용한 `--controlled` 설정이 기본 학습 환경으로 고정된 것은 아니다.

초기 평지 정책을 전체 rough+DR에서 평가하면 총 38회의 넘어짐 reset이 발생했다. 같은 정책/Adam에서 120회 적응한 grid v1은 27회로 줄었지만 여전히 불합격이었다. 접지된 발 바닥과 grid 높이 추정의 차이가 P95 4.35cm였으므로, v2는 실제 하중 접촉 높이를 저장하고 거친 지면에서는 clearance를 최소 조건으로 사용한다. 몸통 지면 높이는 네 발 지지 높이 평균의 EMA로 갱신한다. 평지의 보상 식은 유지한다.

접촉 높이 v2의 전체 넘어짐은 36회(정지 4, 전진 14, 회전 2, 전진+회전 16)로
같은 조건의 grid v1 27회보다 나빴다. 원인 가설이 타당해 보여도 실제 개선이
확인되지 않았으므로 기본값에서 제외했다. 최종 `stable`은 grid v1이며,
`stable_contact`는 재현용 실험 preset으로만 보존한다. 기존 v2 runtime_snapshot의
preset 이름 `stable`은 당시 contact mode였으므로 재현 시 `stable_contact`를 쓴다.

지형 적응 후 평지에서도 회전 중 전진 이탈이 증가했다. 따라서 검증된 평지 정책을 지형 적응 checkpoint로 대체하지 않는다. 전체 DR의 짧은 학습·평가는 실물 안전성이나 45,000회 수렴을 보증하지 않는다. 거친 지면 평가의 발 높이·upright 높이 판정은 환경 원점 기준 근사이므로 정확한 국소 clearance로 해석하지 않는다.

## 재실행

```bash
cd /data/team/users/suhwan/vbc-as2piper/low-level-as2piper-gait

# 평지 reward 검증을 처음부터 재현. W&B 비활성.
bash run_gait.sh --preset stable --controlled --num-envs 4096 \
  --iterations 201 --seed 7 --run gait_stable_fresh_reproduction_unique

# 원래 rough/full-DR 환경의 장시간 학습 진입점. 자동 실행하지 않았음.
# W&B project: as2piper-low, 매번 고유 run ID, 기존 run 덮어쓰기 거부.
bash run_stable.sh --max_iterations 45000
```

세부 수식은 `REWARD_CHANGES_KO.md`, 원인 가설은 `CAUSES_KO.md`, 전체 7조건 결과는 `experiments/gait_ablation_v1/PAIRED_COMPARISON_KO.md`에 기록했다. 실행별 runtime_snapshot에는 config·소스·asset hash를 저장했다.


## 후속 보상 비교 상태 (2026-11-01)

세부 수치와 판정은 [보상 효과 분리 비교](experiments/gait_reward_proxy_v2/REWARD_EFFECT_COMPARISON_KO.md)에 기록했다. Contactmatch와 발 접지 높이 기준, VBC deadband를 각각 분리해 비교했으나 rough/full-DR에서 명령별 95% 무낙상과 99% upright를 만족한 후보는 없다. F에 deadband를 더한 H는 600 update에서 회전 시 전진 이탈을 줄였지만 총 낙상은 7회에서 9회로 늘어 안정성 기준에 실패했다. cycle 및 실제 메시 접촉 검사까지 포함한 결과는 gait 실험 보고서에 기록했다.


## 45k 최종 모델 평가 (2026-11-02)

45,000 update checkpoint는 학습 평지 조건에서 64개 평가 환경 모두 무낙상이었고 명령 속도 오차가 작았다. 저장된 학습 환경은 실제로 plane이며 mass/CoM/push 랜덤화는 비활성 상태였다. Rough 지형 holdout 두 seed에서는 33회 및 27회의 낙상이 발생했고 혼합 전진·회전이 크게 저하됐다. [전체 성능 분석](experiments/gait_full_env_v1/FINAL_45K_PERFORMANCE_KO.md)
