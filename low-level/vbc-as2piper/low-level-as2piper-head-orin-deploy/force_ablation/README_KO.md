# Contact Force Reward Ablation

기존 No-contact 태스크와 checkpoint를 수정하지 않는 별도 태스크/실행 경로다.

| 조건 | 태스크 | max_contact_force | feet_contact_forces scale |
|---|---|---:|---:|
| A | as2piper_nocontact_force_a | 40N | -0.001 |
| B | as2piper_nocontact_force_b | 200N | -0.001 |
| C | as2piper_nocontact_force_c | 40N(진단용; 보상에는 무관) | 0 |

공통: feet_height_target=0.18m, height scale+1.0, gait clock=True,2Hz, contact observation 제거, 학습755/배포737, shaping scale-2와 정지 walking_mask, 원래 PPO·DR·기타 reward·환경 수 유지. 훈련은 기존6144환경·45000 iteration이며 DAgger/PPO 업데이트 주기도 원본 그대로다. 이번 준비 작업에서는 학습하지 않는다.

## 초기화·보존

Seed 기본1. seed별 initial_policy.pt를 한 번만 생성하고 A/B/C actual runner에 strict load한다. Model SHA256이 공통 초기화와 일치하고 Adam/history Adam state가 비어 있으며 iteration/counter0인지 확인한 후 시작한다. 이 파일은 학습된 checkpoint가 아닌 공통 초기 가중치다. 기존 파일은 덮어쓰지 않으며 다른 policy contract면 실패한다. 새 run/evaluation 디렉터리도 이름이 존재하면 실패한다.

원본6144 환경/10x20 terrain 등은 훈련 경로에서 유지한다. 검증만16환경/4x4 terrain/120step으로 줄인다. 검증에서도 DR 범위·활성 flag를 유지한다. 모든 조건에 동일 override를 적용한다. 같은seed는 sampling/RNG와초기weight를 맞추지만 GPU PhysX의 물리 trajectory가 bitwise 같다는 보장은 아니다.

프로젝트 루트에서:

```bash
bash low-level-as2piper-head-orin-nocontact/run_force_ablation.sh --mode prepare --seed 1
bash low-level-as2piper-head-orin-nocontact/run_force_ablation.sh --mode validate --condition A --seed 1
bash low-level-as2piper-head-orin-nocontact/run_force_ablation.sh --mode validate --condition B --seed 1
bash low-level-as2piper-head-orin-nocontact/run_force_ablation.sh --mode validate --condition C --seed 1
```

기본 mode는validate다. 실제 훈련은 추후 검토 후 명시적으로 --mode train을 사용해야 한다. 예시이며 이번에 실행하지 않았다:

```bash
bash low-level-as2piper-head-orin-nocontact/run_force_ablation.sh --mode train --condition A --seed 1 --name cf_A40_seed1
bash low-level-as2piper-head-orin-nocontact/run_force_ablation.sh --mode train --condition B --seed 1 --name cf_B200_seed1
bash low-level-as2piper-head-orin-nocontact/run_force_ablation.sh --mode train --condition C --seed 1 --name cf_Coff_seed1
```

W&B project 기본as2piper-low. 이름/ID는 --name, 미지정시 조건·seed·UTC timestamp로 구분한다. 이름 재사용·resume·원본 checkpoint덮어쓰기 경로는 제공하지 않는다. 다른seed를 시험하면 그seed별 공통 초기화를 새로 만들어 A/B/C를 한 묶음으로 비교한다.

## 지표

계측은 reward 함수가 실제 호출될 때 raw를 한 번 기록한다. 타이머/jerk와 같은 stateful reward를 재호출하지 않는다. C의 force 함수만 순수 진단용으로 호출해 raw를 남기며 weighted0으로 고정한다. Raw 기록과 별개로 실제 reward 합산에는 C force가 없다.

- 속도: vx/vy/yaw tracking RMSE, 실제 vx/yaw 평균.
- EE: 실제 TCP position error mean/P95.
- Reset: timeout과물리 실패 분리,1000 env-step당 실패율.
- 발 net contact force norm: pooled/발별 평균·P95·P99, 비접촉0 표본 포함. 센서6축 norm이나 Fz만의 통계와 구분.
- 모든 활성 reward: raw mean, weighted mean(/100이전), branch별signed contribution과absolute contribution.

훈련 중 한 native24-step rollout마다 ablation_metrics.jsonl과W&B에 기록한다. Reward absolute share=E|weighted term|/ΣE|weighted term|, signed share=Eweighted/Ebranch reward. 작은/음수분모의 signed share는 해석에 주의하며 absolute share와 함께 본다. 지표의 크기는 reward의 인과적 성능 개선 효과가 아니다.

Train용 계측 subclass는 reward/simulation을 수정하지 않고 관측·action·RNG에 영향을 주는 sampling을 추가하지 않는다. Native PPO/DAgger 알고리즘은 직접 사용한다. 훈련 manifest에 전체 설정·source hash·공통초기화hash를 저장하고 기존 loss/episode logging에 metric을 추가한다.

## 후속 checkpoint 평가

동일 seed·seconds·terrain·명령으로 세 조건의 같은 iteration checkpoint를 평가한다:

```bash
bash low-level-as2piper-head-orin-nocontact/run_force_ablation.sh --mode evaluate --condition A --seed 1001 --terrain flat --seconds 40 --checkpoint /absolute/path/to/A/model_45000.pt
```

B/C도 해당 checkpoint와조건만 바꾸고 동일 실행. rough도 별도 실행. Default32환경, 조건별8환경: 정지/[vx0.4]/[yaw0.6]/[vx0.4,yaw0.6]. DR/push는평가에서만off,2초warmup 제외, 중앙 EE 구면 궤적은시간함수로고정. 원본 terrain generator4x4에서 flatheight0/rough원래height범위를쓴다. 이는배포안전인증이나전체EE범위평가가아니다. Rough net-force reporting이 실제접촉정답이라는 가정도 하지않는다.

학습 manifest가 있는 checkpoint는 condition과source hash가다르면평가를거부한다. JSON/NPZ는 experiments/contact_force_ablation/{validation,runs,evaluations} 아래 별도 저장한다. Evaluation은inference만수행하며weights를저장하거나갱신하지않는다.
