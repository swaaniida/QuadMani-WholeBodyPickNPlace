# Contact Force Reward Ablation 설정·검증

## 구성 및 불변 조건

기존코드를수정하지않고 force_ablation/ 모듈과 run_force_ablation.sh만추가했다. A/B/C를독립태스크·run·출력디렉터리로분리했다. 세조건의설정차이는 contact force threshold와feet_contact_forces scale뿐임을전체config비교로확인했다. 조건C의threshold40은 raw진단기준일뿐실제보상에는무관하다.

공통: feet_height_target0.18m·scale+1, gait clock5 활성·2Hz, contact observation44 제거,755/737 입력계약, 정지contact shaping walking_mask, shaping scale−2,원래PPO·DR·기타reward. 훈련규모도기존6144환경·45000iteration·24step rollout을그대로사용한다. 원본실행스크립트와기존태스크는보존했다.

| 조건 | thresholdN | scale | 네발각150N weighted | 네발각250N weighted |
|---|---:|---:|---:|---:|
| A | 40.0 | -0.001 | -0.44 | -0.84 |
| B | 200.0 | -0.001 | -0.00 | -0.20 |
| C | 40.0 | 0.0 | 0.00 | 0.00 |

Probe는episode age>2s인동일상태이며값은/100이전이다. A/B는 실제 reward 목록에포함되고C는 scale0으로목록에서제외된다. C의raw force값은side effect없는함수를진단용으로만계산하여기록한다. 기여도와학습반환reward에는0으로반영한다.

## 동일 초기화

Seed 기본1. seed_1/initial_policy.pt는학습전공통가중치이며생성시독점쓰기,기존파일덮어쓰기금지. 각actual runner에strict load하고모델hash·빈두Adam state·iteration0·PPO counter0을검사한다. 세조건의initial/final model hash는모두동일했다.

Model SHA256: `fd0f3dfcbf5343c3f3e0a699ca50e71e4389fd069cfc445c8d09a90acec16f0f`

실제DR mass/COM/friction/motor strength tensor SHA256도모두동일했다: `f5509ed9f771a4bc48d0a4868219b7e0956f34b6eacba6f8f1522ade09b3c7fd`.

따라서초기파라미터나DR설정차이가아닌force reward차이로학습비교할수있다. 단동일seed라도GPU PhysX의개별물리궤적bitwise재현을보장하지않는다. 더강한통계결론에는같은seed묶음의A/B/C를반복해야한다.

## 검증 실행 범위

- prepare: 공통초기가중치와전체조건config 저장.
- A/B/C 각각 actual runner/model/storage 생성, 같은weight load,16환경×120step(2.4초) PD/IK rollout. 검증용terrain만4x4·border5m으로축소했으며DR는원래활성범위를유지했다. 검증override는세조건동일하고훈련config에저장하지않는다.
- 실제호출raw×scale합이native leg/arm reward와일치함을매step검사했다. Stateful reward는재호출하지않는다.
- 모든조건의정지shaping force/vel=0,이동명령에서는음수penalty. Force source는net contact force, EE source는실제TCP이다.
- 미학습 model_0 fixture로3초의checkpoint-load/evaluate경로도실행하여정지·전진·회전·복합각8환경의metric JSON/NPZ생성을확인했다. 이결과는미학습정책이므로성능평가결과로사용하지않는다.
- actual train mode와learn()은한번도실행하지않았다. Optimizer step0,가중치불변. 장시간학습시작전설정/계측검증이다.

## 평가·기록 지표

Training중native24step logging에속도RMSE(vx/vy/yaw)·실제속도·EE position mean/P95·timeout/실패reset·1000env-step당실패·발net force norm평균/P95/P99(pooled및발별)·모든활성reward raw/weighted/signed share/absolute share를추가했다. 원본loss/episode 로그는유지한다. ablation_metrics.jsonl과동일W&B step에저장하며project는as2piper-low다.

weighted는/100이전,abs share=E|weighted term|/ΣE|weighted term|. C의force contribution은항상0. Force통계는비접촉0도포함하는norm이고Fz/6축sensor norm과다르다. Rough net contact의보고누락가능성때문에독립실측접촉정답으로보지않는다.

후속평가는같은seed·terrain·duration·iteration으로각조건checkpoint를로드한다. 기본flat/rough각32환경,4명령조건,40초중초기2초제외,고정중앙EE궤적,평가에서만DR/push off. 학습manifest의condition/source hash가다르면거부한다. 자세한명령은README_KO.md.

## 파일·보존

- configs.py: 세조건/태스크, runtime.py: 계측환경·원본runner logging확장, run.py: prepare/validate/train/evaluate.
- experiments/contact_force_ablation/seed_1/: 공통미학습weight와전체config.
- experiments/contact_force_ablation/validation/*/result.json,samples.npz: 조건별검증결과.
- experiments/contact_force_ablation/evaluations/*/: 평가경로검사결과(미학습fixture).
- verification_summary.json,preservation_before.json: 기존B1 checkpoint·현재No-contact production code·PPO·URDF 보존검사.

기존파일hash보존을확인했다. Git status는기존dubious ownership오류라설정을변경하지않고hash로확인했다. 사용자검토전장시간학습은시작하지않는다.
