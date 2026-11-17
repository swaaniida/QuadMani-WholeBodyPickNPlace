# B 조건 20,000 iteration 체크포인트 검증

기록 UTC: `2026-11-12T13:50:13.478116+00:00`

## 판단

**10,000회보다 제어 성능이 개선됐으며 기존 설정으로 학습을 계속하는 것이 타당하다.** 모델·loss의 명확한 발산이나 지속적인 정지 정책은 관찰되지 않았다. 보상·PPO·DR·진행 중 학습은 변경하지 않았다.

- 정지 yaw 속도 RMS: 평지 0.538→0.030rad/s, rough 0.515→0.028rad/s. 양쪽 모두 약94% 감소.
- 전진 0.4m/s 명령: 평지 0.673→0.584m/s, rough 0.704→0.581m/s. 개선됐으나 약45~46% 과속이 남음.
- 평지·rough 전체 8조건에서 실패 reset 0건. 각 지면의 timeout 96건은 정상 10초 episode 종료.
- EE 평균오차 약1.5~1.6cm 유지. 팔은 기존 IK/position drive를 포함한 시스템 결과이며 arm actor 단독 성능이 아님.
- 접촉력 tail은 일관되게 개선되지 않음. 평지 전진 P99 229→295N, 회전 183→212N, 복합255→277N으로 증가.

## 평가 정합성

- checkpoint: `/data/team/users/suhwan/vbc-as2piper/low-level-as2piper-head-orin-nocontact/experiments/contact_force_ablation/runs/as2piper_headpiper_orin_200N_0.18m/model_20000.pt`
- SHA256: `63e27acd37ce0795e98c1246200f18b3b2ac67ee26108e1930b09f3d777a01f9`
- 10,000회 평가와 같은 source hash·seed1·평가 코드·GPU PhysX·history 추론 경로. 모델 strict load와 평가 전후 모델 hash 일치, optimizer update0회.
- 지면별 32환경, 정지/전진/회전/복합 각8환경, 40초. 초기2초 제외하여 조건별15,200 env-steps. DR/push는 평가에서만 비활성.
- 명령 (vx,vy,yaw): (0,0,0), (0.4,0,0), (0,0,0.6), (0.4,0,0.6). 중앙 EE 구면 궤적과 orientation delta0은 10,000회와 동일.
- 시작 전 MIG 메모리 20,711/40,192MiB, 평가 중 약23,248/40,192MiB. 메모리 여유를 확인한 뒤 평지와 rough를 순차 실행. 영상·추가 학습은 실행하지 않음.
- 실패 reset은 활성 roll/pitch/height 종료 기준에서 timeout을 제외한 값이다. 몸통 접촉 종료는 현재 설정에서 활성화되지 않았으므로 실패0만으로 모든 비정상 접촉이 없다고 해석하지 않는다.
- 한 seed·제한된 명령·외란 없는 시뮬레이션 비교이며 실물 배포·ground picking·강한 push 안정성 검증은 아님.

## 고정 명령 제어 비교

|지면|명령|실제 vx 10k→20k (m/s)|yaw RMSE 10k→20k (rad/s)|EE 평균/P95 20k (cm)|실패 reset 10k→20k|
|---|---|---:|---:|---:|---:|
|flat|정지|-0.065→-0.002|0.538→0.030|1.54/1.80|0→0|
|flat|전진|0.673→0.584|0.167→0.063|1.54/1.97|0→0|
|flat|회전|0.004→-0.035|0.121→0.070|1.54/1.89|0→0|
|flat|전진+회전|0.663→0.571|0.225→0.072|1.54/2.09|1→0|
|rough|정지|-0.036→-0.002|0.515→0.028|1.55/1.81|1→0|
|rough|전진|0.704→0.581|0.266→0.125|1.57/2.35|0→0|
|rough|회전|0.019→-0.021|0.180→0.114|1.57/2.22|0→0|
|rough|전진+회전|0.684→0.571|0.214→0.117|1.56/2.34|0→0|

순수 회전 시 전후 drift는 20,000회 평지 -0.035m/s, rough -0.021m/s이다. 회전 평균 yaw는 0.6rad/s 명령에 평지0.593, rough0.580rad/s.

## 접촉력·Reward

발 4개와 전체 시점의 시뮬레이터 net force norm을 풀링한다. 비접촉 0 포함. Weighted reward는 실제 합산의 `/100` 이전 값이다.

|지면|명령|20k 평균/P95/P99 (N)|force weighted reward|force 절대 기여 (%)|
|---|---|---:|---:|---:|
|flat|정지|57.2/107.6/127.0|0.00000|0.000|
|flat|전진|63.6/155.0/295.0|-0.00605|0.110|
|flat|회전|58.4/161.4/211.6|-0.00325|0.060|
|flat|전진+회전|62.5/162.6/276.8|-0.00451|0.081|
|rough|정지|18.0/103.1/146.6|-0.00040|0.008|
|rough|전진|16.2/118.9/174.7|-0.00137|0.025|
|rough|회전|21.1/126.8/187.4|-0.00145|0.027|
|rough|전진+회전|21.0/131.9/186.0|-0.00165|0.029|

- Contact Shaping force/vel 모두 정지에서 정확히0, 이동에서 음수 페널티 확인. 센서 observation은 제거 상태 유지.
- Contact force reward는 활성이고200N 초과분에 페널티를 부여한다. 200N은 물리적 상한이 아니므로 P99가200N을 넘을 수 있다.
- Rough 평균 net force는 평지보다 현저히 작다. 기존 trimesh 접촉 추정 문제 때문에 실제 하중이나 지면별 물리적 안전성을 이 수치로 직접 비교하지 않는다.
- 원본 전진 reward는 목표 이상의 속도에서 포화하므로 과속이 남아도 높은 reward가 가능하다. 과속의 단독 원인은 이번 평가로 확정하지 않음. 보상 수정은 수행하지 않았다.
- 모든 reward raw/weighted/기여도 및 물리 지표의10k→20k 비교는 `comparison.csv`, 원자료는 각 평가의 `samples.npz`/`result.json`에 보존.

## 학습 로그의 구간 평균 비교

서로 다른 무작위 명령·DR·push가 섞인 구간이다. 동일 명령의 짝지은 실험이 아니며, 아래 숫자는 iteration별 지표의 평균이다. P95/P99 평균은 전체 샘플을 합친 분위수가 아니다.

|지표|9,000~9,999|19,000~19,999|
|---|---:|---:|
|vx RMSE 평균 (m/s)|0.2255|0.2290|
|yaw RMSE 평균 (rad/s)|0.2828|0.3283|
|EE 평균 (cm)|1.8730|1.9607|
|실패 reset / 100만 env-steps|4.8218|3.8656|
|collision weighted|-0.1179|-0.0916|
|leg total /100 이전|3.0376|3.0774|

낙상 종료 빈도·충돌 페널티는 감소했으나, 혼합 학습 지표의 yaw/EE 평균은 소폭 증가했다. 고정 명령 평가의 개선과 구분하며 모든 조건의 일반화가 단조롭게 개선됐다고 주장하지 않는다.

## PPO·ROA 검증

|항목|10k|20k|
|---|---:|---:|
|표시 Value loss|0.0041|0.0040|
|표시 Surrogate loss|-0.0024|-0.0037|
|History latent loss|0.0174|0.0131|
|Privileged regularization loss|0.0157|0.0124|

- 모델 tensor 전체 유한값. Main optimizer state41개, step 예시380,000. 업데이트 누적 확인.
- History encoder/privileged encoder 가중치의10k→20k L2 변화량3.047/1.864. Adaptation module이 실제로 갱신됐음을 확인.
- 20,000은 `iteration %20==0`인 DAgger 갱신 회차이다. History loss는 해당 DAgger 값이며 PPO Value/Surrogate/privileged regularization 표시값은 직전 PPO 갱신의 유지값이다. Loss 크기 자체를 성능 점수로 해석하지 않는다.
- 현재 source/PPO/reward scale/DR/학습 프로세스는 변경하지 않았다. 기존 checkpoint와 B1 원본은 보존.
