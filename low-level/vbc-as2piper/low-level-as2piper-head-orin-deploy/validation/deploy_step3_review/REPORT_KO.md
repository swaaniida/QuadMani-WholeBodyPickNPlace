# 3단계 사전 검토 — 정지 Ground Proximity / Valid Contact

**상태: 사전 검사 및 후보식 계산 완료. production reward/config/PPO 수정·학습 없음. 적용 전 사용자 검토 대기.**

## 현재 문제와 확인 경로

- `stand_still`: exp(−0.05×다리12관절 기본각 절대편차합), scale+1.0. 네 발 접지 여부를 평가하지 않음.
- 정지에서 contact shaping(force/vel), feet_height, feet_air_time은0. Contact-force 항목은 과도한 힘만 벌하고, drag는 이미 접촉한 발의 움직임만 벌함. 따라서 공중에 정지해 있는 RR에 직접적인 지속 무접촉/높이 비용이 없음.
- 45k의 기존 실물리 기록(정지 base 명령, 바닥 EE구간12~60초, 8replicas): RR 평균27.7mm/P95약45mm 들림, 힘norm>2N 접촉0%. 20k/28k/30k에서도 문제 확인.
- 이 기록은 학습 당시 카메라 추가 전 정확한 URDF로 평가한 결과. 최신 카메라 URDF로 새 policy를 평가한 결과가 아님. 기존 기록 재분석이므로 새 GPU 사용 없음.

## 제안 — 기존 보상 유지, 두 penalty만 추가

1. Ground Proximity: 최근0.5초의 발별 최소 지면 간격에서5mm 허용폭을 뺀 뒤20mm로 정규화, 제곱 후[0,1] 제한, 네 발 평균. 지속적인 발 들림에만 비용을 부여. 짧은 스텝 중 최소값이 정상 접지를 포함하면 비용0.
2. Valid Contact: 양의 Fz>2N를 현재/이전 프레임으로 필터링하고, 최근0.5초 접촉 비율을 계산. 비율0.6 미만의 부족분을 정규화해[0,1] 제한, 네 발 평균. 세 발의 접촉력이 커져도 RR의 접촉 부족을 대체할 수 없음. 힘을 네 발에 동일 배분하도록 강제하지 않음.
3. 정지 명령에만 적용, 기존 walking mask 공유. 다리 Leg reward 경로에만 추가. 배포 관측/센서 입력이나 policy 차원 변경 없음; 접촉력은 시뮬레이션 net contact force 사용.
4. 후보 scale은 각각−0.5. 둘 다0~1 비용이므로 전체 최대 추가−1.0, 이상적인 지속 RR30mm 들림/무접촉은−0.25(기존 /100 적용 후−0.0025/step). 원본 scale 변경 없이 새 항의 시작값 후보이며 학습 최적값 검증이 아님.
5. reset 직후 정상 접지 이력 확보까지의 warmup 및 알려진 push 직후 회복 구간은 별도 처리. 시간/기준은 아직 production 적용하지 않음. 속도가 커지면 비용을 단순0으로 하는 gate는 정책이 계속 움직여 회피할 수 있어 권장하지 않음.

## 기존 물리 기록에 후보식을 계산한 결과

| Checkpoint | RR 평균 들림(mm) | RR 접촉률 | Proximity 비용 | Contact 비용 | scale각−0.5 추가합(/100 전) |
|---|---:|---:|---:|---:|---:|
| 20000 | 10.8 | 3.3% | 0.036 | 0.238 | -0.137 |
| 28000 | 30.0 | 2.0% | 0.201 | 0.262 | -0.232 |
| 30000 | 11.6 | 21.0% | 0.063 | 0.212 | -0.137 |
| 45000 | 27.7 | 0.0% | 0.183 | 0.250 | -0.216 |

기존 trajectory에 보상만 사후 계산한 결과이며, 정책이 바뀌거나 보행/EE 추종이 개선됐다는 증거가 아님. 원본 stand_still/Leg 총합 약0.956/4.352는 별도의 in-distribution 정지 평가 참고값으로, 위 바닥 trajectory에 대한 동일조건 기여도 비교가 아님.

## 회복 스텝 계산 시험

| 합성 조건 | scale각−0.5 추가 보상 평균 | 최소 |
|---|---:|---:|
| four_feet | 0.0000 | -0.0000 |
| persistent_RR_lift | -0.2500 | -0.2500 |
| brief_120ms_RR_step | 0.0000 | -0.0000 |
| long_400ms_RR_step | -0.0090 | -0.0750 |
| repeated_short_taps | -0.0836 | -0.0917 |

합성 발 높이/힘 신호를 넣은 보상 논리 시험으로 실제 push rollout 아님. 네 발 정상 접지와120ms 스텝은0, 지속 RR 들림은벌점, 접촉을 짧게 반복하는 tapping도contact비율 항에서벌점. 모든 외란 회복을 무벌점으로 보장하지 않음.

## 적용 전 해결할 Rough 기준 오류

- 기존 `_reward_foot_ground_heights()`는5cm격자에서3개높이 최솟값 사용. 경사 각도/격자 내 위치 오차 및 발구의 접촉 법선 보정이 없음.
- 실제 함수를 AST로 실행한30°평면 기하학 시험: 실제 접촉 sphere의 지면 간격0인데28.05mm로계산. 5mm허용폭의Ground Proximity에 그대로 넣으면 정상접지 오벌점 가능.
- 실제 terrain은 fast mesh·downsample을 사용하므로 원래 heightfield와 충돌메시의 차이도 따로 확인해야 함. 이 분석은 현재 rough모든지형의 실제 오차를 계측한 결과가 아님.
- 적용 권고: **새 정지 보상 전용으로 실제 충돌 terrain mesh의 표면/법선을 참조하는 ground 기준을 먼저 검증**. 접촉 시 발높이 anchor는 보조 대안이지만 디딤 위치가 바뀌면 같은 지면으로 간주하지 않도록 별도 검증 필요. 기존 feet_height/base_height 보상 경로는 유지.
- Known push/recovery·reset/warmup·rough false positive 검사 후 후보−0.5/−0.5 적용 여부 결정. 실제 짧은 PPO로 안정화/스텝 허용 여부를 판단하는 것은 사용자 검토·허가 후 진행.

## 보존 / 재현

- 원본 B1+Z1, 기존45k 코드·checkpoint 수정 없음. 변경 파일은 이 validation 분석/기록 및 CHANGES_KO.md뿐. Production4핵심파일 hash 검사 전후 일치.
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/usr/bin/python3 low-level-as2piper-head-orin-deploy/validation/deploy_step3_review/analyze.py
```
