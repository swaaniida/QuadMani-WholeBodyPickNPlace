# Reset / Push 검증

## 적용한 수정

- `check_termination`이 config의 roll/pitch/높이 임계값을 실제로 읽도록 수정했다. 원본 코드의 0.8/0.8/0.1 하드코딩을 제거했다. 임계값 수치는 바꾸지 않았다.
- 높이 종료는 월드 Z에서 발밑 지면 대비 몸통 높이로 변경했다. 지면은 기존 heightfield grid에서 몸통 XY 아래 세 정점의 최소값을 사용한다. 평지는 0 m다. 국소 높이는 격자 해상도에 따른 근사다.
- 종료 원인 contact/roll/pitch/height/timeout과 종료 직전 실제 각도·월드 높이·지면 높이·상대 높이를 기록한다. 정상 episode 로그에도 termination_* 원인 비율과 상태 평균을 추가했다.
- Push는 world XY root 속도를 순간 덮어쓰는 원본 방식이다. 실제 이전 속도·적용 속도·delta velocity·정지 판정·배율을 last_push에 기록한다.
- 정지 판정은 명령 벡터의 모든 성분이 0인지 확인한다. 원본의 commands.sum()==0 조건이 +전진/−yaw 등 부호 상쇄 명령까지 정지로 분류하는 문제를 수정했다.
- 정지 push 배율을 설정으로 분리하고 기본값을 2.5에서 1.0으로 조정했다. 기본 축별 속도 상한은 0.39639 m/s이며, 2.5배 stress는 축별 0.99097 m/s다. 속도 2.5배는 같은 질량에서 운동에너지 6.25배에 해당한다.

## 높이와 종료 조건

| 항목 | 실제 적용 |
|---|---|
| Reset 시작 Z | env origin + 0.3264069024 m |
| 접지 자세의 기하학 높이 | 0.3244069024 m |
| 높이 종료 | root world Z − 몸통 아래 지면 높이 < 0.10 m |
| Roll 종료 | 절댓값 > 0.8 rad, 약 45.84° |
| Pitch 종료 | 절댓값 > 0.8 rad, 약 45.84° |
| 접촉 종료 | 선택된 body 힘 > 1 N. 현재 terminate_after_contacts_on=[]이므로 활성 body 없음 |
| Timeout | episode_length_buf > max_episode_length |

Reset Z는 patch origin 기준이다. Rough에서는 초기 XY 변동과 지면 형상 때문에 실제 첫 프레임의 지면 상대 높이가 0.3264 m와 다를 수 있다. 높이 reward의 기준값은 종료 임계값과 별개다.

## 독립 강도 시험

각 조건별 32개 환경, 12초, full DR, 랜덤 EE 목표, 8초에 push 1회. 다리 기본 PD와 팔 IK를 사용했다. 학습된 보행 정책은 사용하지 않았다. 0배는 속도 덮어쓰기도 하지 않는 무외란 대조군이다. 외란 강도마다 별도 프로세스로 실행했다.

| 지형 | 배율 | 축별 속도 상한 (m/s) | 생존 환경 | Reset 횟수 | 종료 원인 |
|---|---:|---:|---:|---:|---|
| plane | 0 | 0.000 | 32/32 | 0 | {} |
| plane | 1 | 0.396 | 32/32 | 0 | {} |
| plane | 1.5 | 0.595 | 32/32 | 0 | {} |
| plane | 2.5 | 0.991 | 28/32 | 4 | {'pitch': 3, 'roll': 1} |
| trimesh | 0 | 0.000 | 32/32 | 0 | {} |
| trimesh | 1 | 0.396 | 32/32 | 0 | {} |
| trimesh | 2.5 | 0.991 | 26/32 | 6 | {'roll': 5, 'pitch': 1} |

- 모든 최종 시험에서 push 이전 reset은 0회였다. 기본 1배는 평지·rough 모두 32/32 환경이 끝까지 유지됐다.
- 평지 2.5배의 4회 reset은 pitch 3회, roll 1회다. Rough 2.5배의 6회 reset은 roll 5회, pitch 1회다. 높이·접촉·timeout으로 종료된 환경은 없었다.
- 높은 강도의 push에 따른 실제 자세 변화가 종료를 유발했다. 임계값 완화로 reset을 숨기지 않고 정지 배율을 1배로 조정했다. 1.5배는 평지에서만 시험했으며 32/32 환경이 유지됐다.
- 평지 강도별 초기 상태 및 DR 해시는 동일했다. Rough는 Python/NumPy/Torch 시드와 지형 hash를 고정했다. 첫 physics step 이후 전체 상태 hash는 rough 실행 간 다를 수 있어 완전한 bitwise 상태 복제를 보장하는 비교는 아니다.
- 난수 seed는 초기화 20261111, push 벡터 20261112다. Rough는 원본 rough 생성기를 4×4 패치로 줄여 평가했다. 이 유한 표본 결과를 장시간·학습된 보행의 안정성 보장으로 해석하지 않는다.

## 실제 임계값 및 정지 판정 확인

- 정상 상태, roll 초과, pitch 초과, 높이 미달, timeout, 접촉 힘 주입, config roll 임계값 변경, 지면 Z=10 m 오프셋의 8개 검사에서 실제 종료 함수를 호출하고 원인을 assertion으로 확인했다.
- 접촉 검사는 검증 중 base_link를 임시 선택해 1.01 N을 주입했다. 운영 config의 접촉 종료 body 목록은 그대로 비어 있다.
- 정지/부호 상쇄 이동/일반 이동 명령에 실제 push 함수를 적용해 배율 [2.5,1,1]을 확인했다. 결과는 command_detection.json에 있다.

## 파일과 재현

- `audit_reset_push.py --multiplier {0,1,1.5,2.5} --terrain {plane,trimesh}`로 실행한다. Python 3.8 및 run_head.sh와 같은 PYTHONPATH가 필요하다.
- 각 하위 폴더의 result.json에 이벤트별 원인·각도·높이·발/몸통/비발 접촉·적용 속도가 있다. traces.npz에는 시간별 상태·종료 mask가 있다. summary.json은 비교 요약이다.
- 원본 보존 및 보호 URDF hash는 preservation.json에 기록했다.

이번 단계는 Reset/Push에 한정했다. Reward, PPO 및 학습 smoke 검증은 다음 확인 이후 진행한다.
