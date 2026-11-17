# Standing Arm Base 중심 및 확대 후보 도달성 검사

## 적용한 변경

- 가상 중심을 URDF FK의 nominal standing piper_base_link 원점으로 설정: [0.22435, 0, 0.4563069024246581]m. z=standing base height0.3244069024246581 + fixed mount0.1319. reset 높이0.3264069024를 사용하지 않음.
- 기존 가상 중심[0.28,0,0.50] 대비 전방55.65mm·높이43.69mm 감소. 몸통 높이/roll/pitch 불변 특성, 실제 Arm Base 관측/제어 IK 경로 유지.
- 초기 EE 위치[0.614764581,0,0.631445574]m와 방향 유지. 초기 구면[0.4278984687,0.4216862841,0], pitch offset2.4552161377으로 환산.
- 실행 설정은 현재 r0.35~0.65m·좌우±0.30rad 유지. 높이0.02~1.00m 유지를 위해 고도각만 새 중심에 맞춰 재계산. 확대0.80m/±1.20rad는 아래 미검증 표본이 남아 실행 설정에 미적용.
- 실제 sampler122880개 CPU 검사 통과. 초기 EE 위치/방향 및 나머지 sampler 함수 불변 확인. 새 중심의 현재 범위 정적 명목방향45개 통과.

## 확대 후보

- r0.35~0.80m, yaw±1.20rad(VBC와 동일), elevation−0.5768467347~0.7472395459rad(−33.05~42.81°), 높이0.02~1.00m. 독립 구면 추출 유지.
- yaw0: 바닥z0.02m 목표 x0.894899m, 상단z1m 목표 x0.811204m, 중간 높이z0.456307에서 최대x1.02435m. 측면 목표에서는 전방성분이 줄어듦.
- 최신 카메라 포함 URDF 사용. 정적 full-body Leg IK + Arm6D IK, hard joint limits, 발 위치오차<2mm, TCP 위치<3mm·방향<2°, 비인접 링크/지면 관통>1mm 제외, CoM 지지다각형 안쪽20mm 이상 검사.
- 기본 탐색: 몸통 높이0.25~0.39m, roll±20°, pitch−25~35°, XY/yaw 고정, 네 발 위치 고정. 미통과는 높이0.20~0.42m, roll±35°, pitch−35~50°로 확장하고 더 긴 수치해 탐색.

| 표본 | 수 | 고정발 최종 통과 | 발 배치 조정 포함 |
|---|---:|---:|---:|
| 위치 grid | 75 | 71 | 73 |
| 임의 자세 seed20261115 | 96 | 94 | 94 |
| 방향 delta 극단 ±0.08rad | 96 | 96 | 96 |

- 합계267개 중 고정발261개, 발 배치 조정 후2개 추가=263개 통과. 발 XY 조정은 축별±50mm 범위 + 수치 오차<2mm, 실제 이동 최대 약50.12mm. 조정된 실제 발 위치로 CoM 지지다각형 재계산. 이는 새로운 정적 지지 자세이며 stepping 전이/접촉 동역학 검증은 아님.

## 남은4개 및 원인 분리

| 목표 x/y/z(m) | 최종 위치오차(mm) | 방향오차(°) |
|---|---:|---:|
| 1.0003, 0.0000, 0.2616 | 13.06 | 3.18 |
| 1.0214, 0.0000, 0.5244 | 14.75 | 3.32 |
| 0.9546, -0.1030, 0.2774 | 3.49 | 0.95 |
| 0.9852, -0.1006, 0.2757 | 17.26 | 4.29 |

- 미통과점은 전방 최대 반경의 중간 높이에 집중. 기본/확장 고정발 해에서 뒤 다리 calf 상한, 일부 목표에서는 arm joint5 상한이 함께 활성화됨.
- 위4개 모두 방향 제약을 제거한 위치-only 재검사에서는 발 고정 상태로 위치·관절한계·충돌·CoM 조건 통과. 따라서 위치 자체가 닿지 않는다는 결론은 부정확함. 현재 지정 그리퍼 방향과 전신 자세 제약을 동시에 만족하는 해를 이번 탐색에서 찾지 못한 상태.
- 반경을 바로 축소하기 전에 위치-그리퍼 방향 연동식이 해당 원거리 중간 높이에서 적절한지 검토할 근거가 됨. 방향식은 이번에 임의 변경하지 않음.
- 유한 수치 탐색의 미발견이며 전역적으로 도달 불가능하다는 증명 아님. 통과 표본도 모든 연속 목표/목표 사이 경로·rough·정책 추종·실기 토크 안정성 보장이 아님.

## 보존 및 산출물

- Reward/PPO/DAgger/ROA/DR·20DOF/18action·755/737obs·400/40gain·effort·URDF·초기 관절각 유지. 온라인 IK 유효성/bank 없음. Ground Proximity/Valid Contact 미적용. 학습/optimizer update0, GPU 사용 없음.
- 보호 source60개 SHA256 및 기존 checkpoint226개 크기/mtime 일치. 기존 B1 baseline 수정 없음. Git dubious ownership은 전역 설정을 바꾸지 않고 백업/해시/unified patch로 대체.
- before_*: 복구본. spherical.patch: 이번 config 변경. grid/random/corners.json: 원시 탐색. refine_*_relaxed.json: 확장 몸통 탐색. step_refined.json: 발 배치 조정. position_only.json: 방향 제약 제거 진단. verified_poses.json/summary.json: 통합 결과. verification.json: 실제 sampler 검사.
- 시각화: reports/standing_armbase_reach_20261115/workspace_reachability.png. 보라색 원은 가림 없이 표시한 실제 standing Arm Base 원점, 주황은 미적용 확대 후보, 빨강은 남은4개 표본.
- 재검증: Python3.8+torch로 verify.py, 기존 unitree_rl_mjlab/.venv/bin/python으로 summarize.py. solver는 check.py grid/random/corners/refine, step_refine.py, position_only.py. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 사용.
