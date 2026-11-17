# 독립 구면 샘플링 적용

## 적용

- 수정 대상: deploy 복제본의 `as2_head_orin_config.py`, `manip_loco.py`만 변경. 기존 nocontact 학습 코드와 45k checkpoint 보존.
- 수평거리/높이에 따라 반경을 바꾸던 원통형 영역 생성 분기 제거. 원본 VBC처럼 r·고도각·좌우각을 각각 독립 uniform 추출. 공간 부피에 대한 균일분포는 아님.
- 가상 중심: 몸통 XY + yaw 기준 전방0.28m, 좌우0m, 지면 기준 높이0.50m. 몸통 높이/roll/pitch 불변. 실제 Arm Base·관측·제어 IK 좌표계 변경 없음.
- 반경0.35~0.65m, 고도각−0.8307859155~0.8776364192rad(−47.60~50.29°), 좌우각±0.30rad(±17.19°). 목표 orientation delta 각±0.08rad 유지.
- 이전 영역은 nominal x0.45~0.975m와 z0.02~1.0m의 조합을 허용. 새 구면은 상·하단에서 전방 거리가 줄어 기존 먼 상단 모서리를 제외한다.
- yaw0일 때: 전체 최대 전방0.93m(z0.50m), 바닥z0.02m 목표 x0.718292m, 상단z1.0m 목표 x0.695331m. 바닥 최전방도 기존0.975m에서 줄어듦. 반경을 단순히 늘려 먼 바닥을 다시 넣으면 상단·중간 고도에서도 다른 목표가 늘어나므로 별도 재검사 필요.
- 초기 EE 실제 위치[0.614764581,0,0.631445574]m와 pitch2.033529854rad 유지. 초기 구면을[0.3596460256,0.3741548333,0]으로 환산하고 arm_induced_pitch를2.4076846869로 보정. 상단 pitch=offset−고도각 관계 유지, 바닥 pitch 연동식 유지.
- 기존 EE 경로 박스·지면/높이 검사·10회 실패 시 이전 목표 유지 동작 보존. 온라인 IK 유효성 검사·bank 추가 없음.

## 정적 검증

- 최신 카메라 포함 URDF SHA256: 7e37cadb8e55fddb50e6bcd10440c772997c72e11f72c7afddeef281954de02b.
- 위치 경계/중간63개 + 방향 delta 극단336개 + seed20261115 임의256개 =655개. 전체에 대해 목표별 전신 Leg IK+Arm6D IK 자세 발견.
- 검사: 팔/다리 hard joint limit, 발 위치 오차2mm 미만, TCP 위치3mm·방향2° 미만, 비인접 링크/지면 관통1mm 초과 없음, CoM 지지 다각형 안쪽20mm 이상.
- 최대 TCP 오차2.999mm, 방향1.870°, 발 위치 오차1.415mm. 최소 CoM 여유89.609mm.
- 몸통 XY/yaw/roll 고정, 높이0.25~0.39m·pitch−25~35° 안에서 최적화. 발 위치 고정. 모든 목표가 동일 몸통 자세에서 도달한다는 의미 아님. 관절한계 여유는 별도로 강제하지 않음.
- 초기 90회/seed 탐색에서 2개가 오차 기준 미달. 더 긴 오프라인 수치해 탐색으로 두 목표 모두 오차 사실상0까지 수렴. 원시 실패와 정밀화 결과를 각각 보존. 최종 verified_poses.json에 통합.
- 실제 런타임 sampler 메서드를 CPU에서 실행하여122880개 생성 경계·독립 추출·초기 위치/방향 불변 검증. 고도 실측 범위0.02069~0.99946m.

## 한계 및 보존

- 유한 표본의 평지 정적 자세 검증이다. 연속 범위 전체의 수학적 도달성 증명, 모든 목표 사이의 충돌 없는 이동, 학습 정책 추종, rough/동적 안정성·실기 토크 보증은 아니다. 반경과 중심만으로 이들을 보장할 수 없다.
- Reward·PPO·DAgger/ROA·DR·URDF·관측/action·제어 gain/effort/속도·초기 관절 자세 유지. Ground Proximity/Valid Contact 미적용. 학습/optimizer update0, GPU 사용 없음.
- 보호 source60개 SHA256 및 기존 checkpoint226개 크기/mtime 일치. git status는 기존 dubious ownership으로 실패하여 전역 Git 설정을 바꾸지 않고 백업+unified patch+해시로 대조.

## 산출물 / 재검증

- `spherical.patch`: 두 실행 파일 변경 diff. `before_*`: 변경 전 복사본.
- `grid.json`, `corners.json`, `random.json`: 원시 탐색. `refined_random.json`, `refined_corners.json`: 추가 정밀 탐색.
- `verified_poses.json`, `summary.json`, `verification.json`: 최종 자세/수치/실제 sampler 결과.
- 기존 Python3.8+torch: `PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 low-level-as2piper-head-orin-deploy/validation/spherical_reach/verify.py`.
- 기존 Python3.11+MuJoCo: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 unitree_rl_mjlab/.venv/bin/python low-level-as2piper-head-orin-deploy/validation/spherical_reach/summarize.py`.
