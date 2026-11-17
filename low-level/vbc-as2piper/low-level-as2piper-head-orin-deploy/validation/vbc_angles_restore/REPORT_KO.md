# 바닥 pitch 제거 및 VBC 위치 각도 적용

## 현재 적용

| 항목 | 이전45k AS2 학습 | 현재 deploy |
|---|---|---|
| 반경 | 0.40~0.48m | 0.35~0.80m |
| 위치 고도각 | 0.10~0.50rad(5.73~28.65°) | 원본VBC −π/2.5~π/3 (−72~60°) |
| 위치 좌우각 | ±0.25rad(±14.32°) | 원본VBC ±1.20rad(±68.75°) |
| Standing 기준 중심 | 실제Arm Base [0.22435,0,0.4563069024]m | 동일한standing Arm Base 가상중심 |
| 높이 상한 | standing 위치범위 약0.68643m | 0.4563069024+0.80sin60°=1.1491272255m |
| 방향 pitch | 기준각−고도각+delta | 동일한 식 |

- 비교 대상은 기존 `low-level-as2piper-head-orin-nocontact`의45k 학습 범위이며, 중간 검토안0.65m 등은 그림에 넣지 않음.
- `position_linked_floor_pitch` 및 floor_pitch_* 설정4개, `_get_ee_goal_pitch` 혼합 helper 제거. `_update_curr_ee_goal`의 pitch 대입식을 이전 학습 코드와 같은 직접 계산식으로 복원. 기준각2.4552161377, roll offset0, orientation delta각±0.08rad 유지. 위치 고도각과 orientation pitch를 혼동하지 않음.
- 반경·위치 고도각·좌우각은 각각 독립 uniform 추출. 구면 부피에 균일한 분포는 아님. 최대높이는 명목중심+최대반경·최대고도각으로 산출하며 기존1m 상한 제거. 현재 최대높이는1.149127225452209m.
- 기존 지면clearance 하한0.02m·몸통 충돌 박스·구면 보간 경로 검사·10회 거절 후 이전목표 유지 동작 보존. 수식상 최소높이−0.304538m의 지면 아래 후보는 최종목표로 채택하지 않음.
- 가상중심은 standing Arm Base 기준이며 현재 몸통 height/roll/pitch를 따라 움직이지 않음. 이전45k의 실제ArmBase 기반 중심은 당시 몸통 자세를 따라 움직였음. 그림은 양쪽 모두 nominal standing으로 놓아 위치범위 자체를 비교.
- 최신 카메라 포함 URDF 로봇 위에 파랑=이전45k, 주황=현재 범위 표시. 지면 위2cm 제한만 반영한 외곽선이며 몸통박스·시작목표별 경로 거절을 뺀 전체 도달 가능한 영역이라는 뜻은 아님.

## 검증

- 실제 sampler 메서드 CPU122880개: r/고도각/좌우각 bounds, 최대높이 수식, 유한값 확인.
- 실제 `_resample_ee_goal` + `_collision_check` CPU40960개: 단순 몸통박스·지면·경로높이 검사 통과. 실제 수용목표 높이0.020007~1.143289m(무작위 표본; 이론상한1.149127m).
- 원시 후보 중 약14.90%는2cm clearance 미만. 이는 최종목표 비율이 아니라 재추출 전 독립표본 비율.
- pitch 대입 AST가 이전45k 코드와 동일함 확인. 초기 EE 위치·방향, 가상중심, 나머지 resampling/충돌/center 함수 AST 불변 확인.
- Reward·PPO/DAgger/ROA·DR·관측/action·gain/effort·URDF 유지. 보호source60개 SHA256, checkpoint226개 크기/mtime 일치. 학습/optimizer update0, 온라인 IK 유효성 검사0. 기존 B1 baseline 수정 없음.
- 이번 요청은 설정/영역 시각화이며 확장 범위 전체의 전신IK·정책 추종·동적 검증을 새로 완료한 것이 아님.

## 이전 미통과4개에 대한 정확한 해석

- 이전 standing_armbase_reach 검사에서 남은4개는 목표높이0.2616/0.5244/0.2774/0.2757m로 전부0.25m 이상. 당시에도 바닥 pitch 혼합비율이0이었으므로, 바닥용 혼합식을 제거해도 이4개 목표의 지정방향은 변하지 않음.
- 따라서 바닥 혼합식이 이전4개 미통과의 원인이라고 결론내릴 수 없고, 이번 제거만으로 해당4개가 해결됐다고 해석하면 안 됨. 위치-방향 조합과 탐색한 전신/발배치 제약이 함께 남아 있음. 원시 결과는 보존.

## 산출물

- 시각화: `reports/ee_range_previous_vs_current_20261115/previous_vs_current.png`.
- before_config.py.txt, before_manip_loco.py.txt, before_CHANGES_KO.md: 변경 전 복구본.
- changes.patch: runtime2개 파일 변경 diff. Git status는 기존 dubious ownership으로 실패하며 전역 설정 변경 없음.
- verify.py / verification.json: CPU 실제메서드 검증 및 보호파일 검사.
- 재검증: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 low-level-as2piper-head-orin-deploy/validation/vbc_angles_restore/verify.py`.
- 렌더: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 unitree_rl_mjlab/.venv/bin/python reports/ee_range_previous_vs_current_20261115/render.py`.
