# 높이1m 범위 및 한계 자세 재시각화

- 현재 설정의 높이 상한1.00m 유지. 전방 nominal x0.45~0.975m, 지면 높이0.02~1.00m, yaw±0.30rad. 이번 작업은 보고서/분석/이미지만 생성, 학습 코드/URDF 변경 없음.
- `workspace_comparison_3d.png`: 파랑=이전 학습 범위, 주황=현재 생성 범위. 중간 단계 기준선 없이 두 범위만 비교. 회색은 연속 생성 예시이며 IK 통과 점이라는 의미가 아님.
- `extreme_poses_2x3.png`: 960×640 원본 렌더 크기 유지, 2행3열. 개별 `pose_1.png`~`pose_6.png`.
- 목표 좌표: 초기 몸통 XY0/yaw0, 지면 기준 고정 좌표. 초기 가상 중심은(0.22435,0,0.4563069024246581)m. 현재 sampler의 명목 그리퍼 방향식, orientation delta0.
- 현재 URDF의 fixed-foot Leg IK 가능 몸통 후보30개: h0.25/0.27/0.30/0.3244/0.35/0.37/0.39m 중 다리 한계 내 해 존재 자세, pitch0/−15/−25/+15/+25/+30/+35° 조합. 몸통 XY/yaw와 발 위치 고정, 몸통 roll은0. 고정 발/유한 후보 검사이며 다른 디딤/roll까지 전역 탐색한 결과가 아님.
- 실제 actuator 제어/학습 policy의 추종 영상이 아닌 정적 기구학적 자세. 충돌은 URDF collision(convex 근사), 같은 fixed 조립체/인접 링크 제외, 비인접 관통1mm 초과 거부. 정적 CoM 여유20mm와 중력 wrench 평형 대조. 기존5°관절여유 필터는 사용하지 않고 hard joint limits 준수.

| 목표 | XYZ(m) | 도달 자세 | 몸통 높이/pitch | 위치오차(mm) | 방향오차(°) |
|---|---|---|---|---|---|
| ① 최전방 바닥 | 0.975/0.000/0.020 | 도달 확인 | 0.270 / 30° | 0.000 | 0.000 |
| ② 좌측 전방 바닥 | 0.941/0.222/0.020 | 도달 확인 | 0.270 / 35° | 0.000 | 0.000 |
| ③ 우측 전방 바닥 | 0.941/-0.222/0.020 | 도달 확인 | 0.270 / 35° | 0.000 | 0.000 |
| ④ 가까운 상단 1.0m | 0.450/0.000/1.000 | 도달 확인 | 0.324 / 0° | 0.000 | 0.000 |
| ⑤ 전방 상단 1.0m | 0.750/0.000/1.000 | 도달 확인 | 0.324 / 0° | 0.000 | 0.000 |
| ⑥ 최전방 상단 1.0m | 0.975/0.000/1.000 | 이번 탐색 미도달 | 0.390 / 0° | 138.000 | 3.980 |

5개는 위치오차<0.01mm·방향오차<0.001°. 최전방 상단은 최선 후보의 실제 TCP와 목표를 분리하고 주황선으로 오차 표시. 이138mm 오차는 지정한 방향을 동시에 맞추려는 이번 탐색 결과이며 위치만 최적화한 최소 거리나 전역 도달 불가능성 증명이 아님.
목표 생성 영역/위치·방향 연동은 유지하고 IK 유효성 필터/bank를 재도입하지 않음. 동적 보행/정책 추종/DR/물체·payload/실기 토크 검증 결과가 아님.

재현:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
unitree_rl_mjlab/.venv/bin/python reports/workspace_height1m_20261115/solve_poses.py
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
unitree_rl_mjlab/.venv/bin/python reports/workspace_height1m_20261115/render_workspace.py
```
