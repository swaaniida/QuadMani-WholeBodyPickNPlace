# 바닥~상단 연속 구면 생성 범위 적용

- 기존: r0.40~0.48m, elevation0.10~0.50rad, yaw±0.25rad. 지면 높이 약0.496~0.686m, yaw0 기준 전방 최대 약0.702m.
- 변경: r 외접 범위0.22565~0.8682391m, elevation−1.0934979~1.0503443rad, yaw±0.30rad. 단순 독립 외접 범위 추출이 아닌 연속 고도각→해당 각도의 기하학적 반경 구간 추출.
- 지면 z0.02~0.85m, 가상 중심에서 수평거리0.22565~0.75065m(전방 기준 nominal x0.45~0.975m). yaw≠0에서 실제 x는 center_x+(nominal_x-center_x)cos(yaw). yaw0에서 최대전방0.975m.
- 범위식: rho=r*cos(phi), z=center_z+r*sin(phi). phi를 연속 추출하고 rho/z 경계를 만족하는 반경 하한~상한에서 연속 추출. 허용 공간 내 부피 균일 분포가 아님. IK로 유효성을 제한하지 않음.
- 방향: 상단/홈은 기존 pitch=arm_induced_pitch−phi 유지. z0.25→0.10m에서 바닥 방향과 혼합. 바닥 방향은 nominal x0.55→0.975m에 따라175→150°. 기존 roll/pitch/yaw delta 각±0.08rad 유지. 시각화6개 목표의 명목 방향식과 정합.
- 기존 충돌 박스(전방+20mm)·지면 검사 유지. 구면 보간 경로의 높이도 같은 가상 지면 기준0.02~0.85m 검사.
- 확인된 기존 오류 수정: 재추출10회 모두 거절해도 마지막 거절 목표가 남던 경로를 이전 목표/방향 유지로 수정. IK/bank 검사나 샘플러 추가 반복 없음.

## 검증

- 실제 메서드로122880개 목표 생성: 전방 nominal x0.4500005~0.9749904m, 지면 높이0.0200011~0.8499978m. 수학적 생성 경계 확인이며 전체 IK 도달 가능성 인증이 아님.
- 실제 환경400개 목표 재추출: 위치 경로 충돌/지면 검사 통과, 목표 유효성 IK 호출0회, bank없음. 강제10회 실패 시 기존 목표/방향 보존.
- CPU PhysX4환경·2초 zero-action hold: reset0, 유한값, 755obs/18action/20DOF, gain400/40. 확장 작업공간 전신 추종이나 동적 보행 성능 검증이 아님.
- Reward/PPO/DAgger/ROA/DR/gain/URDF/관측/action 차원 유지. 원본 source39개, checkpoint226개 metadata 보존. 실제 학습/optimizer update0.

## 파일

- geometry_verification.json, physx_verification.json, expanded_workspace.patch.
- 이전 이미지 ../..(프로젝트루트 reports/workspace_limits_20261115)는 변경 전/확장 검토 비교 자료. 이번에 그 확장 위치 영역과 후보 방향식을 실제 생성기에 적용.
- git status는 기존 dubious ownership 오류. 글로벌 설정 변경 없이 원본 백업·해시·unified diff 확인.
