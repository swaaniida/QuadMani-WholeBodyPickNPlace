# 후보 bank 및 IK 유효성 제한 철회

- 제거: 학습/배포 실행 경로의 bank import/로딩, 자산 hash gate, 목표 ID·전이 graph·bank reset, bank 전용 위치 연동 pitch. active helper와 JSON을 제거하고 before/에 보존.
- 복구: 연속 구면 위치 추출 + 독립 orientation delta, 최대 10회 단순 EE 경로 충돌 박스/지면 검사. 목표 생성 시 IK·CoM·전신 자세·토크 유효성 검사 없음.
- 유지: 팔 제어 DLS IK, 지면 기준 가상 목표 좌표계, 실제 Arm Base/TCP, 20DOF/18action, 755/737 입력, 최신 카메라 URDF, 박스 전방 +20mm, 고정 gain400/40, effort100, 속도/지연 처리.
- Reward·PPO·DAgger·ROA 및 다른 DR 수정 없음. 기존 45k 코드·checkpoint 보존.
- 샘플링 범위: r0.40~0.48m, 고도각0.10~0.50rad, yaw±0.25rad, orientation delta각±0.08rad. 바닥~상단 범위 확장과 새 방향식 확정은 이번 변경에 포함되지 않음.
- 이전 workspace 분석/graph 검증은 역사 기록으로 보존하며 실행 시 참조하지 않음.

검증: preservation_verification.json 및 physx_verification.json. CPU 소규모 무학습 검사만 수행, optimizer update 없음. git status는 dubious ownership 오류, 전역 설정 변경 없이 파일 해시/별도 patch로 확인.

확인 결과: 연속 목표 400개 생성, 단순 충돌 검사 100회·목표 유효성 IK 호출 0회. PhysX CPU 4환경·2초 무학습 zero-action hold: 유한값, reset0, 관측755/action18/DOF20 유지, 실제 arm gain400/40 확인. 보행/넓은 EE 추종 성능 검증 결과가 아님.
