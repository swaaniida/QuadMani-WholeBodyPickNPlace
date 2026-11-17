# 이 스냅샷의 작업 지침

- 최신 상태는 docs/CURRENT_STATE_KO.md와 PROJECT_SNAPSHOT.json/각 run manifest를 우선한다. docs/historical/AGENTS.md는 초기 중앙 장착 문제의 역사적 문서다.
- 현재 개발은 low-level-as2piper-head-orin-deploy. nocontact200N·45k와 reference B1 및 그 checkpoint는 보존한다.
- 사용자 승인 없는 reward/scale/PPO/DR/초기자세/차원/URDF 변경, 학습 중지·삭제·restart 도구 실행 금지.
- policy18action,URDF20DOF,학습755/배포737,contact44제거와gait5활성을 유지. 원본orientation zero를contact padding으로 오해해 제거하지 않는다.
- 코드와 수치가 문서와 다르면 실제 코드/manifest/물리검증 우선. 추정 gain/effort를 공식 하드웨어 인증값이라고 주장하지 않는다.
- 최신 URDF/45k 당시 URDF를 바꿔치기하지 않는다. baseline export에는 exact trained_asset를 적용했고 출처를 기록했다.
- 처음에 git status/수정대상을 확인. 원본 baseline과 checkpoint 보호; 별도 실험폴더 사용. 기존 프로세스/다른 사용자 작업 종료 금지.
- 인증파일/API키/개인SSH키를 읽거나 복사/커밋하지 않는다. W&B는각자로그인; 업로드전 scripts/verify_snapshot.py와 LFS 점검.
- 다른 서버용 shell cache 경로 조정은 export-only 변경이다. native PPO/환경코드는 보존한다.
- 과거 스크립트의 서버 절대경로는 scripts/relocate_historical.py로 복사본에서 치환한다. reference/orin_source는 기구 원본참고이며 독립 실행모델이 아니다.
- GPU여유 확인 후 작은 검증부터. 평가에는optimizer업데이트 없음. 과거PD/scripted demo를 학습된 gait 성능이라고 해석하지 않는다.
- 명시적 요청 없이 하위 에이전트/자동 장기 모니터링/실험추가를 하지 않는다.
