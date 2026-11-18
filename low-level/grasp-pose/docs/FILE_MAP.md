# GitHub 보존 범위

실행 경로를 깨뜨리지 않도록 기존 `graspgenx-realsense/` 구조를 유지한다. 루트 `.gitignore`는 **allowlist**로 작성했다. 녹화와 환경을 지우지 않고 Git에서만 제외한다.

| 중요도 | 경로 | 역할 |
|---|---|---|
| 필수 | `graspgenx-realsense/server/` | 공식 GraspGenX 서버 어댑터와 시간/메모리 측정 |
| 필수 | `graspgenx-realsense/tracking_service/` | SAM 초기화 + AOTT 시간적 추적, 동기 RGB-D snapshot, 비동기 grasp |
| 필수 | `graspgenx-realsense/local_client/*.py` | RealSense, 통신, ICP, 지면 필터, 선택, 기록, 재생 |
| 필수 | `graspgenx-realsense/local_client/requirements*.txt` | Windows CPU 환경, Open3D/Viser 확장 |
| 필수 | `graspgenx-realsense/configs/` | sweep/ICP/지면/선택 설정, 과거 환경 snapshots |
| 필수 | `graspgenx-realsense/scripts/` | 고정 revision 설치, 공식 검증, benchmark, ZIP 생성 |
| 필수 | `graspgenx-realsense/tests/` | geometry, SE(3), queue/reset, 후보 순서, 프레임 연결 회귀 검사 |
| 필수 | `local_client/assets/franka_umi_*` | 공식 gripper 설정·CAD·sweep; prefix는 앞의 project 경로 생략 |
| 필수 | `local_client/assets/wrist_camera_rig/` | 수정된 위쪽 손목 체결 URDF, 실제 mesh + 표시용 LOD, 변환·출처 |
| 필수 | `graspgenx-realsense/assets/wrist_camera_mount/*.stl` | 사용자 제공 원본 거치대 CAD; mm 단위 |
| 중요 | `docs/LESSONS.md`, `docs/REPRODUCE.md`, `docs/CURRENT_STATE.md` | 경험, 복원 절차, 최신 동작의 기준 문서 |
| 중요 | `docs/evidence/` | 선별된 실제 benchmark 요약과 과거 보고서; 원본 수치 보존 |
| 이력 | `docs/history/`, 기존 `*REPORT.md`, `TASK.md` | 초기 설계와 단계별 기록. 최신 상태의 근거로 단독 사용 금지 |

## GitHub에 넣지 않는 것

- `.venv*`, `.runtime`, `.cache`, `.deps`, 다운로드한 `tools/uv`: OS/드라이버/경로 종속. 설치 스크립트로 복원.
- `upstream/`: 공식 저장소와 모델/checkpoint. revision과 다운로드 경로만 보존하며 중첩 Git 저장소를 add하지 않는다.
- `outputs*/`, `result/`, `results/`, `img_logs/`: 녹화 RGB/Depth/Mask, 후보, CSV, 로그, 임시 결과. **원본은 현재 디스크에 보존**한다.
- `dist/`, ZIP, 자동 생성 `example_results/`, `example_auto_session/`: 코드에서 재생성. 입력 녹화와 별개로 보관.
- 실제 장치 `realsense_calibration*.json`, `camera_tcp_transform*.json`, device-calibrated URDF: 장치별 실측/개인 환경으로 별도 백업.
- 제조사 전체 PDF: 라이선스 중복/불필요 용량을 피하고 `THIRD_PARTY_NOTICES.md`의 원문 링크 사용.

## 별도 백업이 필요한 데이터

Git은 녹화 보관소가 아니다. 기록마다 `comparison.csv`/`poses.csv`와 참조 NPZ·RGB-D·Mask·metadata를 **세션 폴더 전체로** 백업해야 한다. CSV만 있으면 재구성이 불가능하다. 이 정리 작업은 기존 녹화/환경을 삭제하거나 이동하지 않는다.

카메라 rig는 자체 포함되어 바로 재생 가능하다. 그러나 전체 AS2/PiPER URDF와 원본 UMI 제작 모델은 별도 `vbc-as2piper` 저장소 소유다. 이 저장소의 rig로 전체 로봇을 복원할 수는 없다. `build_wrist_camera_rig.py --robot-urdf ...` 재생성 시 해당 원본도 필요하다.
