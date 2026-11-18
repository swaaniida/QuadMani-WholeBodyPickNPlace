# RealSense → Tracking SAM → GraspGenX

Intel RealSense D435i의 RGB-D를 서버로 전송하고, SAM 초기 분할 + AOTT 연속 추적, GraspGenX 후보 생성, ICP 보정·지면 충돌 필터·Top-1 선택·기록·3D 재생을 수행하는 PoC입니다. Windows는 CPU 클라이언트, A100은 모델 추론을 담당합니다. 로봇 제어와 policy 학습은 포함하지 않습니다.

## 먼저 읽을 문서

- [처음부터 설치·실행](docs/REPRODUCE.md)
- [현재 알고리즘·좌표계·제약](docs/CURRENT_STATE.md)
- [시행착오·측정 결과·배운 점](docs/LESSONS.md)
- [필수 파일과 제외 대상](docs/FILE_MAP.md)
- [고정 모델·checkpoint·dependency revisions](docs/DEPENDENCIES.json)
- [GitHub push 준비·검사](docs/GITHUB.md)
- [이번 정리의 변경·검증 결과](docs/VERIFICATION.md)
- [외부 코드·CAD 출처](THIRD_PARTY_NOTICES.md)

## 주요 진입점

| 기능 | 파일 |
|---|---|
| GraspGenX 서버 | `graspgenx-realsense/scripts/start_server.sh` |
| Tracking SAM 서버 | `graspgenx-realsense/scripts/start_tracking_server.sh` |
| 최신 ICP/ground/temporal 기록 | `graspgenx-realsense/local_client/main_temporal.py` |
| 기존 Tracking/ROI 실행 | `local_client/main_tracking.py`, `local_client/main.py` (위 project 아래) |
| 최신 기록 재생 | `graspgenx-realsense/local_client/view_temporal.py` |
| diffusion/두 OBB branch 비교 | `graspgenx-realsense/local_client/run_obb_replay.ps1` |
| 카메라·손목·UMI rig | `graspgenx-realsense/local_client/view_wrist_rig.py` |

실행 코드의 기존 디렉터리 구조를 유지했습니다. 환경·checkpoint·녹화·ZIP은 삭제하지 않고 Git에서만 제외했습니다. 원본 CAD/URDF와 표시용 mesh, 고정 revision, 설치 스크립트, 회귀 검사 및 선별된 검증 보고서를 보존합니다.

2026-11-17에 저장소 구성을 정리했습니다. 초기 연구 계획은 [원문](docs/history/ORIGINAL_RESEARCH_PLAN.md)에 보존했습니다. 하위 프로젝트의 기존 README/REPORT/TASK는 시점별 이력이므로 최신 동작은 위 CURRENT_STATE/REPRODUCE를 기준으로 확인하세요.
