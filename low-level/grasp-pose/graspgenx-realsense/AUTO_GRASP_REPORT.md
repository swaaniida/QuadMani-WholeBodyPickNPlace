# 자동 Top-1 Grasp 기록 및 프레임별 데이터 검증

구현: 기존 Windows Tracking SAM 클라이언트에 단일 요청 자동 스케줄러를 추가했다. 각 요청이 완료되면 최신 유효 프레임으로 다음 요청을 보낸다. 추론 후 기존 카메라 기준 가중 점수 S로 재정렬하고 Top-1만 저장한다. 추론 중 실시간 3D 갱신을 생략하고 종료 후 세션 단위로 재생한다.

사용법 및 데이터 스키마: [local_client/AUTO_GRASP.md](local_client/AUTO_GRASP.md).

## 실제 모델 60초 검증

공식 `real_world/00`의 실제 RGB-D 한 프레임을 반복 입력했다. SAM 클릭 초기화→AOT 연속 추적→공식 GraspGenX scene-depth 추론→Top-1 재정렬→모든 추적 프레임의 RGB-D/Mask 아카이브→오프라인 재생을 실제 실행했다. 가짜 pose나 모델 대체는 사용하지 않았다. **정지 데이터 재생이며 Windows USB 카메라 또는 움직이는 실제 RGB-D 시퀀스 검증은 아니다.** 통신은 서버 내부 localhost로, SSH/Windows 렌더링 성능이 포함되지 않는다. 저장 경로는 기존 grasp-pose 내 NFS다.

| 지표 | 측정 |
|---|---:|
| Top-1 기록 | 67개 / 60.1584초 |
| 실제 완료 pose FPS | **1.1137** |
| 연속 완료 간격 기반 FPS | 1.1127 |
| GraspGenX 평균 추론 | 633.40ms |
| GraspGenX P50 / P95 | 619.97 / 754.73ms |
| 모델 시간 역수(참고값) | 1.5788 FPS |
| 클라이언트 프레임 준비→Top-1 저장 평균 | 976.53ms |
| 위 E2E P50 / P95 | 886.17 / 1517.26ms |
| 결과 완료 간격 P50 / P95 | 825.89 / 1288.78ms |
| RGB-D/Mask + Top-1 저장 단계 평균 | 54.57ms |
| 추적 프레임 아카이브 | 458개 / 누락 0 / 쓰기 오류 0 |
| 자동 요청 실패 | 0 |

실제 모델은 요청 num_grasps=200에 대해 GraspMoE 추가 분기를 포함한 후보 250개를 반환했다. 기존 threshold 및 top-K 처리 후 100개에서 최종 S#1 한 개를 저장했다. 요청 수를 실제 생성 후보 수로 대신 기록하지 않는다.

모든 pose에서 Frame ID와 저장된 RGB/Depth/Mask ID가 일치하고, 저장 후 Depth 배열이 원본과 정확히 같은지 확인했다. NPZ에는 `(4,4)` Top-1 pose 하나만 있으며 전체 후보 배열은 저장하지 않았다. 오프라인 Viser에서 첫/마지막 sample, RGB/Depth/Mask, 공식 CAD 및 전체 위치 이력을 생성했고 HTTP 200을 확인했다. Windows 전용 실제 카메라/브라우저 조작은 사용자가 실행해야 한다.

29개 테스트 통과. 추가로 아카이브 원자적 저장 변경 후 관련 2개 테스트를 재검증했다. 테스트 fixture는 기하학/스케줄링 검증용으로만 사용하며 위 벤치마크 수치에는 포함하지 않았다.

## 보존한 지연 사례

이전 30초 시험에서 한 추론에 **21.148초**가 걸렸다. 해당 실행의 전체 결과는 17개/36.455초 = **0.4663 FPS**이며 제외하거나 성공적인 60초 결과에 섞어 숨기지 않았다. 원인은 확정하지 않았다. 로그는 `outputs/auto_grasp/benchmark_final.log`, 요약은 `benchmark_with_outlier.json`에 보존했다. 최종 FPS는 입력과 서버 부하에 따라 변하므로 Windows에서 직접 기록한 summary를 기준으로 판단해야 한다.

초기 RGB/Depth/Mask를 여러 작은 파일로 쓰던 방식은 NFS 쓰기 대기 때문에 Top-1 저장 단계 평균 약 996ms가 발생했다. Frame ID별 단일 NPZ, 빠른 무손실 압축, pose 컨텍스트 우선 저장으로 최종 평균 약 55ms가 됐다. 다른 추적 프레임이 쌓이면 큐 4개만 유지하고 누락 상태를 명시한다. 저장된 Top-1의 RGB-D/Mask는 항상 우선 보존한다.

## 코드 범위

- `local_client/auto_grasp.py`: 자동 요청 정책, Top-1/CSV/latency/안정성 기록.
- `local_client/frame_archive.py`: Frame ID별 비동기 RGB-D/Mask 아카이브.
- `local_client/view_session.py`: 종료 후 slider/기록 간격 재생.
- `local_client/main_tracking.py`: A/Q 및 자동 실행 옵션; 기존 G/V 모드 유지.
- `local_client/tracking_client.py`: 추적 응답을 기록기에 전달하는 선택적 비차단 hook만 추가. 모델 호출/통신/요청 정책은 기존대로다.
- `tracking_service/serve.py`: 실제 후보 배열에서 필터 전/후 개수를 응답 통계에 추가. 기존 sampling/threshold/top-K/모델 호출은 그대로다. 이 통계를 적용하기 위해 유휴 Tracking 서비스만 재시작했다. GraspGenX 서버는 재시작하지 않았다.
- Tracking 모델 backend, 공식 GraspGenX sampler, GraspGenX 서버 및 RPC client는 SHA256으로 변경 없음을 확인했다.

## 결과/재현

- `outputs/auto_grasp/latest_benchmark.json`: 최종 측정 상세 및 실제 세션 경로.
- `outputs/auto_grasp/benchmark_60s.log`: 실제 67개 추론 결과 로그.
- 세션 내 `poses.csv`, `tracking_frames.csv`, `frames_with_poses.csv`, `summary.json`, `frames/`, `samples/`.
- `dist/graspgenx-realsense-windows.zip`: 갱신된 Windows 코드/설명/실제 결과 발췌.

```bash
.venv-client/bin/python scripts/benchmark_auto_grasp.py --duration 60
.venv/bin/python -m pytest -q tests
```

카메라를 움직이면 위치/회전 변화는 카메라 움직임도 포함한다. 좌표는 항상 해당 프레임의 camera optical 기준이며, 중력 보정·로봇 TCP·카메라 이동 추정은 추가하지 않았다.
