# Grasp Pose 선택 기준 변경 검증

변경 범위: Windows 클라이언트의 결과 후처리, 2D/Viser 표시, 결과 저장/재생. `tracking_service/*.py`, `local_client/tracking_client.py`, `local_client/grasp_client.py`, `server/serve.py`는 SHA256 비교로 변경 없음을 확인했다. 모델 서버는 재시작하지 않았다.

공식 gripper base +Z approach / +X closing을 사용한다. 수식과 특이 자세 처리, 카메라 기준과 중력 기준의 차이는 [선택 기준 안내](local_client/GRASP_SELECTION.md)에 기록했다.

## 실제 저장 결과 재생 검증

새 모델 추론이나 Windows 카메라 촬영을 수행한 검증이 아니다. 기존 공식 RGB-D 데이터로 실행해 저장해 둔 실제 모델 결과를 재사용했다. 원본 결과 파일은 그대로 유지했다.

| 입력 | 후보 수 | 원래 C 기준 순위 → 새 S 기준 순위 | C | R | P | Y | S |
|---|---:|---|---:|---:|---:|---:|---:|
| official_rgbd_replay: 새 선택 | 100 | 8 → 1 | .982232 | .997997 | .944073 | .999799 | .981749 |
| official_rgbd_replay: 기존 C#1 | 100 | 1 → 21 | .994613 | .008926 | .959421 | .997243 | .892788 |
| SAM/AOT static RGB-D: 두 기준 동일 | 100 | 1 → 1 | .872790 | .983501 | .947820 | .942639 | .898349 |

첫 번째 재생은 S Top-10 + 기존 C#1의 11개 공식 CAD, 두 번째는 10개 공식 CAD가 생성됐다. Viser 로컬 HTTP 200, C/S 비교 라벨, NPZ confidence 원본 보존과 Score 저장 round-trip을 확인했다. 2D overlay 이미지를 직접 확인했다.

100회 반복한 재정렬 평균 약 0.36ms, P95 약 0.37ms(서버 CPU 측정). Windows 처리시간 측정값은 아니다.

- 검증 상세: `outputs/ranking/official_rgbd_ranked_verification.json`, `tracking_ranked_verification.json`
- 후보별 비교 CSV: `outputs/ranking/*_comparison.csv`
- 결과 JSON/NPZ: `outputs/ranking/grasps/`
- 비교 overlay: `outputs/ranking/rgb/official_rgbd_ranked_overlay.png`
- 테스트: `.venv/bin/python -m pytest -q tests` → **27 passed**.
- 배포: `dist/graspgenx-realsense-windows.zip` (업데이트된 클라이언트/안내/실제 비교 예제 포함).

재현:

```bash
.venv-client/bin/python scripts/verify_grasp_ranking.py
.venv-client/bin/python scripts/verify_grasp_ranking.py --source outputs --stem official_rgbd_replay --run-id official_rgbd_ranked
```

현재 변경의 Windows 실시간 카메라 실행은 사용자가 업데이트 후 G/V로 확인해야 한다. 기존 SSH 터널/서버 설정은 유지하며, 서버 재시작이나 추가 라이브러리 설치는 필요 없다.
