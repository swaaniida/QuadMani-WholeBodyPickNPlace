# 시행착오와 얻은 결과

## 유지할 설계

- **환경 분리:** GraspGenX torch2.5.1+cu121, Tracking torch1.13.1+cu117. 시스템 CUDA/driver를 덮어쓰지 않는다. Windows CPU 전용. 예전 env 폴더를 복사하는 대신 pinned source+설치 스크립트 사용.
- **실제 추적:** SAM은 첫 클릭의 segmentation만 수행하고 AOTT가 다음 프레임에 memory propagation한다. SAM을 매 프레임 재실행하는 구조가 아니다.
- **동일 시점:** mask/depth/pose를 촬영 Frame ID로 연결한다. 최신 RGB 위에 과거 결과를 무표시로 붙이지 않는다. 서로 다른 PC wall clock을 빼서 latency를 구하지 않는다.
- **bounded worker:** tracking/ICP/grasp는 병렬 처리하고 latest-frame 정책을 쓴다. busy grasp에 요청을 계속 넣으면 `One grasp job is already running`이 발생한다. 오래된 응답은 reset generation으로 무효화한다.
- **모든 후보 먼저:** confidence top100을 먼저 자르면 바닥을 피하는 낮은 confidence 후보를 잃는다. 전체 반환 후보 → 지면 필터 → scoring 순서가 필요하다.
- **지면 충돌:** OPEN mesh 전체를 검사해야 palm/몸통 관통도 잡힌다. Mesh를 pose에 직접 곱하는 공식 base-frame convention을 유지한다. Sweep volume은 파지 conditioning 영역이며 실제 충돌 solid mesh와 다르다.
- **카메라 보정:** 객체가 가까워졌다는 이유로 grasp trajectory가 물체에 다가가는 로봇 경로라고 해석하지 않는다. 정적 배경 ICP로 reference frame을 고정하고 camera trajectory와 grasp 변화량을 따로 표시한다.
- **불신 정합:** fitness만 높아도 평면 배경은 회전/이동을 충분히 구속하지 못한다. RMSE·step·normal observability·keyframes와 함께 본다. identity를 가짜 성공 transform으로 쓰지 않는다.
- **temporal scale:** 위치+회전을 곱한 Gaussian은 ICP 오차/장면 변화로 e-19 이하가 되기 쉬웠다. 현재는 회전만 sigma20°; 180°는 다른 자세. timestamp gap만으로 이전 pose를 버리는 조건은 제거했다.
- **clearance:** 초기 `3mm + RMSE`에서 현재 `RMSE`만으로 변경했다. 바닥의 낮은 물체는 잡음·가려진 형상·open gripper geometry 때문에 후보 대다수가 reject일 수 있다. 필터 통과가 실제 접촉/파지 성공을 의미하지 않는다.
- **OBB 세계축:** 공식 world-Z heuristic을 optical camera 좌표에서 그대로 쓰면 지면 top-down이 아니다. camera OBB(+camera Z)와 world OBB(-plane normal)는 canonical frame으로 변환해 분리한다. 동일 point cloud·동일 discriminator로 모든 branch를 평가해야 비교 가능하다.
- **UI 실패 분리:** near=0은 잘못된 depth 설정이다(`0<near<far`). 닫힌 OpenCV window에서 trackbar를 읽으면 NULL window 오류가 난다. USB 미검출, GUI 이벤트 정지, model busy, 녹화 duration 종료는 다른 원인이다. 서버가 살아 있다는 사실만으로 client 정상이라고 판단하지 않는다.
- **Windows 경로:** `C:\Users\USER`에서 `Expand-Archive -DestinationPath ..`를 쓰면 `C:\Users`에 풀려 권한 오류가 난다. 항상 프로젝트 폴더로 이동하고 `.venv\Scripts\python.exe`를 명시한다. 전역 Python은 cv2가 없을 수 있다.
- **네트워크:** 현재 환경은 Tailscale SSH22 대신 일반 SSH2222/key를 사용했다. 로컬5556 예약/차단은15556/15557로 우회한다. 모델 서버는 localhost bind, SSH tunnel만 공개한다.
- **재생:** threadpoolctl 누락은 reconstruction 실패 원인이었다. Open3D+threadpoolctl을 별도 requirements에 포함했다. sparse frame ID는 연속 숫자가 아닐 수 있어 ID+1을 trajectory 연결 조건으로 쓰면 안 된다.
- **CAD:** 손목의 초기 clocking이 아래쪽이었다. camera–mount fitting은 유지하고 flange Z를 기준으로 둘을 함께180° 돌려 위쪽 두 볼트축에 맞췄다. SDK intrinsics와 CAD 외형 extrinsics를 혼동하지 않는다.

## 실제 측정에서 보존할 수치

아래는 2026-11-12, A100-SXM4-80GB **MIG3g.40gb/40192MiB**, 640×480 기록 입력의 과거 측정이다. 현재 서버 상태나 Windows 성능을 뜻하지 않는다. 세부 조건과 원본 요약은 `evidence/tracking_model_profile_20261112.{md,json}`.

| 측정 | 평균 | 해석 |
|---|---:|---|
| SAM GPU model | 1442.52ms | 초기 클릭 때; warmup10+100회, 반복 실제 이미지 |
| AOTT GPU model | 16.70ms | model-only 약59.86FPS; genuine DAVIS sequence |
| AOTT pre/post/transfers 포함 | 28.96ms | 약34.54FPS; network/GUI 제외 |
| localhost RGB-D replay cap30 | 19.26FPS | Windows USB/SSH가 아님 |
| localhost replay + GraspGenX | 17.90FPS | 병렬 실행; grasp7회 평균643.47ms |
| Tracking replica loaded allocated | 3194.37MiB | SAM+AOTT+DINO 함께 resident |
| SAM peak allocated | 6446.63MiB | reserved6972MiB, nvidia-smi 전체 점유와 구분 |

SAM641.091M + AOTT5.734M parameters. GroundingDINO172.840M도 resident. FLOPs는 profiler supported operators 하한이며 전체 연산량이라고 주장하지 않는다.

동일한 실제 녹화5프레임 branch 분석에서 diffusion safe1/1000, camera OBB104/250, world OBB50/250였다. 평균 penetration93.74/12.40/25.17mm. **이 녹화에 한정된 결과**이며 OBB의 일반적 우월성이나 실물 성공률이 아니다. 각 branch confidence 최고 후보는5/5프레임에서 관통했다. 점수와 물리적 feasibility는 독립 검사가 필요하다는 근거다.

## 다음 시작 때 우선 확인

1. 공식 sample → 서버 health → 저장 RGB-D replay → 실제 Windows 카메라 순으로 분리 검증.
2. 실제 camera K/scale/serial과 aligned color 좌표 확인.
3. ground plane과 mesh frame 검증 후 collision reject 수를 해석.
4. 낮은 물체는 후보별 clearance·confidence·원본 geometry를 비교. 필터를 풀어 성공처럼 보이게 만들지 않는다.
5. EE 제어 전에 실제 PiPER UMI TCP/hand-eye/질량·관성 실측이 별도 필요하다.
