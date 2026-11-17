# B1+Z1 45k — 정지 명령 EE 추종 20초

- 영상: `b1z1_stand_ee_20s.mp4` (20초, 25fps, 1280×670). 동일한 실제 PhysX rollout을 사선/후면 두 고정 시점으로 재생.
- 체크포인트: `/data/team/users/suhwan/vbc-legacy/low-level/logs/b1z1-low/original-run01/model_45000.pt`. 원본799차원 관측, deterministic history-encoder policy, 원본 IK/관측/접촉 보상/제어를 그대로 사용. strict checkpoint load.
- GPU 권한상 사용량 확인이 불가하여 CPU PhysX+CPU inference 실행. MuJoCo는 저장 상태 렌더링만 사용하며 동역학을 다시 계산하지 않음.
- 원본 flat trimesh(height0), DR·push off. 선속도 vx/vy 및 yaw 각속도 명령 모두0 유지. 물리적 몸통 이동/기울기는 허용.
- seed20261115, 초기조건4개, 2초 준비 후20초 측정. 연속영상을 위해 평가 episode timeout만60초로 연장. 기존 학습 코드·체크포인트는 수정하지 않음.
- 목표6구간: 정면→좌측→우측→상단→바닥 근처→먼 하단. 각 구간 약3.33초, 구면 보간2.3초+hold. 위치 및 orientation delta는 원본 B1 학습 범위 안이며 모든 endpoint/구간의 원본 단순충돌/지면 검사 통과. 원본 pitch=0.38−고도각, tool roll=π/2와 각 구간delta 적용. 지면근처 목표 z≈0.0493m.

## 발 들림 결과

접촉은 net contact force의 양의 Fz>2N 기준. 지면 간격은 sphere collision 반경40mm를 뺀 값이며, visual 반경30mm와 구분. 힘이 없으면서 지면 간격20mm를 초과하는 연속 구간은4개 초기조건 모두0초.

| 반복 | FL 접촉률 | FR 접촉률 | RL 접촉률 | RR 접촉률 | 최대 일시 간격 | 평균EE오차 | reset |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100.0% | 100.0% | 100.0% | 100.0% | 0.03mm | 5.95cm | 0 |
| 1 | 100.0% | 100.0% | 92.0% | 98.3% | 12.19mm | 6.28cm | 0 |
| 2 | 100.0% | 100.0% | 100.0% | 100.0% | 0.03mm | 6.35cm | 0 |
| 3 | 100.0% | 100.0% | 96.4% | 98.7% | 9.16mm | 6.29cm | 0 |

- 가장 큰 일시 들림은 반복1의 RL: 최대12.19mm, Fz≤2N 연속시간 최대0.06초. 반복1 RR은 최대7.64mm, Fz≤2N 연속시간 최대0.02초. 0.5초 rolling minimum이5mm를 넘는 지속 들림은 모든 발/반복에서0%.
- 영상은 최대 일시 들림이 관찰된 반복1을 선택. 반복0/2는 네 발 접촉률100%. 평균EE오차는4개 반복에서5.95~6.35cm. 선택한 반복의 실제EE 최소높이는 약4.53cm.
- 따라서 이번 평지·무외란·20초 조건에서 AS2처럼 한 발을 계속 든 정지 정책은 관찰되지 않음. 모든 EE 목표/외란/rough 조건에 대한 보장은 아님. 순간적인 접촉력 부족과 지속적 발 들림을 구분.
- visual 발이 정상 접촉 중 약1cm 떠 보이는 것은30mm visual vs40mm collision 반경 차이일 수 있음. 영상 HUD는 실제 net force와 collision 기준 간격을 함께 표시.

## 산출물/재실행

- `states.npz`: 50Hz×1000프레임×4환경의 root/joints/EE/foot positions·forces·clearance/실제속도/reset. `metrics.json`: 원본해시·발별접촉·EE·조건. `phase_metrics.json`: 구간별EE높이/오차.
- `video_verification.json`: 영상프레임/재생시간 확인. `contact_sheet.jpg`, frame_*.png: 장면 검토.
- `run.sh`: 기존 Python3.8/Isaac Gym 경로를 설정해원본소스 읽기전용 평가. `render.py`: OSMesa MuJoCo 재생. `finalize.py`: 첫 초기화프레임의 구간label정정/영상검증. 첫프레임은20ms 부동소수점 경계상 초기목표이며 이후6구간 진행.
- 원본 checkpoint·URDF·ManipLoco·B1 config SHA256 전후 일치. 실제 학습·optimizer update·W&B 실행 없음. AS2 학습 코드는 변경하지 않음.
