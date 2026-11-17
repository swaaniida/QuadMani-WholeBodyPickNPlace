# 바닥 중심 EE Pose Trajectory — 4 checkpoint 비교

[64초 비교 영상](ee_ground_compare_64s.mp4) · [대표 프레임](comparison_preview.jpg) · [지표 CSV](metrics.csv) · [전체 요약 JSON](summary.json)

## 조건

- 10k/20k/30k/45k 각각8개 명목 replica, 총32개 PhysX 환경. 영상은각 checkpoint의첫 replica(0/8/16/24)를사전에선택했으며성공사례선별없음.
- 선속도 vx/vy 및 yaw 각속도 명령은모든step에서0. 단, 몸통은물리적으로고정하지않아정책이만드는이동·기울기·실패가나타난다.
- 마찰·질량·CoM 등물리extrinsics는명목값으로고정, DR·push OFF. 물리값자체를없애는것이아니다. 정책은history encoder 추론을사용하며privileged latent를직접입력하지않는다.
- 64초중12~60초의48초를바닥근처(목표높이1~16cm)에집중. 목표범위 x0.45~0.72m, y±0.18m, z0.01~0.80m. 좌표는초기AS2 base xy와지면z를기준으로고정해몸통이움직여도목표가따라움직이지않는다.
- Orientation은world rpy 목표를최소jerk 보간: roll±0.12rad, pitch1.70~3.05rad, yaw±0.35rad. 영상의방향오차는quaternion geodesic angle.
- 물체없음·그리퍼열림유지. 접근/접촉목표/들기·내리기동작이며실제파지성공시험은아니다.
- 평상시10초timeout만평가용90초로연장해연속64초영상을기록. Roll/pitch/height 실패termination은유지하고실패reset을숨기지않는다.
- reset jitter를0으로하고동일한초기관절명령을주었지만첫물리step실제상태에는q최대0.03639rad, root기록최대0.01739 차이가있다. 영상초기3초는안정화구간이며통계는초기2초제외. 단일seed 명목replica 비교로서엄밀한동일상태·다중seed강건성시험은아니다.

## 결과

바닥구간은12~60초, 각checkpoint의8개환경전체평균이다. Reset의영상열은선택한1개환경의건수로, 8개합계와구분한다.

|Checkpoint|바닥 위치오차 평균(cm)|바닥 방향오차 평균(°)|Reset: 영상/8개합계|최저TCP높이(cm,8개전체)|
|---|---:|---:|---:|---:|
|10000|53.29|32.05|5/25|-0.137|
|20000|1.96|6.87|0/0|0.050|
|30000|2.46|8.72|0/0|0.040|
|45000|1.98|5.71|0/0|0.043|

- 20k/30k/45k는본경로에서실패reset없이바닥근처움직임을수행했다.20k와45k의바닥위치오차는약2cm로비슷하며방향오차가남아있다.
- 10k는몸통이동과오차·실패가크다. 낮은TCP높이만으로바닥작업성공이라고판정하지않는다.
- 손가락mesh의지면거리±3mm와손가락net contact force>0.5N이함께기록된프레임은영상replica에서20k95개,30k269개,45k91개(원본25Hz기록기준). 기하학과힘로그는바닥접촉을뒷받침하지만contact pair를직접저장하지않았으므로net force만으로self-contact와완전히분리하지는않는다.
- 기존학습EE명령은r[0.40,0.48]m,pitch[0.10,0.50]rad이다. 정상서기명목에서목표z약0.50~0.69m로바닥영역이빠져있다. 본영상은학습범위밖의경로를직접준시험이며기존checkpoint와학습설정을변경하지않았다.
- 바닥도달은기구적도달불가와다르다. 기존DLS IK/position drive와학습된leg 정책을포함한시스템이경로를수행하는결과이며arm actor 단독학습성능이아니다.

## 영상 표시·검증

- 왼쪽위10k, 오른쪽위20k, 왼쪽아래30k, 오른쪽아래45k. 빨강=목표TCP, 청록=실제TCP. 위치오차·방향오차·목표/실제높이·실제몸통속도·phase·시각표시.
- 물리기록은GPU PhysX 50Hz, 상태는25Hz저장. 영상은12.5fps(원속도64초),H.2641280×1000,800frames. 프레임수와재생시간검증완료.
- MuJoCo는정확한URDF의관절/root상태를CPU로재생하는렌더러로만사용. MuJoCo동역학을별도로진행하거나스크립트로팔·몸통포즈를성공상태에맞춘것이아니다.
- 비reset프레임TCP FK오차최대약6.4µm.10k의reset경계2프레임은reset후q/root와reset전EE캐시의불연속으로별도기록. 화면에실패RESET표시.
- 시점방향은공통고정.10k가크게이동해프레임밖으로나가려는경우에만목표와로봇이보이도록시야확대표시. 다른세checkpoint는확대없음.
- 유효최종데이터는`data_final/`이다. `data/`, `data_matched/`, `data_nominal/`은초기화·저장·중단점검용으로최종비교에서제외했다.
- `data_final/states.npz`: 실제root/q/TCP목표·실제pose·오차·몸통속도·손가락힘·reset. `metadata.json`: 목표waypoints,checkpoint/source hash,override와전체지표. `ground_contact_geometry.json`: 바닥기하검사.

## 구간

|시간(s)|동작|
|---|---|
|0~3|초기정지안정화|
|3~9|상단·전방확장|
|9~15|바닥접근·접촉목표|
|15~27|바닥좌우·먼곳·가까운곳|
|27~42|살짝들기·내리기·대각선이동|
|42~57|바닥중앙·좌우쓸기·접촉목표유지|
|57~64|바닥에서들기·초기자세복귀|

## 관련 코드

- `low-level-as2piper-head-orin-nocontact/force_ablation/ground_trajectory_compare.py`: 평가전용trajectory·capture.
- `low-level-as2piper-head-orin-nocontact/force_ablation/render_ground_compare.py`: CPU렌더링.
- 학습코드/PPO/reward/URDF/기존모델은수정하지않았다. 평가script2개와산출물만추가했다.

## 시점 조작 3D 뷰어

[3D 뷰어 접속·조작 설명](viewer/README_KO.md) · 포트8774 전달 후 http://localhost:8774 . 단일/4분할·회전/확대·시간 이동·속도 조절 가능. 새 물리 시뮬레이션 없이 기존 기록만 재생한다.
