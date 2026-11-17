# 바닥 EE 동작 — 시점 조작 3D 뷰어

## 현재 기본: vy 명령0 고정36초

[36초 vy=0 EE+보행 3D재생](http://localhost:8774/?scenario=motion36_vy0). **선속도 명령vy는 전구간0**이며 vx/yaw·EE일정은이전36초와같다. 실제vy는drift측정을위해표시한다. 기존motion36 주소도새vy0기록으로연결된다. 이전좌우명령시험은motion36_lateral,64초정지는ground64선택으로보존한다.

새 기록: `datasets/motion36_vy0/`. 원본·MP4·보고서: `reports/ee_locomotion_vy0_compare_20261114/`. 서버8774는그대로사용한다. 브라우저를새로고침한다.

## 이전36초: 좌우명령포함 기록 (보관)

이전 기록은 **[36초 좌우명령 시험](http://localhost:8774/?scenario=motion36_lateral)**으로 보관한다. 전후·좌우·양방향 회전·대각선·복합 명령, 명령/실제 속도 HUD와 파란 선속도 명령 화살표가 포함된다.

실험 선택에서 **64초 정지 + 바닥 EE 비교**로 전환하면 기존 기록을 그대로 볼 수 있다. `로봇 따라가기`는 이동 중 모델을 화면 중심에 유지하며, 비교 시 시점 연결은 각 로봇 중심의 상대 시점을 맞춘다. 36초 목표는 xy가 몸통/yaw를 따르고 z는 지면 기준이다. 64초는 world 고정 목표다.

새 데이터는 `datasets/motion36/`에 있다. 서버는 기존8774그대로 사용하며 정적 기록만 재생한다. 새 rollout은 이미 종료됐고 아래의 기존64초 설명과 구분한다. 36초 결과는 `reports/ee_locomotion_compare_20261114/REPORT_KO.md`를 참조한다.

## 접속

서버가 실행 중인 주소: `http://127.0.0.1:8774`

원격 VS Code의 **포트(Ports)** 탭에서 **8774**를 전달한 뒤, PC 브라우저에서 **http://localhost:8774**를 연다. 전달된 로컬 포트가 달라지면 VS Code가 표시하는 주소를 사용한다.

SSH 포워딩을 직접 사용할 때는 PC 터미널에서 실행한다.

```bash
ssh -N -L 8774:127.0.0.1:8774 vbc@astrago
```

서버 재실행:

```bash
bash /data/team/users/suhwan/vbc-as2piper/reports/ee_ground_compare_20261114/viewer/serve.sh
```

HTML을 file://로 직접 열면 브라우저가 바이너리 데이터 로딩을 막을 수 있으므로 HTTP 주소를 사용한다. `serve.sh`는 127.0.0.1에만 바인딩한다. 다른 포트가 필요하면 `PORT=8775 bash .../serve.sh`로 실행하고 해당 포트를 전달한다.

## 조작

- 왼쪽 드래그: 시점 회전. 휠: 확대/축소. 오른쪽 드래그: 화면 이동.
- **재생/일시정지**, **시간 슬라이더**, **0.25/0.5/1/2배속**.
- 체크포인트 선택: 10k/20k/30k/45k. 기본은 45k 단일 화면.
- **4개 동시 비교**: 모든 패널이 같은 시각을 재생한다. `비교 화면 시점 연결`을 켜면 한 패널의 회전·확대·이동을 다른 패널에도 적용한다.
- **기본/측면/정면/위쪽**, **로봇 중심** 버튼.
- 목표 경로·실제 EE 궤적, 목표·실제 방향 축 표시/숨김. 팔 투명도 조절.
- **Space** 재생/정지, **좌우 화살표** 시간 이동, **R** 기본 시점. 입력 상자에 포커스가 있으면 키보드 단축키가 적용되지 않을 수 있다.

빨강은 목표 EE, 청록은 실제 EE다. 패널별 위치·방향 오차, 선택 모델의 목표/실제 높이·좌표·실제 몸통 속도를 표시한다. 선속도·각속도 명령은 원래 기록 전체에서 0이다. 실패 reset도 표시한다.

## 데이터와 렌더링

- 기존 `../data_final/states.npz` 기록만 사용했다. **새 학습·물리 시뮬레이션 없음**.
- 서버는 정적 파일만 제공한다. 사용자가 여는 브라우저에서 WebGL 렌더링하며 서버의 NVIDIA GPU에서 실시간 MuJoCo/Isaac Gym을 실행하지 않는다.
- 64초, 25Hz 상태 기록 1600프레임, 4체크포인트. 브라우저 재생은 이 프레임을 시각화하며 오프라인 MP4의12.5fps보다 촘촘한 원본 기록을 사용한다.
- 원본 URDF의 시각 형상 **31개**와 body **31개**. AS2·PiPER·UMI·Orin 포함, 명목 질량 23.328064kg.
- MuJoCo는 기록된 root/q에서 FK를 계산하는 내보내기에만 사용했다. 동역학을 다시 적분하지 않는다. 비reset 프레임 TCP 최대 FK 오차 0.00000636m.
- reset 경계의 q/root와 EE 기록이 불연속인 2개 프레임은 내보내기에서 별도로 기록했다. reset 전후의 점프를 새로운 물리 움직임으로 해석하지 않는다.
- 좌표: 초기 AS2 base xy / 지면 z 기준, metre, z 위쪽. 바닥 쪽 궤적은 기존 학습 범위를 벗어난 평가다. 원본 실험 조건과 초기화 차이는 [기존 평가 보고서](../REPORT_KO.md) 참조.

## 구성

| 파일 | 역할 |
|---|---|
| index.html / viewer.js | UI와 Three.js 재생기 |
| manifest.json | 형상·checkpoint·기록 배열 구조·FK 검증 정보 |
| positions.bin / indices.bin | 공유하는 원본 시각 메시 |
| poses.bin | 기록 frame×checkpoint×body의 world position/quaternion |
| signals.bin | 목표·실제 EE pose, 오차, 실제속도, phase, 접촉 진단 |
| vendor/ | 로컬 Three.js·OrbitControls·한국어 글꼴. CDN 연결 불필요 |
| export_replay.py | NPZ와 기존 렌더용 모델에서 데이터 재생 파일 생성 |
| serve.sh | 로컬 HTTP 서버 |
| browser_checks.json | 실제 Chromium/WebGL 조작 검증 결과 |
| single_view.png / four_views.png | 브라우저 확인 화면 |

실행 파일(index.html, viewer.js, manifest.json, *.bin, vendor/)과 serve.sh를 포함한 이 폴더 전체를 복사하면 원본 SDK나 학습 checkpoint 없이 기록을 재생할 수 있다. export_replay.py를 다시 실행하려면 기존 data_final과 MuJoCo가 필요하다.
