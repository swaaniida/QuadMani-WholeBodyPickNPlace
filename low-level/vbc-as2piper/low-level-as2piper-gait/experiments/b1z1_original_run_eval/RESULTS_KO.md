# B1Z1 원본 VBC 체크포인트 평가

## 평가 대상

- run: `b1z1-low/original-run01`
- checkpoint: `model_45000.pt`
- 원본 경로: `/data/team/users/suhwan/vbc-legacy/low-level/logs/b1z1-low/original-run01/model_45000.pt`
- 실행 설정은 로컬 W&B 기록에서 확인했다. 학습은 `--observe_gait_commands`, seed 1로 실행됐다. 이를 빼면 관측 이력이 5개 채널 짧아지고 체크포인트가 로드되지 않는다.
- 평가 코드는 원본 `vbc-legacy`에서 worktree 안으로 복사했다. 원본 코드·체크포인트는 읽기만 했다.

## 평가 방법

- Isaac Gym PhysX에서 원본 B1Z1 `ManipLoco` 태스크로 rollout했다. 체크포인트의 deterministic policy를 사용했다.
- 평지, 환경 1개, seed 1. 원본 `play.py` 평가 관례에 따라 base mass randomization을 켜고 push와 base CoM randomization은 껐다. episode 종료 시간만 충분히 늘려 연속 영상을 얻었다.
- locomotion 영상은 서기 4초, 전진 5초, 후진 5초, 제자리 회전 5초, 전진+회전 5초로 구성했다. EE 목표는 원본 VBC sampler에서 계속 재생성된다.
- EE 영상은 정지 명령에서 원본 VBC EE pose sampler만 12초 실행했다.
- A100 MIG에서 Isaac Gym 카메라 센서 초기화가 실패해, PhysX state를 저장한 뒤 기존 B1Z1 MuJoCo 장면에서 렌더링했다. 영상은 저장된 PhysX 상태 재생이며 MuJoCo 재시뮬레이션은 아니다.

## 영상

- [평지 보행 명령 시퀀스 24초](videos/b1z1_original_run01_45000_locomotion_flat.mp4)
- [원본 rough 지형 보행 명령 시퀀스 24초](videos/b1z1_original_run01_45000_locomotion_rough.mp4)
- [정지 상태 EE 목표 추적 12초](videos/b1z1_original_run01_45000_ee_tracking_flat.mp4)
- [장면 검토 contact sheet](videos/review_contact_sheet.jpg)

빨간 점은 명령 EE 목표, 초록 점은 실제 EE 위치다. 화면에 command/measured velocity와 EE position error를 표시했다.

## 측정 결과

| 구간 | 명령 vx / yaw | 측정 vx / yaw | EE 오차 평균 / P95 | upright 비율 |
|---|---:|---:|---:|---:|
| 서기 | 0 / 0 | 0.082 / 0.086 | 0.061 / 0.102 m | 1.00 |
| 전진 | 0.50 / 0 | 0.477 / -0.028 | 0.123 / 0.234 m | 1.00 |
| 후진 | -0.30 / 0 | -0.404 / 0.012 | 0.125 / 0.206 m | 0.93 |
| 회전 | 0 / 0.60 | 0.020 / 0.585 | 0.091 / 0.121 m | 0.92 |
| 전진+회전 | 0.35 / 0.50 | 0.353 / 0.398 | 0.332 / 0.536 m | 0.85 |
| rough 전진 | 0.50 / 0 | 0.404 / -0.025 | 0.126 / 0.231 m | 1.00 |
| rough 후진 | -0.30 / 0 | -0.372 / 0.000 | 0.096 / 0.150 m | 0.91 |
| rough 회전 | 0 / 0.60 | 0.023 / 0.490 | 0.091 / 0.131 m | 0.91 |
| rough 전진+회전 | 0.35 / 0.50 | 0.298 / 0.367 | 0.313 / 0.581 m | 0.79 |
| 정지+EE 목표 재생성 | 0 / 0 | 0.019 / 0.075 | 0.123 / 0.342 m | 0.81 |

속도 단위는 m/s, yaw 단위는 rad/s다. upright는 projected gravity의 수평 성분이 0.5 미만인 프레임 비율(기울기 약 30도 미만)이다. episode reset은 없었지만 이는 해당 원본 설정에 fall 종료 조건이 없기 때문이므로, reset 0을 안정성 통과로 해석하면 안 된다.

## 판정

- 평지에서 전진 및 제자리 회전의 평균 속도 추적은 대체로 정상이다. 후진은 명령보다 빠르고 측면 속도 편차가 컸다.
- 전진+회전에서는 EE 평균 오차가 0.33m, P95가 0.54m로 커졌고 upright 비율도 0.85로 낮아졌다.
- 정지 상태의 넓은 EE 목표 재생성에서도 평균 오차가 12cm, P95가 34cm였다. 영상상 팔이 여러 방향으로 움직이지만 목표 추적과 자세 안정성이 모두 충분하다고 판정하기 어렵다.
- 따라서 이 체크포인트는 기본 전진·회전 gait의 동작은 보이나, 전신 조작과 EE pose tracking이 제대로 안정화됐다고 보기는 어렵다. rough 지형에서도 기본 전진·후진·회전 명령을 따르지만 복합 전진+회전 시 upright 비율이 0.79로 낮아지고 EE P95 오차가 0.58m까지 증가했다. 평지/rough 모두 seed 1, 환경 1개라 다른 seed·대규모 환경에서의 일반화는 별도 평가가 필요하다.

재실행 상태와 상세 시계열은 같은 폴더의 `videos/*_states.npz`, `videos/*_metrics.json`에 있다.
