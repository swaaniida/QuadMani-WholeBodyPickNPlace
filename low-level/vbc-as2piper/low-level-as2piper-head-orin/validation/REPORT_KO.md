> 이 문서는 Reset/Push 후속 수정 전의 초기 검증 기록이다. 현재 설정과 추가 시험 결과는 reset_push/REPORT_KO.md에 있다.

# Head + Orin 검증 결과

## 적용 설정

- EE 구면 범위 r=[0.40,0.48] m, 고도각=[0.10,0.50] rad, 방위각=[-0.25,0.25] rad, 방향 변화 각 축 ±0.08 rad.
- 다리 Kp hip/thigh/calf = [80.0, 80.0, 92.489] Nm/rad.
- 다리 Kd = [2.941, 2.776, 2.785] Nms/rad, armature=[0.026,0.026,0.038] kg m².
- 공칭 하중의 최대 관절 처짐 0.12 rad, 감쇠비 0.7, 대각 관성 기반 5 Hz를 설계 가정으로 사용했다. 이 계산은 모터 실측 식별을 포함하지 않는다.
- 추가 몸통 질량 0~2.333 kg. base_link CoM 변동 x ±0.02809 m, y/z ±0.02192 m.
- push 속도 상한 0.39639 m/s. 원본 정지 명령에서는 2.5배로 적용되어 축별 약 ±0.99097 m/s이다. 8초 간격으로 root 수평 속도를 덮어쓴다.
- 마찰계수 0.3~3.0, motor strength 0.7~1.3, 다리 reset 0.8~1.2배는 원본 범위를 사용했다. DR 시험은 원본 초기 속도 변동 ±0.1도 포함했다.

## 오프라인 IK 및 충돌

| 범위 | 자세 수 | IK 실패 | 비인접 링크 충돌 | 통과 |
|---|---:|---:|---:|---:|
| proposed | 1728 | 72 | 0 | 1656 |
| medium | 1728 | 11 | 0 | 1717 |
| balanced | 1728 | 9 | 0 | 1719 |
| inner | 1728 | 0 | 0 | 1728 |

- 각 범위는 위치 4×4×4 격자와 방향 변화 3×3×3 격자를 조합했다. IK 허용오차는 위치 5 mm, 방향 3°다. 실패는 이 수치해법에서 수렴하지 않은 경우이며 기구학적 불가능을 증명하지 않는다.
- 선택한 inner 범위의 128개 구면 보간 경로를 경로당 11점으로 검사했다. IK 실패 0개, 충돌 경로 0개다.
- 충돌은 URDF collision 형상으로 만든 MuJoCo convex mesh 모델에서 1 mm 초과 관통을 판정했다. 직접 연결된 부모·자식 링크의 조립 중첩은 제외했다. 연속 공간 전체의 무충돌 보장은 아니다.

## Isaac Gym 물리 rollout

조건별 16개 환경, 각 12초, seed 20261111. 다리 action=0으로 기본 PD 자세를 유지하고 팔은 원본 VBC DLS IK/position drive로 움직였다. PPO나 학습된 보행 정책은 사용하지 않았다. 통계는 2초 이후이며 reset 횟수는 전체 구간이다.

| 조건 | reset 수 | EE 위치 평균 / P95 (cm) | EE 방향 P95 (°) | roll/pitch P95 (°) | 팔 접촉 최고 (N) |
|---|---:|---:|---:|---:|---:|
| hold | 0 | 1.36 / 1.37 | 3.10 | 0.84 | 0.00 |
| ee | 0 | 1.69 / 2.95 | 5.82 | 1.05 | 0.00 |
| dr | 3 | 1.72 / 2.99 | 6.30 | 2.18 | 0.00 |
| dr_rough | 3 | 1.83 / 3.23 | 6.85 | 7.92 | 0.00 |

- hold는 초기 EE 고정, ee는 랜덤 EE 목표다. dr은 평지에서 모든 domain randomization과 push를 적용했다. dr_rough는 동일 조건에 원본 rough 생성기의 4×4 패치를 사용한 소규모 시험이다.
- 정지·랜덤 EE 목표는 각각 16/16개 환경에서 12초 동안 reset 없이 유지됐다. 평지 DR 및 rough DR은 각각 3개 환경이 첫 push 이후 reset됐다. 안정적인 외란 회복 통과로 판단하지 않는다.
- 관측은 [16,744], 실제 DOF 20, action 18이며 모든 관측이 유한했다.
- PhysX DOF force sensor로 position drive의 팔 토크를 측정했다. 팔 effort 포화율은 모든 조건에서 0이다. 축별 측정 최고값은 각 rollout JSON의 arm_measured_effort_peak_abs_Nm에 있다. URDF 팔 한계 100 Nm는 하드웨어 정격 검증값이 아니다.
- 기본 PD·팔 IK의 실행 정합성과 제한된 정지 안정성을 확인했다. 학습된 locomotion, 장시간 수렴, 실물 보행 및 지면 pick 동작은 검증하지 않았다.

## 재현

오프라인은 저장소 루트에서 `unitree_rl_mjlab/.venv/bin/python low-level-as2piper-head-orin/validate_workspace.py`로 실행한다. collision_model.urdf는 head URDF의 collision만 사용하며 메시 경로를 절대경로로 바꾼 입력이다.

물리 rollout은 low-level-as2piper-head-orin, 그 안의 .deps와 rsl_rl_local, 공유 third_party/isaacgym/python을 PYTHONPATH에 넣고 Python 3.8에서 `validate_rollout.py --case hold`, `--case ee`, `--case dr`, `--case dr --terrain trimesh`로 실행한다. W&B와 학습 optimizer는 실행하지 않는다.

실행 중 발견한 누락된 stop_update_goal 기본값, plane 생성 분기, 평지 env origin, measured_heights 미존재 처리를 새 복제본에서 수정했다. 원본과의 diff는 ../changes.patch에 기록했다.
