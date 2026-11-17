# Feet Height 0.18 및 정지 Contact Shaping 마스크

## 변경

low-level-as2piper-head-orin-nocontact의 AS2 config에 feet_height_target=0.18을 명시했다. Base-height reward 기준 비율0.3244069/0.55=0.58983, 다리 길이0.42544/0.70=0.60777. B1 0.30에 곱하면0.17695~0.18233m이므로0.18m을 선택했다. Reset 초기 높이 비율은0.32641/0.50≈0.653으로 reward 기준 비율과 구분하며 reset 높이는 변경하지 않았다.

feet_height scale+1.0, height reward 수식(앞발 두 지면 상대 중심 높이의norm에 대한hinge), base_height_target, contact force threshold40N, shaping scale−2, sigma,2Hz gait 및 PPO 모두 유지했다. 0.18은 발 하나를18cm 들라는 직접 명령이 아니다.

Shaping force/vel 각각에 기존 _get_walking_cmd_mask()를 곱했다. Gait phase 정지 판정과 같은 기준을 사용하므로 stop에서 가상의 swing phase와 접촉력/발 속도 때문에 penalty가 생기는 문제를 제거한다. |vx|>0.2 또는 |vy|>0.2 또는 |yaw_rate|>0.5일 때 이동으로 판단하는 기존 deadband도 유지했다. 순수 회전·후진도 이동에 포함한다. 이외 reward 함수는 AST 비교로 불변 확인.

## 검증

16환경×4초 rough, zero leg action PD+arm IK로 실제 reward 호출과mask를 포함한 analytic 수식 비교. Raw 및metric 모두 정지0, 이동에서는 기존 수식 유지. 아래 weighted는 /100 이전이다. 학습된 보행 성능 시험은 아니다.

| 조건 | force weighted | velocity weighted |
|---|---:|---:|
| stand | -0.000000 | -0.000000 |
| forward | -0.342479 | -0.022157 |
| turn | -0.532160 | -0.007949 |
| forward_turn | -0.297716 | -0.006262 |

Shadow probe에서 접촉력50N·stance 속도1m/s를 강제로 공급해도 정지와deadband에서는 force/vel 모두0이었다. 전진·후진·좌우 회전·복합 명령은 force−2, stance slip−1.729329를 반환한다. 센서 NaN/stale velocity NaN 독립성, contact44 제거,755 관측/737 배포 계약·history·실제 runner storage[24,16,755]도 재검사했다. 관측에 접촉 센서 값은 재추가하지 않았다.

기존 B1 checkpoint·744 AS2 원본·URDF·gain·PPO 파일의hash 보존 확인. 실제 학습/optimizer step0. 변경 전 파일은 before_source/, diff는 changes.patch, 수치는 after/result.json 및 change_scope.json. 이전 gait_clock 보고서의 정지 shaping penalty 설명은 이번 수정으로 대체한다.

재현: 복제본 루트에서 기존 PYTHONPATH를 설정하고 `python audit_observations.py --output-stage feet_height_walkmask`. Git status는 기존 dubious ownership 오류여서 Git 설정을 변경하지 않고 별도 diff/hash 사용.
