# 동일 VBC 환경 보상 비교

[실험 조건과 판정 기준](PROTOCOL_KO.md) · [보상 효과 분리 비교](REWARD_EFFECT_COMPARISON_KO.md)

목표: 전진0.25m/s, 회전0.6rad/s. 초기2초 이후 속도를 집계하며 넘어짐은 전체64환경×20초에서 집계한다.
raw foot/body contact force에는 SDK 보고 누락 가능성이 있다. 접촉 bit와 정상반력 N은 구분한다.

| 보상 | 완료 update | 전진 실측(m/s) | 제자리 회전 yaw(rad/s) | 혼합 vx/yaw | 넘어짐 | 무넘어짐 환경 | Baseline |
|---|---:|---:|---:|---|---:|---:|---|
| original_vbc_port | 201 | 0.187 | 0.056 | 0.124 / 0.049 | 209 | 10.9% | FAIL |
| original_vbc_port | 401 | 0.323 | 0.066 | 0.270 / 0.065 | 114 | 34.4% | FAIL |
| original_vbc_port | 600 | 0.381 | 0.087 | 0.392 / 0.167 | 62 | 50.0% | FAIL |
| stable_net_force | 201 | -0.001 | -0.008 | -0.002 / -0.003 | 5 | 92.2% | FAIL |
| stable_net_force | 401 | 0.006 | -0.021 | 0.000 / -0.003 | 2 | 96.9% | FAIL |
| stable_net_force | 600 | 0.011 | -0.042 | 0.005 / -0.018 | 7 | 92.2% | FAIL |
| stable_proxy | 101 | 0.001 | 0.005 | -0.001 / 0.003 | 7 | 92.2% | FAIL |
| stable_proxy | 201 | 0.012 | 0.009 | 0.006 / 0.013 | 10 | 85.9% | FAIL |
| progress_gain2_proxy | 101 | 0.052 | 0.022 | 0.025 / 0.024 | 22 | 76.6% | FAIL |
| progress_gain2_proxy | 201 | 0.164 | 0.117 | 0.091 / 0.109 | 39 | 59.4% | FAIL |
| stable_proxy_ground | 101 | 0.005 | 0.021 | 0.003 / 0.014 | 14 | 82.8% | FAIL |
| progress_gain2_proxy_yaw4 (branch201) | 301 | 0.172 | 0.254 | 0.111 / 0.230 | 32 | 68.8% | FAIL |
| progress_gain2_proxy_yaw4 (branch201) | 401 | 0.214 | 0.401 | 0.148 / 0.345 | 42 | 65.6% | FAIL |
| progress_gain2_proxy_ground_floor (branch201) | 301 | 0.227 | 0.216 | 0.164 / 0.186 | 72 | 48.4% | FAIL |
| progress_gain2_proxy_smooth_yaw4 (fresh0) | 101 | 0.011 | 0.033 | 0.003 / 0.028 | 9 | 89.1% | FAIL |
| progress_gain2_proxy_smooth_yaw4 (fresh0) | 201 | 0.058 | 0.195 | 0.039 / 0.139 | 13 | 84.4% | FAIL |
| progress_gain2_proxy_smooth_yaw4 (fresh0) | 401 | 0.160 | 0.462 | 0.105 / 0.359 | 18 | 78.1% | FAIL |
| progress_gain2_proxy_smooth_yaw4_contactmatch (fresh0) | 101 | 0.015 | 0.037 | 0.006 / 0.027 | 14 | 84.4% | FAIL |
| progress_gain2_proxy_smooth_yaw4_contactmatch (fresh0) | 201 | 0.082 | 0.191 | 0.045 / 0.130 | 13 | 84.4% | FAIL |
| progress_gain2_proxy_smooth_yaw4_contactmatch (fresh0) | 401 | 0.241 | 0.429 | 0.167 / 0.404 | 12 | 84.4% | FAIL |
| progress_gain2_proxy_smooth_yaw4_contactmatch (fresh0) | 600 | 0.304 | 0.500 | 0.241 / 0.452 | 13 | 84.4% | FAIL |
| progress_gain2_proxy_smooth_yaw4_footground (fresh0) | 101 | 0.019 | 0.041 | 0.010 / 0.025 | 16 | 81.2% | FAIL |
| progress_gain2_proxy_smooth_yaw4_footground (fresh0) | 201 | 0.108 | 0.276 | 0.045 / 0.194 | 14 | 81.2% | FAIL |
| progress_gain2_proxy_smooth_yaw4_footground (fresh0) | 401 | 0.184 | 0.491 | 0.133 / 0.448 | 15 | 84.4% | FAIL |
| progress_gain2_proxy_smooth_yaw4_footground (fresh0) | 600 | 0.230 | 0.507 | 0.183 / 0.474 | 7 | 89.1% | FAIL |
| progress_gain2_proxy_smooth_yaw4_contactmatch_deadband (fresh0) | 101 | 0.012 | 0.039 | 0.005 / 0.026 | 7 | 92.2% | FAIL |
| progress_gain2_proxy_smooth_yaw4_contactmatch_deadband (fresh0) | 201 | 0.092 | 0.186 | 0.040 / 0.154 | 12 | 84.4% | FAIL |
| progress_gain2_proxy_smooth_yaw4_contactmatch_deadband (fresh0) | 401 | 0.199 | 0.395 | 0.108 / 0.348 | 14 | 82.8% | FAIL |
| progress_gain2_proxy_smooth_yaw4_contactmatch_deadband (fresh0) | 601 | 0.299 | 0.475 | 0.212 / 0.447 | 6 | 92.2% | FAIL |
| progress_gain2_proxy_smooth_yaw4_contactmatch_deadband (fresh0) | 1000 | 0.309 | 0.478 | 0.238 / 0.461 | 9 | 87.5% | FAIL |
| progress_gain2_proxy_smooth_yaw4_footground_deadband (fresh0) | 101 | 0.015 | 0.039 | 0.007 / 0.029 | 13 | 85.9% | FAIL |
| progress_gain2_proxy_smooth_yaw4_footground_deadband (fresh0) | 201 | 0.086 | 0.283 | 0.049 / 0.224 | 15 | 84.4% | FAIL |
| progress_gain2_proxy_smooth_yaw4_footground_deadband (fresh0) | 401 | 0.153 | 0.468 | 0.109 / 0.436 | 12 | 82.8% | FAIL |
| progress_gain2_proxy_smooth_yaw4_footground_deadband (fresh0) | 600 | 0.222 | 0.508 | 0.150 / 0.462 | 9 | 87.5% | FAIL |
| stable_proxy_resume1 | 301 | 0.012 | 0.025 | 0.004 / 0.023 | 11 | 84.4% | FAIL |
| progress_gain2_proxy_resume1 | 301 | 0.193 | 0.225 | 0.133 / 0.195 | 47 | 64.1% | FAIL |
| progress_gain2_proxy_resume1 | 401 | 0.226 | 0.237 | 0.161 / 0.221 | 63 | 54.7% | FAIL |
| stable_proxy_ground_resume1 | 201 | 0.025 | 0.084 | 0.013 / 0.055 | 8 | 89.1% | FAIL |
| stable_proxy_ground_resume1 | 301 | 0.039 | 0.122 | 0.014 / 0.097 | 6 | 92.2% | FAIL |

## 재개 및 비교의 한계

모든 후보는 처음에 scratch로 시작했다. 2026-10-30 07:12:17 UTC에 세 실험의 로그가 동시에 중단되었다. traceback 및 현재 cgroup OOM 기록만으로 정확한 종료 원인은 확정할 수 없다.
stable_proxy와 progress_gain2_proxy는 201회, stable_proxy_ground는 101회 완료 checkpoint에서 optimizer/history 상태를 복원했다. 이후 결과에는 resume1을 표시한다. 물리 환경·episode·지지 높이 cache는 초기화되므로 중단 없는 학습과 완전히 동일하지 않으며, 서로 다른 재개 시점도 비교 한계다.
yaw4는 progress_gain2_proxy의 동일201회 checkpoint/Adam/history/seed에서 분기하며 yaw weight만2→4로 변경한다. 환경은 재초기화한다. 높이 기준은 grid를 유지한다.
ground_floor 분기도 동일201회 정책에서 시작하며 마지막 접촉 높이 기준과 swing 최소 높이 보상을 묶어 비교한다. stance 목표는 대칭0, 평지의4cm 대칭 목표는 유지한다. 이것은 높이 보상 그룹의 조합 효과이며 개별항의 효과를 모두 분리했다고 주장하지 않는다.
기존 온라인 run 및 B1 코드는 중지·수정하지 않았다.

## 학습 상태

갱신 시각: 2026-10-31T16:37:01.605451+00:00
- stable_proxy: 중단 / 실행 확인 안 됨; logger 200 / 600
- stable_proxy_resume1: 완료; logger 334 / 600
- progress_gain2_proxy: 중단 / 실행 확인 안 됨; logger 201 / 600
- progress_gain2_proxy_resume1: 완료; logger 502 / 600
- stable_proxy_ground: 중단 / 실행 확인 안 됨; logger 133 / 600
- stable_proxy_ground_resume1: 완료; logger 304 / 600
- progress_gain2_proxy_yaw4: 완료; logger 432 / 600
- progress_gain2_proxy_ground_floor: 완료; logger 331 / 600
- progress_gain2_proxy_smooth_yaw4: 완료; logger 494 / 600
- progress_gain2_proxy_smooth_yaw4_contactmatch: 완료; logger 599 / 600
- progress_gain2_proxy_smooth_yaw4_footground: 완료; logger 599 / 600
- progress_gain2_proxy_smooth_yaw4_contactmatch_deadband: 완료; logger 999 / 1000
- progress_gain2_proxy_smooth_yaw4_footground_deadband: 완료; logger 599 / 600

모든 후보가 통과 전에는 안정적 보행이나 장시간 수렴으로 표시하지 않는다.
