# AS2 PiPER 전후 이동 레일과 전방 팔 베이스 안정성 검토

**설계 가정:** 컴퓨팅 장치 1.0 kg을 본체 중앙 상부(z=155 mm)에, 레일 0.50 kg을 전후 축에, 캐리지 0.50 kg을 팔 베이스에 배치했다. PiPER 베이스는 중앙에서 시작해 전방으로 50 mm 이동한다. 세 질량은 실측 전 가정이며 양 조건에 똑같이 반영했다. 총 모델 질량은 24.89 kg이다.

**시험 한계:** 발 움직임은 관절 PD+중력 보상을 사용한 작은 scripted stepping 동작이다. 가상 base force와 학습 보행 정책은 없다. 네 발이 대부분의 프레임에서 접촉하므로 실제 보행 안정성이나 보증 시험으로 볼 수 없다. 자홍색 점은 전체 시스템 CoM이다.

| 조건 | base 기준 평균 CoM x (m) | rail 끝 CoM x (m) | 최대 pitch/roll (deg) | 지지 여유 최솟값 (m) | 3발 미만 |
|---|---:|---:|---:|---:|---:|
| rail_center_actuator_step | 0.007 | 0.010 | 2.08/0.49 | 0.166 | 0.25% |
| rail_forward_actuator_step | 0.019 | 0.022 | 1.35/0.59 | 0.166 | 0.25% |

팔 베이스를 +50 mm 이동했을 때 rail release 시 CoM은 base 기준 x=+10.0 mm에서 +21.8 mm로 이동했다. 변화량은 11.9 mm다. 양 조건에서 유효한 지지 여유 최소는 같았고, 전방 조건의 최대 pitch는 더 커지지 않았다. 이 짧은 저진폭 시험에서 전방 이동이 즉시 전복을 유발하는 징후는 보이지 않았다.

두 조건 모두 초기 1초 동안 base가 약 36 mm 가라앉았으며, 작은 scripted stepping 구간에서 base가 각각 45/39 mm 뒤로 이동했다. 실제 gait 정책 시험이 아니므로 이 이동을 보행 성능으로 해석할 수 없다. 지지 여유는 3발 이상 접촉한 프레임에서만 계산했다.

## 영상과 데이터

- [중앙 고정 vs 전방 +50 mm 동시 비교](./center_vs_forward_side_by_side.mp4)
- [중앙 고정 조건](./rail_center_actuator_step.mp4)
- [전방 이동 조건](./rail_forward_actuator_step.mp4)
- [CoM·기울기·지지 여유 시계열](./com_stability_timeseries.png)
- [재현 스크립트](../../evaluate_forward_arm_rail.py)
- 상세 시계열은 두 `*.csv`, 요약값은 `summary.json`에 기록했다.
