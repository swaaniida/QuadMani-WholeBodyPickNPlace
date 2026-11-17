# AS2 단독·중앙 PiPER·머리 위 PiPER 비교

AS2와 PiPER 원본 URDF 링크 질량·관성을 사용했다. 레일/어댑터 질량은 제외했다. 장착 좌표는 기존 연구 MJCF의 후보값이며, 실제 체결면 기준으로 확인되지 않았다.
**중요:** head 위치는 사용자가 지정한 PiPER 밑면 중심 `(0.2151, 0, 0.1319) m`를 반영했다. 베이스 링크 원점은 mesh 바운딩박스 중심 오프셋 `(-0.00925, 0, 0) m`를 보정해 `(0.22435, 0, 0.1319) m`로 설정했다. PiPER 밑면 footprint의 평면 높이를 유지해 STL을 25×17 ray grid로 검사하면 간격/관통이 나타난다(상세값은 JSON). 실제 PiPER 접촉 패드와 AS2 하중 지지부/체결 홀은 별도 검증이 필요하다. 중앙 후보는 이전 비교 후보이다. CoM/관성과 작업공간 결과는 기구 배치 비교용이며 보행 안정성 증거로 사용할 수 없다.
저장소에는 AS2 공식 보행 생성기/균형 제어기가 없고, 확인된 Unitree control plugin은 관절 명령 전달기다. 동적 보행 및 외란 비교는 공식 제어기를 재현할 수 있는 실행 artifact와 검증된 장착 좌표가 확보된 뒤 수행한다.

| 구성 | 총 질량 (kg) | CoM x/y/z, base 기준 (m) | 관성 주값 (kg·m²) |
|---|---:|---:|---:|
| AS2 only | 17.640 | 0.009/-0.000/-0.009 | 0.246, 0.644, 0.736 |
| AS2 + center PiPER | 22.456 | 0.009/-0.000/0.070 | 0.811, 0.934, 1.402 |
| AS2 + head PiPER | 22.456 | 0.056/-0.000/0.076 | 0.627, 1.383, 1.667 |

| 작업공간 지표 | Center | Head | 변화 |
|---|---:|---:|---:|
| 샘플 voxel 부피 (m³) | 0.246 | 0.259 | 1.05x |
| 머리 기준 최대 전방 도달 (m) | 0.481 | 0.696 | +0.215 |
| 머리 앞 TCP 표본 수 | 1924 | 4648 | +2724 |

작업공간은 동일 seed의 관절 샘플을 URDF 관절 범위에서 뽑고, MuJoCo collision 및 지면 아래 TCP를 제외했다. 3 cm voxel 점유 부피는 연속 reachable volume의 근사치이며, 추출 표본 밀도와 collision proxy에 영향을 받는다.

- [3D workspace와 AS2 mesh 시각화](./workspace_3d_comparison.png)
- [workspace point cloud 데이터](./workspace_points.npz)
- [질량·관성·workspace 상세값](./summary.json)
- [고정 mount study MJCF](../../models/mount_workspace/)
