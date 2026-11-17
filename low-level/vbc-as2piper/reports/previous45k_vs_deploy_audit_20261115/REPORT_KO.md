# 이전45k 학습 코드와 현재 deploy 비교

## 비교 기준

- 이전: low-level-as2piper-head-orin-nocontact의 B조건45k 학습. manifest source_sha256 대조 결과 현재 폴더에서 학습 후 달라진 추적 파일은 URDF만 확인. 학습 당시 URDF는 reports/deployment_workspace_review_20261114/trained_asset/urdf/as2piper_head_orin_training.urdf 복구본(SHA d0d667...)을 사용.
- 현재: low-level-as2piper-head-orin-deploy. 실행 Python/launcher/hardware JSON61개 비교에서 수정2개(as2_head_orin_config.py, manip_loco.py), 추가1개(piper_control.py). 나머지58개 동일. URDF 별도 비교.

## 실제 동작 차이

| 항목 | 이전45k | 현재deploy | 의미 |
|---|---|---|---|
| EE 목표중심 | 실제Arm Base 위치+몸통회전 | standing Arm Base 높이/roll/pitch 불변 가상중심 | crouch/tilt해도 목표중심이 같이 내려가거나 기울어지지 않음 |
| 중심값 | arm_base_offset[0.22435,0,0.1319]을 몸통 pose로 변환, 추가offset0 | body XY + yaw*[0.22435,0,0.4563069024] + local ground z | standing에서 같고 몸통pose변경 시 달라짐. XY/yaw는 몸통을 따라감 |
| 반경 |0.40~0.48m|0.35~0.80m|근거리~원거리 확장|
| 위치고도각 |0.10~0.50rad|−π/2.5~π/3(−72~60°)|원본B1의 위치각도 사용|
| 위치yaw |±0.25rad|±1.20rad(±68.75°)|좌우범위 확장|
| 목표높이 |standing에서 약0.496~0.686m|가상지면기준0.02~1.149127m|상한=center_z+rmax*sin(phi_max)|
| 경로기하검사 |실제몸통좌표 box 및 지면검사|동일기본검사 + 가상지면기준2cm/최대높이 경로검사|목표생성IK 검사는 추가하지 않음|
| collision box |몸통전방 경계0.326m|0.346m|머리전방끝을 포함하도록20mm확장, 나머지면유지|
| 거절10회 후 |마지막거절후보가남을수있음|기존목표와orientation delta복구|무효목표잔류오류 수정|
| 팔목표전송 |관절한계 clamp만|관절한계+3rad/s 목표slew filter|50Hz tick당 최대0.06rad, 실측관절속도보장은 아님|
| 팔 URDF/PhysX velocity |5rad/s|3rad/s|모델속도제한을정합|
| URDF |카메라추가전23.328064kg|손목camera/mount 포함23.430340kg|약0.102276kg 추가, 기존공통link질량/관성/장착원점 동일|

## 코드 정리이나 실효동작 유지

- gain400/40: 이전은arm/finger DOF에하드코딩, 현재는config에서분리해동일값적용. arm gain DR기구는추가했으나 현재range[1,1]로고정하여실제400/40유지.
- 이전 arm_motor_strength_range[0.7,1.3]는샘플되지만팔policyaction/torque가마스킹되어실제400/40drive에반영되지않았음. 현재[1,1]명시고정. 다른DR값/hardware.json 동일. privilegedobs에는leg strength만포함되어arm strength입력은없음.
- 다리 action_delay config3→1로명시화. 이전코드는config3과무관하게global_steps240000전0tick/이후1tick지연으로하드코딩. 현재도동일schedule, 지연20ms유지. action history저장길이5→3이나관측차원/실제적용지연동일.
- arm 추가명령delay0, acceleration None. 실제새지연/가속도제한없음. reset때새filter의상태만해당env에맞춰초기화.
- 기준각/고도각으로pitch계산하는식은이전과동일. floor pitch혼합은중간에추가했다가현재완전제거. 기준각2.4552161377, roll offset0, orientation delta각±0.08 유지.

## 동일 항목

-20DOF/18action, 팔action마스킹/별도DLS IK, 그리퍼열림유지.
-755training/737deployment, proprio67/history670/priv18, contact observation제거 및Gait Clock 활성화상태.
-초기joint각·root reset높이0.3264069024·초기standing EE위치/방향.
-Reward소스/scale, feet height0.18m, 기존Contact Shaping 및정지walking_mask, base/defaultpose보상, velocity/EE reward.
-PPO/GAE/actor-critic/history/DAgger/ROA/JIT 경로와설정.
-다리gain/armature, 다리DR/질량·CoM DR/friction/push 및standing push2.5배.
-IK50Hz·물리200Hz(4substeps동안동일팔목표유지), arm effort100Nm×6모델값. effort실기정격검증완료 의미아님.

## 학습 조건 선택과 미적용

-기존45k는B조건(max_contact_force200N). 복제본force_ablation/configs.py도동일하여 **B태스크를선택하면200N**. 공통AS2기본태스크는이전/현재모두40N이므로기본진입점을B와혼동하면안됨.
-Feet Height0.18,Contact Clock on등은이번에새로바꾼것이아니라이전45k에도적용됐던값.
-Ground Proximity/Valid Contact 보상미적용. 오프라인workspace bank/IK유효성필터없음. 팔제어용IK는이전과동일하게존재.
-새범위전체의실기/동적안정성인증이나새학습완료 의미아님. 이번요청에서는비교만수행, 코드수정·학습없음. 기존학습/checkpoint보존.

## 근거파일

-audit.json: 파일목록/hash drift/URDF semantic변경. runtime.patch: 실행파일3개전체diff. asset.patch: 실제45k URDF와현재URDF diff.
-Git status는기존dubious ownership으로실패. 전역Git설정변경없음.
