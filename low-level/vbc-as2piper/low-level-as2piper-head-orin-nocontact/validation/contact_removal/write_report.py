from pathlib import Path
import json,csv,hashlib,difflib,ast
import numpy as np
root=Path(__file__).resolve().parents[2];out=root/'validation/contact_removal';old=root.parent/'low-level-as2piper-head-orin'
obs=json.loads((out/'after/result.json').read_text());ppo=json.loads((out/'ppo_path.json').read_text());height=json.loads((out/'swing_height_distributions.json').read_text());force=json.loads((out/'force_threshold_recommendation.json').read_text())
protected=json.loads((out/'preservation_before.json').read_text());changed=[p for p,h in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not changed
checks={}
for f in ['legged_gym/envs/rewards/maniploco_rewards.py','legged_gym/envs/manip_loco/b1z1_config.py','validation/hardware.json','resources/robots/as2piper/urdf/as2piper_head_orin_training.urdf','rsl_rl_local/rsl_rl/algorithms/ppo.py','rsl_rl_local/rsl_rl/storage/rollout_storage.py','legged_gym/utils/helpers.py']:
 checks[f]=hashlib.sha256((root/f).read_bytes()).hexdigest()==hashlib.sha256((old/f).read_bytes()).hexdigest()
assert all(checks.values())
proof={'protected_files':len(protected),'changed_original_files':changed,'copied_reward_gait_hardware_ppo_unchanged':checks,'optimizer_steps':ppo['optimizer_steps'],'network_parameters_unchanged_in_backward_test':ppo['all_parameters_unchanged']}
(out/'preservation_after.json').write_text(json.dumps(proof,indent=2)+'\n')
delta=[];stats=[]
for path in list((root/'legged_gym').rglob('*.py'))+list((root/'rsl_rl_local').rglob('*.py'))+[root/'run_head.sh']:
 rel=path.relative_to(root);orig=old/rel
 if orig.exists():
  diff=list(difflib.unified_diff(orig.read_text().splitlines(True),path.read_text().splitlines(True),fromfile='744-original/'+str(rel),tofile='700-nocontact/'+str(rel)))
  if diff:delta+=diff;stats.append({'file':str(rel),'added':sum(x.startswith('+') and not x.startswith('+++') for x in diff),'removed':sum(x.startswith('-') and not x.startswith('---') for x in diff)})
(out/'observation_changes.patch').write_text(''.join(delta));(out/'diff_stat.json').write_text(json.dumps(stats,indent=2)+'\n')

def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
report='''# Contact Observation 제거 및 Reward/Gait 추천

## 적용 범위

기존 `low-level-as2piper-head-orin` 744차원 코드는 그대로 보존했다. 독립 복제본 `low-level-as2piper-head-orin-nocontact`에만 관측 제거를 구현했다. 새 태스크는 `as2piper_head_orin_nocontact`. 기존 B1 45000 checkpoint·AS2 URDF·20 DOF/18 action·TCP/IK·좌표계·gain·초기 자세 유지.

구현은 Observation 제거와 입력 계약 검사뿐이다. Feet height=0.30m, contact force=40, gait=False, 모든 reward scale·PPO 알고리즘 유지. 아래0.15/200N 등의 값은 추천안이며 설정에 반영하지 않았다.

## 1. 입력·학습/배포 경로

| 구간 | 기존 | 새 구성 |
|---|---|---|
| 현재 proprio |66|62|
| Privileged physics |18|18|
| History |10×66=660|10×62=620|
| 학습 observation |744|700|
| 배포 입력 |726|682|
| Actor backbone 입력 |66+latent20=86|62+latent20=82|
| Critic backbone 입력 |66+priv18=84|62+priv18=80|
| Policy action |18|18|

현재 contact53:57를 실제 concatenate 목록에서 제거했다. History는 제거된62차원 frame만 저장하며 삭제한44개를0으로 채우는 경로는 없다. 새 인덱스: IMU0:5, q5:23, dq23:41, 최신 leg action41:53, command53:56, EE position56:59, 기존 EE orientation zero59:62, priv62:80, history80:700. 기존 orientation zero3은 원본 기능이며 contact padding이 아니다.

학습 actor는62+priv latent20, DAgger teacher는priv18, student는10×62 history. 배포는 `cat(obs[:,:62],obs[:,80:])`의682개와 history branch를 사용한다. Helper/JIT·play/interface가 config 차원을 읽어 자동 연결된다. Actor/Critic 입력 크기 guard를 추가하여 이전744 입력이 조용히 잘못 해석되지 않게 했다. JIT는 이전726을 거부한다. 기존744/B1 checkpoint의 weight는 새 입력 layer와 호환되지 않으며 이식·padding·부분 로드는 구현하지 않았다.

검증: 16환경×4초 rough PD/IK 관측, 모든62 필드와history/scale/reset 검사. Contact bit 반전·6축 sensor NaN 주입에도 전체700 observation 불변. 실제 runner 생성·storage700·inference18 확인. PPO/DAgger 실제 backward를 통과하고 optimizer.step 직전 중단. 두 optimizer state 비어 있음, 모든 파라미터 해시 불변. JIT682 대 native history 출력 최대차0; privileged 변경 영향0. 실제 PPO 학습은 실행하지 않았다. 이 검증은 새 정책의 보행 성능 시험이 아니다.

## 2. Contact Reward 계산 경로

Reward 소스는 복제 전과 SHA256 동일하다. Active contact 함수 source에서 실제6축 sensor tensor 참조가 없음을 확인했다. feet_contact_forces/jerk는 simulator net contact force, feet_drag/air_time는 simulator contact_filt, collision은 simulator body contact, shaping force/vel는 simulator force/live foot velocity를 사용한다. 따라서 실제 발 센서가 없는 배포 observation과 별도로 학습 reward를 계산할 수 있다.

기존 `foot_contacts_from_sensor` 필드는 이름만 남아 있고 값은net-force contact_filt다. Actor 관측에서는 읽지 않는다. Isaac force sensor 할당 자체는 유지했으나 실물 관측에 요구되지 않는다. 미활성 함수 `_reward_foot_contacts_z`는 sensor tensor를 읽지만 현재 scale이 없어 호출되지 않는다. 나중에 활성화하면 별도 검토가 필요하다. Rough에서 net-contact 보고 누락 가능성은 여전히 존재하며 observation 제거가 reward 접촉 신호의 정확성을 보장하지 않는다.

## 3. B1 45000 스윙 높이 계측

보호된 원본 환경/정책을 다시 읽기 전용 로드하여 flat/rough 각32환경×40초, 네 command조건8환경씩 평가했다. 원본 gait=True·history actor·reward 유지, DR/push off, 이전 평가와 같은 중앙 EE 궤적. 초기2초 및 episode age≤2초 제외. Ground는 각 발 아래 terrain grid 최소3점 높이. 발 중심 높이에서 B1 collision radius0.04m를 뺀 값을 발바닥 clearance로 정의했다.

아래 스윙 표본은 sensor6≤1.5 AND net force≤2N인 표본이다. sensor-only/net-only/clock-swing 정의도 JSON에 별도 기록하여 rough 판정 불확실성을 드러냈다. 국소 지면 기준 높이는 전방 장애물 전체 경로의 최소 clearance를 의미하지 않는다.

'''
rows=[]
for v in height['records']:
 if v['condition']!='stand' and v['selection']=='agreed_swing':
  z=v['bottom_clearance_m'];rows.append([v['terrain'],v['condition'],z['count'],*[f"{100*z[k]:.2f}" for k in ['p50','p95','p99']]])
report+=table(['지면','명령','표본수','P50 cm','P95 cm','P99 cm'],rows)
report+='''
연속 스윙 event의 peak도 swing_events.csv에 별도 보관했다. 평지 전진 peak P95 약8.22cm, 복합8.46cm다. 표본 P95와 event peak P95는 다른 통계다. 다리 길이 B1 0.70m, AS2 0.42544m로 길이비0.60777. 단순 기하학 유사 환산 시 전진 표본P95 약4.40cm, peakP95 약5.00cm가 된다. 질량·관성·주파수·terrain은 상사 조건이 아니므로 AS2 성능 예측으로 인증하지 않는다.

## 4. AS2 feet_height_target 후보

현재 수식은 `min(sqrt(h_FL²+h_FR²)−target,0)×walking_mask`이며 h는 발 중심의 지면 상대 높이다. 목표 이하만 음수인 hinge이고 발 하나의 clearance 목표나 squared tracking error가 아니다. AS2 앞발 하나가 접지해 중심높이0.028m라고 가정하면 보상0에 도달할 다른 앞발 clearance는 다음과 같다.

'''
report+=table(['target','한 발 접지 시 다른 앞발 clearance','판단'],[[f'{t:.2f}m',f'{100*np.sqrt(t*t-.028**2)-2.8:.2f}cm',label] for t,label in zip([.15,.18,.30],['세 후보 중 우선 검증','더 높은 스윙 유인, 평지 근거 약함','AS2 평지 기본값 후보로 과도함'])])
report+='''
동일한 B1 궤적을 길이비로 AS2에 투영하고, 기존 AS2 PD rollout 상태에도 후보 수식을 사후 적용했다. reward 설정이나 물리 궤적을 바꾸지 않은 counterfactual이다.

'''
rows=[]
with (out/'feet_height_candidates.csv').open() as f:
 for v in csv.DictReader(f):
  if v['terrain'] in ['flat','plane'] and v['condition']=='forward':rows.append([v['source'],v['target_m'],f"{float(v['mean_raw_hinge']):.5f}",f"{100*float(v['target_reached_fraction']):.2f}%"])
report+=table(['기준 상태','후보m','mean raw (=scale1 weighted)','보상0 도달률'],rows)
report+='''
추천: 지정된 세 후보 중0.15를 다음 학습 비교의 우선 후보로 선택. 다만 실제 B1 유사 스윙 투영의 pair-norm P95는 약0.079m로 세 후보 모두보다 낮다. 0.15도 이미 검증된 최적 높이라는 뜻은 아니다. 모든 표본이 hinge 아래이면 target을0.30→0.15로 줄여도 같은 상태에서의 높이 미분은 같고 walking 구간 reward offset만 바뀐다. 따라서 이 변경만으로 발을 덜 들거나 더 안정적으로 걷는다고 단정할 수 없다. 실제 학습 없이 세 값의 정책 성능 우열을 비교하지 않았다. 현재 설정0.30은 유지했다.

## 5. Contact force threshold 추천

B1 질량57.98697kg, AS2 적재 모델23.32806kg(비0.4023). B1 steady 구간 net force 분포를 사용했다. 원본 sensor6 norm의40 threshold는 거의 작동하지 않으므로40×질량비처럼 환산하지 않는다. AS2 nominal 발별 하중은 전방 약73.6/73.9N, 후방40.6/40.8N이다.

B1 각 발 P95/P99를 B1 정지 발별 평균으로 나누고 AS2 발별 nominal 하중을 곱했다. 이동 조건 전체의 AS2 환산 P95 최대 약176.9N, P99 최대 약222.8N. Pooled P99에 전체 질량비만 적용하면 약127~134N이지만 전방 하중 편중과 발별 tail을 놓친다.

추천: 기존 scalar max_contact_force의 다음 시험 후보200N, 민감도180/200/220N. 이는 보통 지지를 무조건 penalize하지 않고 큰 tail을 억제하려는 reward 기준의 제안이다. 실제 AS2 동적 분포·모터 정격에서 확인한 안전 한계가 아니며 스윙 duty·팔 운동·DR에 따라 재검토해야 한다. 정적 하중40N보다 높은 front support를 정상적으로 허용할 수 있는 후보라는 근거는 있다. 현재40N 및scale−0.001은 변경하지 않았다. 발별 normalize 방식은 향후 별도 설계이며 구현하지 않았다.

## 6. Gait 차이와 방향

| 항목 | B1 45000 | 현재 AS2 / 새 no-contact |
|---|---|---|
| observe_gait_commands |True|False 유지|
| 추가 입력 |gait index1+clock4|없음|
| 설정 frequency |2Hz|2Hz 상속, gate가 꺼져 gait shaping에 사용되지 않음|
| Contact shaping |raw 음수×scale−2 → mismatch 양의 보상|raw 양수×scale−2 → mismatch penalty, gate=False로0|
| Vel source |초기화된 foot_velocities cache|live rigid-body foot velocity|
| Foot contact input |4+history40|완전 제거|

Clock5는 phase·시간·velocity command로 생성하므로 실물 발 센서가 필요하지 않다. Contact observation 삭제와 gait clock 삭제는 별개다. 원본 B1 gait=True의 학습 성공을 AS2 gait=False의 동등성 근거로 삼을 수 없다. 원본 shaping의 부호·stale velocity 문제도 분리해야 한다.

추천 방향: 이번 baseline은 AS2 False를 유지해 observation 제거 효과부터 분리한다. 다음 별도 실험에서만 clock5 + 수정된 contact shaping의 사용 여부를 비교한다. Clock을 켜면 proprio67,학습755,배포737로 다시 달라져 새로운 입력 계약·재학습이 필요하다. 주파수는 질량 비례로 임의 변경하지 않는다. 현재 gain·hardware에서2Hz가 최적이라는 증거도 없으므로 이번에는 변경하지 않았다.

## 7. 산출물·재현·중단

- after/result.json:700차원 실제 관측/runner 검증, after/real_rollout.npz:PD 입력/reward.
- ppo_path.json:Actor/Critic/History/DAgger gradient·JIT·구차원 거부·optimizer0 검증.
- swing_height_distributions.json, swing_events.csv, feet_height_candidates.csv:계측·후보 비교.
- force_threshold_recommendation.json:하중 환산·추천 근거.
- b1_swing/*/samples.npz:원본 B1 계측 시계열, ground/foot center/clearance·sensor/net force·clock 포함.
- observation_changes.patch / diff_stat.json:744 버전 대비 실제 코드 변경.
- preservation_before/after.json:기존 원본·checkpoint·744 버전 보존, 새 reward/gait/hardware/PPO 원본 동일성.

새 복제본 디렉터리에서 PYTHONPATH=`$PWD:$PWD/.deps:$PWD/rsl_rl_local:$PWD/../third_party/isaacgym/python`을 설정하고 `python audit_observations.py`, `python audit_ppo_path.py`로 구조 검증을 재실행할 수 있다. B1 height 계측 스크립트는 원본 B1 PYTHONPATH를 사용해야 한다. 실제 학습용 run_head.sh는 존재하지만 이번에 실행하지 않았다.

Git status는 기존 dubious ownership 오류로 확인할 수 없어 Git 설정을 변경하지 않고 SHA256·별도 diff로 보존 확인했다. 실제 PPO 학습·optimizer 업데이트·checkpoint 덮어쓰기 없이 사용자 검토 단계에서 중단한다.
'''
(out/'REPORT_KO.md').write_text(report)
if '## Contact observation 제거 전용 복제본' not in (root/'CHANGES_KO.md').read_text():
 with (root/'CHANGES_KO.md').open('a') as f:f.write('''\n## Contact observation 제거 전용 복제본\n\n- 원본744 버전은 low-level-as2piper-head-orin에 보존. 이 디렉터리는 low-level-as2piper-head-orin-nocontact, task=as2piper_head_orin_nocontact.\n- Contact4+history40 실제 제거: proprio62, privileged18, history620, 학습700, 배포682. 삭제 슬롯 zero padding 없음. 입력 크기 검사를 추가해 이전744/726을 거부.\n- Actor82/Critic80/history frame62, 실제 runner/storage700, DAgger·PPO backward 및 JIT parity 검사. optimizer0, 파라미터 불변.\n- Reward 소스·scale·gait(False)·PPO·URDF·gain·초기 자세 유지. 실물 foot sensor는 새 actor 입력에 없음.\n- B1 45000의 swing 높이 계측과 AS2 후보 사후 비교를 수행. Height0.15, force200N은 후속 실험 추천이며 반영하지 않음. 현재0.30/40N 유지.\n- 상세: validation/contact_removal/REPORT_KO.md. 실제 학습 미실행.\n''')
print('report saved; protected',len(protected),'files unchanged; changed code',stats)
