from pathlib import Path
import json,csv,hashlib,xml.etree.ElementTree as ET
import numpy as np
root=Path(__file__).resolve().parent;project=root.parents[1]
rows=[];forces=[];abl=[];paired=[]
contact=['feet_contact_forces','feet_jerk','feet_drag','feet_air_time','tracking_contacts_shaped_force','tracking_contacts_shaped_vel','collision','feet_stumble','foot_contacts_z']
results={}
for terrain in ['flat','rough']:
 for mode in ['normal','zero44']:
  x=json.loads((root/(terrain+'_'+mode)/'result.json').read_text());results[terrain+'_'+mode]=x
  for cond,v in x['conditions'].items():
   for name,z in v['rewards'].items():rows.append({'terrain':terrain,'observation':mode,'condition':cond,'reward':name,'scale':x['scales'][name],**z})
   for name in ['sensor6_mixed_units','sensor_force3_N','net_contact_force_N']:
    for i,foot in enumerate(['FL','FR','RL','RR']):forces.append({'terrain':terrain,'observation':mode,'condition':cond,'signal':name,'foot':foot,**{s:v[name][s+'_per_foot'][i] for s in ['mean','p95','p99']}})
 a=results[terrain+'_normal'];b=results[terrain+'_zero44']
 initial={k:float(np.max(np.abs(np.array(a['initial_state'][k])-np.array(b['initial_state'][k])))) for k in a['initial_state']}
 paired.append({'terrain':terrain,'initial_max_difference':initial})
 da=np.load(root/(terrain+'_normal')/'samples.npz');db=np.load(root/(terrain+'_zero44')/'samples.npz')
 for i,cond in enumerate(a['conditions']):
  v=a['conditions'][cond];w=b['conditions'][cond]
  for key in ['vx_mean','vx_rmse','yaw_mean','yaw_rmse','ee_error_mean_m','ee_error_p95_m','failure_resets','timeouts']:
   abl.append({'terrain':terrain,'condition':cond,'metric':key,'normal':v[key],'zero44':w[key],'difference_zero_minus_normal':w[key]-v[key]})
  for col,label,target in [(0,'vx_rmse',v['command'][0]),(2,'yaw_rmse',v['command'][2]),(3,'ee_mean',0)]:
   av=da['physical'][100:,i*8:(i+1)*8,col];bv=db['physical'][100:,i*8:(i+1)*8,col]
   d=(bv.mean(0)-av.mean(0)) if label=='ee_mean' else np.sqrt(((bv-target)**2).mean(0))-np.sqrt(((av-target)**2).mean(0))
   rng=np.random.RandomState(20261111);boots=d[rng.randint(0,8,(5000,8))].mean(1)
   paired.append({'terrain':terrain,'condition':cond,'metric':label,'paired_env_mean_delta':float(d.mean()),'bootstrap95_env_interval':np.quantile(boots,[.025,.975]).tolist(),'note':'8 matched env replicas; one seed, repeated episodes within each env, not independent multi-seed training'})

def savecsv(name,items):
 with (root/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
savecsv('reward_contributions.csv',rows);savecsv('force_statistics.csv',forces);savecsv('observation_ablation.csv',abl)
(root/'paired_checks.json').write_text(json.dumps(paired,indent=2)+'\n')
# Inventory of actual AS2 policy inputs; no production code changes.
audit={'current_proprio':66,'privileged':18,'history_frames':10,'full_observation':744,'deployment_input':726,'foot_contact_current_indices':[53,54,55,56],'foot_contact_history_indices':[84+66*t+j for t in range(10) for j in range(53,57)],'removed_dimensions':44,'proposed_proprio':62,'proposed_full':700,'proposed_deployment':682,'other_direct_foot_force_observation_fields':[],'indirect_dependencies':['history encoder consumes contact history; its input must be rebuilt/retrained','IMU and joint states encode physical contact effects but are available without foot sensors','privileged motor strength/mass/friction are training-only physical parameters, not foot sensors','gait observation optional clock/phase depends on commands/time; current AS2 disabled','reward and termination contact forces are simulation ground truth; independent of actor input','feet_air_time/contact_filt caches are reward state, not additional observation slots','EE trajectory collision uses simulation terrain height; requires separate deploy command validation but is not foot sensor observation'],'proposal_only':True}
(root/'as2_observation_inventory.json').write_text(json.dumps(audit,indent=2)+'\n')
paths=json.loads((root/'preservation_before.json').read_text());changed=[p for p,s in paths.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=s]
assert not changed
proof={'checked_files':len(paths),'changed_files':changed,'all_preserved':True,'checkpoint_sha256':paths['/data/team/users/suhwan/vbc-legacy/low-level/logs/b1z1-low/original-run01/model_45000.pt'],'learning_or_optimizer_steps':0}
(root/'preservation_after.json').write_text(json.dumps(proof,indent=2)+'\n')

def table(header,rr):return '| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rr)
report='''# B1+Z1 Contact Reward / Observation 평가

## 범위·체크포인트

보호된 원본 `vbc-legacy/low-level/logs/b1z1-low/original-run01/model_45000.pt`를 strict load했다. metadata iteration=45000. W&B 저장 실행 인자에도 `--observe_gait_commands`가 있고 checkpoint history encoder 입력71, actor backbone 입력91이다. B1은 proprio71 + privileged18 + history710 =799. 현재 AS2의744와 구분한다. 원본 reward/env 및 원본 rsl_rl을 직접 read-only import, bytecode 쓰기 비활성. 평가용 코드와 산출물만 현재 저장소 reports 아래에 추가했다.

정상 정책은 평지·rough에서 정지/전진/회전/복합 command를 수행한다. 전체 4실행(2지면×2관측), 각32환경(조건별8개)×40초, 초기2초 제외. Command는 [vx,vy,yaw-rate]=[0,0,0]/[0.4,0,0]/[0,0,0.6]/[0.4,0,0.6]. 원본10초 episode/관절 reset 유지, 학습 완료 시점과 같은1step action delay. DR·push만 끄고 nominal hardware로 평가. Terrain은 원본 trimesh 생성기4×4, rough height=[0,0.1]m; flat은 같은 생성기의height=[0,0]m. 경사·계단 시험은 아니다. 중앙 작업공간의 구면 EE 궤적 r=0.60±0.06m, p=0.10±0.12rad, yaw=±0.20rad를 모든 실행에 같은 시간 함수로 공급했다. reset 직후 원본 초기 EE 전환은 유지되며 이후 공통 궤적으로 복귀한다. 전체 원본 랜덤 EE 범위 검증은 아니다.

Inference는 deterministic history branch로 원본 play와 동일. 정상/zero44의 평지 초기 root/dof/obs는 완전히 동일하다(최대차0). Rough는 같은seed·terrain·reset설정이지만 GPU 물리 초기화 이후 root 최대차0.0335(13차원 상태 혼합), dof 최대차0.00853rad, obs 최대차0.0305가 있었다. Rough 결과에는 이 초기 상태 차이가 함께 포함되며 완전히 동일한 state replay가 아니다. 차이는 paired_checks.json에 공개했다. 이후 물리 궤적은 관측 차이에 따라 달라진다. 실제 PPO 학습·optimizer 업데이트0. 저장 결과 reward 합산은 원본 반환 leg reward와 일치한다.

## 1. AS2 관측 의존성 전수 조사

직접 발 센서/접촉 상태 입력은 현재66 중53:57의4개, history84:744를10×66으로 봤을 때 각 frame53:57의40개가 전부다. 그 외 6축 sensor/발 force/발 air-time/terrain height는 actor observation에 직접 들어가지 않는다. 현 contact 입력은 이전 수정에 따라 net force 기반 contact_filt다.

제거 제안 차원: proprio66→62, history660→620, privileged18 유지, 학습744→700, 배포726→682. 정확한 인덱스는 as2_observation_inventory.json. 아직 적용하지 않았다.

전체 입력의 다른 센서 의존성: roll/pitch2·angular velocity3은 IMU, q18·dq18은 AS2/PiPER encoder, action12는 제어 이력, command3은 상위 제어, EE goal3은 arm-base 기준 목표, orientation3은 원본0이다. IMU·joint motion에 접촉 결과가 간접적으로 나타나도 발 센서가 필요한 입력은 아니다. Privileged18의 mass/COM/friction/motor strength는 학습용 물리 파라미터이며 배포에서 제거하는 별도 구간이다. 선택적 gait clock5는 명령·시간 기반으로 발 센서와 무관하며 현재 AS2에서는 비활성이다.

History encoder 및 actor/critic 입력 layer는 새 proprio 크기에 맞춰 재구성·재학습해야 한다. 44개를0으로 만드는 기존 모델 ablation과700차원 새 정책 학습은 다르다. 환경의 contact_filt/air-time cache와 contact reward·termination은 시뮬레이션 학습용 상태이므로 observation 제거와 함께 삭제할 필요가 없다. EE sampler의 terrain/collision 검사 등 배포용 command 유효성 문제는 별도이며 foot-contact 입력44개에 포함되지 않는다.

## 2. Reward 기여도: 정상 observation 결과

이 절은 reward를 제거하거나 바꾼 실험이 아니다. 실제 호출1회마다 raw를 기록하고 원본 scale을 곱했다. 아래 weighted는 원본 /100 이전 값이며, 모든 reward·모든 조건의 signed/absolute 기여도는 reward_contributions.csv에 있다. Absolute share=E|weighted term| / Σ E|weighted term|, signed share=E weighted term / E branch total. Net total이 상쇄되면 signed share는100%를 넘을 수 있다.

'''
rewardrows=[]
for terrain in ['flat','rough']:
 for cond,v in results[terrain+'_normal']['conditions'].items():
  rewardrows.append([terrain,cond]+[f"{v['rewards'].get('leg/'+n,{}).get('weighted_mean_before_div100',0):.5f}" for n in contact[:6]])
report+=table(['지면','조건','contact force','jerk','drag','air time','shaped force','shaped vel'],rewardrows)
report+='''
원본 수식·실제 작용:

- feet_contact_forces: `1[episode_age>2s] Σ max(||sensor6||−40,0)`, scale−0.001. XYZ force와XYZ torque를 혼합한6축 norm이다. 이40은 물리적 net force40N과 같은 기준이 아니다.
- feet_jerk: reward container의 `self.last_contact_forces` 존재를 검사하지만 값을 `self.env.last_contact_forces`에 저장한다. 원본에서는 항상0. 이를 고친 AS2 jerk는 원본에서 없던 실제 penalty를 켠 것이다.
- feet_drag: sensor6 norm>1.5의 contact bit × 발의 |vx|+|vy|+|vz| 합, scale−0.08. 미끄러짐뿐 아니라 수직 속도도 포함한다.
- feet_air_time: sensor bit로 착지 이벤트와 air timer 갱신, 앞2발 기본, `(air_time−0.5)`의 착지 합×walking mask, scale+2. 실제 raw가 음수이면 짧은 체공에 대한 penalty가 된다.
- shaped_force/vel: gait=True에서 raw가 mismatch의 음수이고 scale도−2이므로 weighted는 양수. 원본은 mismatch가 증가할 때 보상이 증가하는 구조다. 정상 정책이 걷는다는 사실이 이 부호의 적절성을 입증하지는 않는다. AS2는 부호를 이미 수정했고 gait=False로 비활성이므로 이번 B1 양의 값을 그대로 목표로 삼으면 안 된다.
- shaped_vel은 초기화 때 advanced indexing으로 복사한 foot_velocities를 읽는다. 원본에서 이 cache를 매 step 갱신하지 않아 현재 발 속도와 다를 수 있다. 실제 live 발 속도는 samples의 foot_velocities에 따로 보관했다. Feet height/collision 등 나머지 활성 함수도 전체 CSV에 기록했다.

Reward 기여도는 최적화 신호 크기를 나타내며 해당 reward가 성능을 얼마나 개선했는지에 대한 인과 실험은 아니다. Reward별 제거·재학습은 실행하지 않았다.

## 3. B1 sensor6 대 실제 net force

아래 평균/P95/P99는 모든 발·환경·시간 표본을 합친 값이다. 발별 값과 force3-only 통계는 force_statistics.csv. Sensor6는 N/Nm 혼합이므로 N으로 표기하지 않는다.

'''
fr=[]
for terrain in ['flat','rough']:
 for cond,v in results[terrain+'_normal']['conditions'].items():
  s=v['sensor6_mixed_units'];n=v['net_contact_force_N'];fr.append([terrain,cond,*[f'{s[k]:.2f}' for k in ['mean','p95','p99']],*[f'{n[k]:.2f}' for k in ['mean','p95','p99']],f"{100*v['sensor6_threshold40_exceed_fraction_after_gate']:.2f}%",f"{100*v['net_threshold40_exceed_fraction_after_gate']:.2f}%"])
report+=table(['지면','조건','sensor평균','sensorP95','sensorP99','net평균N','netP95N','netP99N','sensor>40','net>40'],fr)
report+='''
Sensor는 내부 반력·중력/구동 관련 항까지 포함하는 net wrench로서 지면 외력과 상쇄된다. 실제 net contact force와 스케일이 다르다. 40 threshold를 net force에 그대로 적용한 반사실 penalty는 각 result의 same40N_same_scale_net_force_counterfactual_mean이며 평가 중 적용하지 않았다.

B1 URDF 질량57.98697kg, 네 발 균등 정적 지지142.21N/발. 현재 AS2+PiPER+Orin URDF23.32806kg, 균등 정적 지지57.21N/발. 40N은 AS2에서도 정상 정적 지지보다 낮다. 원본 B1 센서40 기준을 mass ratio만으로 환산해 net force threshold로 쓸 수 없다.

## 4. Observation ablation: 동일 checkpoint 정상 대44개0

B1에서는 current53:57와 history89:799(10×71) 각53:57만0. Gait5·privileged18·나머지 history 유지, 입력799 유지. AS2 proposed700 모델을 시험한 것이 아니다. 아래 normal→zero44. RMSE는 command에 대한 tracking error; reset은 timeout 제외 물리 실패다.

'''
br=[]
for terrain in ['flat','rough']:
 for cond,a in results[terrain+'_normal']['conditions'].items():
  b=results[terrain+'_zero44']['conditions'][cond]
  br.append([terrain,cond,f"{a['vx_mean']:.3f}→{b['vx_mean']:.3f}",f"{a['vx_rmse']:.3f}→{b['vx_rmse']:.3f}",f"{a['yaw_mean']:.3f}→{b['yaw_mean']:.3f}",f"{a['yaw_rmse']:.3f}→{b['yaw_rmse']:.3f}",f"{100*a['ee_error_mean_m']:.2f}→{100*b['ee_error_mean_m']:.2f}",f"{a['failure_resets']}→{b['failure_resets']}"])
report+=table(['지면','조건','vx m/s','vx RMSE','yaw rad/s','yaw RMSE','EE誤差cm','실패reset'],br)
report+='''
동일한 정상 물리 상태에서 두 입력을 각각 actor에 넣는 counterfactual도 매 step 수행했다. 이는 폐루프 trajectory 차이와 분리된 즉시 action 의존도다. Arm 출력6은 원본처럼 mask되어 실제 적용되지 않으므로 다리12 action RMS 차이를 주 지표로 사용한다.

'''
ar=[]
for terrain in ['flat','rough']:
 for c,v in results[terrain+'_normal']['conditions'].items():ar.append([terrain,c,f"{v['offline_action_rms_delta_history']:.4f}",f"{100*v['offline_action_rms_delta_history']/v['normal_action_rms']:.1f}%",f"{v['offline_action_rms_delta_privileged']:.4f}"])
report+=table(['지면','조건','history actor action RMS差','정상 action RMS 대비','privileged actor RMS差'],ar)
report+='''
0으로 채운 입력은 모든 발이 지속적으로 비접촉인 상태를 뜻하므로 분포 밖 입력이다. 결과는 기존 B1 정책의 접촉 입력 의존도를 보여주며, 접촉 입력 없이 처음부터 학습한 AS2 정책의 한계를 입증하지 않는다. 평지는 초기 상태가 동일하고 rough는 위 초기 차이가 있다. Reset/궤적 이후 상태까지 강제로 같게 만들지는 않았다. 8환경 단위 paired bootstrap95% 구간은 paired_checks.json; 단일 seed이며 독립 학습 seed 비교는 아니다. 속도 변화가 실제 보행을 악화시켜도40초 동안 넘어지지 않을 수 있다.

## 5. AS2 비교 및 권고: 아직 적용하지 않음

- Observation44 제거는 실제 센서 없는 배포 계약에 맞는 설계다. 744→700 /726→682가 맞다. 기존 B1 또는 AS2 checkpoint에 zero44를 넣어 그대로 배포하는 방식은 권고하지 않는다. 제거된 입력 차원으로 새 정책을 학습하고 별도 검증해야 한다.
- feet_drag/feet_air_time는 학습 중 simulator contact state로 유지할 수 있다. 실물 foot sensor가 없다는 이유만으로 reward까지 삭제할 필요는 없다. 현재 AS2는 이미 net contact_filt 기반이며 현재 source를 유지했다.
- feet_contact_forces는 과도한 충격 억제 목적을 유지할 수 있으나40N을 확정 적정치로 취급하면 안 된다. 원본B1에서는 거의 작동하지 않던 penalty가 AS2의 net-force 구현에서는 정상 지지에도 작동한다. 새 AS2 학습 보행의 stance/impact 분포로 threshold/scale을 검토해야 하며 이번에는 변경하지 않았다.
- feet_jerk는 원본에서0이므로 B1 성공 사례가 유효성 근거가 될 수 없다. AS2 수정 구현은 새로 활성화된 force-difference penalty다. 다음 AS2 실험에서 별도 reward ablation 우선 대상으로 권고하며 지금 비활성화하지 않는다.
- Shaped force/vel은 원본 부호와 stale velocity 문제를 그대로 이식하면 안 된다. 현재 AS2는 수정된 함수·기존 gait gate=False 유지. B1이 gait 입력을 사용했다는 조건 차이도 분리한다.
- Feet height, collision/stumble, base/EE/velocity reward는 발 센서 observation 제거와 독립이다. 학습 simulator 정보 사용 가능. PPO 구조·scale 유지.

기존 AS2 Reward 검증의 평지 zero-action PD에서 net force 발별 약71.7/72.0/43.7/41.4N, sensor3 norm 약2N, corrected contact-force weighted−0.0694였다. B1 학습 정책과 hardware·control·trajectory가 달라 성능 우열 비교 자료는 아니다. Rough AS2 역시 PD 자료이므로 B1 학습 보행과 직접 비교해 threshold를 확정할 수 없다.

## 파일 및 보존

- reward_contributions.csv: 모든 활성 reward의raw·scale·weighted·signed share·absolute share·nonzero 비율.
- force_statistics.csv: 지면/관측/조건/발별 sensor6·sensor3·net force 평균/P95/P99.
- observation_ablation.csv: 정상/zero44 성능과 차이. paired_checks.json: 초기 동일성·환경 단위 bootstrap.
- 각 실행 samples.npz: raw reward,6축sensor,net force,발속도,reset/timeout,EE오차,action·counterfactual 시계열.
- as2_observation_inventory.json:44개 인덱스·차원·간접 의존성. preservation_before/after.json:원본·checkpoint·AS2 소스/URDF81개 hash 보존.
- 재현: `bash reports/b1_contact_audit_20261111/run_eval.sh --terrain flat` / `--terrain rough`, 각 경우 `--ablate` 추가. 이후 `python reports/b1_contact_audit_20261111/analyze.py`.

Git status는 기존 dubious ownership 오류로 실행되지 않아 Git 설정을 변경하지 않고 파일 SHA256으로 보존 확인했다. 원본 checkpoint·원본코드·AS2코드·reward scale·PPO 파일 변경 없음. 실제 학습을 시작하지 않고 결과 보고에서 중단한다.
'''
report += 'Sensor와net force의 Pearson 상관계수·접촉 판정 비율·불일치·precision/recall은 sensor_comparison.json에 있다. Net-force 기준2N 역시 mesh 시뮬레이션의 비교 기준이며 독립 실측 접촉 정답은 아니다.\n\n## Rough 접촉 신호의 해석 한계\n\n정상 B1에서도 sensor6>1.5 대 net-force>2N 불일치가 평지0.2~1.7%, rough30.1~46.9%로 증가했다. Rough에서 안정적으로 서 있어도 발 net-force norm 평균은55.7N으로 평지142.0N보다 크게 낮다. 이 차이를 실제 지지력 감소로 단정할 수 없으며 mesh contact 보고 누락·다른 링크 접촉 등을 분리해야 한다. 본 시험은 모든 링크의 접촉력 합을 기록하지 않아 이 원인을 확정하지 않았다. Net tensor를 독립적인 접촉 정답으로 취급하지 않는다.\n\n따라서 이전 AS2 observation/reward 불일치0%는 같은 contact_filt를 공유한다는 정합성 결과이지, rough에서의 물리 접촉 판정 정확도100%를 뜻하지 않는다. 관측 제거 이후에도 rough에서 contact reward 계측의 신뢰성 검토는 남는다. 이전 AS2 PD 수치를 as2_previous_reward_comparison.json에 별도 보관했다.\n\n원본 reward 함수 중 foot_contacts_z 등은 함수가 있어도 현재 scale 목록에 없으면 실행되지 않는다. 활성 목록은 각 result.json의 scales와 CSV를 기준으로 한다. Shaped force/vel은 이 B1 run에서 gait=True로 활성이고, 현재 AS2에서는 gait=False로0이므로 두 실험의 기여도를 동일 조건으로 해석하지 않는다.\n'
(root/'REPORT_KO.md').write_text(report)
print('Generated report, CSVs, dependency inventory, paired checks, preservation proof')
