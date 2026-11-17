from pathlib import Path
import json,hashlib,ast,os,grp
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'experiments/contact_force_ablation'
results={}
for p in sorted((base/'validation').glob('*/result.json')):
 x=json.loads(p.read_text());results[x['condition']]=(p,x)
assert set(results)==set('ABC')
assert len({x['initial_model_sha256'] for _,x in results.values()})==1
assert len({x['dr_parameters_sha256'] for _,x in results.values()})==1
assert len({json.dumps(x['source_sha256'],sort_keys=True) for _,x in results.values()})==1
for _,x in results.values():
 assert x['initial_model_sha256']==x['final_model_sha256'] and x['optimizer_steps']==0 and x['reward_sum_verified']
 assert x['env']['rewards']['feet_height_target']==.18 and x['env']['rewards']['scales']['feet_height']==1
 for k in ['tracking_contacts_shaped_force','tracking_contacts_shaped_vel']:
  assert x['metrics']['stand']['RewardWeighted/leg/'+k]==0
 assert x['env']['env']['num_observations']==755 and x['env']['env']['observe_gait_commands']
 assert not x['env']['env']['observe_foot_contacts']
protected=json.loads((base/'preservation_before.json').read_text())
changed=[p for p,h in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
assert not changed
proof={'existing_files_checked':len(protected),'changed_existing_files':changed,'same_initial_model_sha256':results['A'][1]['initial_model_sha256'],'same_dr_sha256':results['A'][1]['dr_parameters_sha256'],'reward_sum_verified_all_cases':True,'optimizer_steps':0,'training_started':False}
(base/'verification_summary.json').write_text(json.dumps(proof,indent=2)+'\n')
rows=[]
for c in 'ABC':
 _,x=results[c];f=x['force_probes'];rows.append(f"| {c} | {x['env']['rewards']['max_contact_force']} | {x['env']['rewards']['scales']['feet_contact_forces']} | {f['150.0']['weighted']:.2f} | {f['250.0']['weighted']:.2f} |")
initsha=proof['same_initial_model_sha256'];drsha=proof['same_dr_sha256']
report='''# Contact Force Reward Ablation 설정·검증

## 구성 및 불변 조건

기존코드를수정하지않고 force_ablation/ 모듈과 run_force_ablation.sh만추가했다. A/B/C를독립태스크·run·출력디렉터리로분리했다. 세조건의설정차이는 contact force threshold와feet_contact_forces scale뿐임을전체config비교로확인했다. 조건C의threshold40은 raw진단기준일뿐실제보상에는무관하다.

공통: feet_height_target0.18m·scale+1, gait clock5 활성·2Hz, contact observation44 제거,755/737 입력계약, 정지contact shaping walking_mask, shaping scale−2,원래PPO·DR·기타reward. 훈련규모도기존6144환경·45000iteration·24step rollout을그대로사용한다. 원본실행스크립트와기존태스크는보존했다.

| 조건 | thresholdN | scale | 네발각150N weighted | 네발각250N weighted |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

Probe는episode age>2s인동일상태이며값은/100이전이다. A/B는 실제 reward 목록에포함되고C는 scale0으로목록에서제외된다. C의raw force값은side effect없는함수를진단용으로만계산하여기록한다. 기여도와학습반환reward에는0으로반영한다.

## 동일 초기화

Seed 기본1. seed_1/initial_policy.pt는학습전공통가중치이며생성시독점쓰기,기존파일덮어쓰기금지. 각actual runner에strict load하고모델hash·빈두Adam state·iteration0·PPO counter0을검사한다. 세조건의initial/final model hash는모두동일했다.

Model SHA256: `'''+initsha+'''`

실제DR mass/COM/friction/motor strength tensor SHA256도모두동일했다: `'''+drsha+'''`.

따라서초기파라미터나DR설정차이가아닌force reward차이로학습비교할수있다. 단동일seed라도GPU PhysX의개별물리궤적bitwise재현을보장하지않는다. 더강한통계결론에는같은seed묶음의A/B/C를반복해야한다.

## 검증 실행 범위

- prepare: 공통초기가중치와전체조건config 저장.
- A/B/C 각각 actual runner/model/storage 생성, 같은weight load,16환경×120step(2.4초) PD/IK rollout. 검증용terrain만4x4·border5m으로축소했으며DR는원래활성범위를유지했다. 검증override는세조건동일하고훈련config에저장하지않는다.
- 실제호출raw×scale합이native leg/arm reward와일치함을매step검사했다. Stateful reward는재호출하지않는다.
- 모든조건의정지shaping force/vel=0,이동명령에서는음수penalty. Force source는net contact force, EE source는실제TCP이다.
- 미학습 model_0 fixture로3초의checkpoint-load/evaluate경로도실행하여정지·전진·회전·복합각8환경의metric JSON/NPZ생성을확인했다. 이결과는미학습정책이므로성능평가결과로사용하지않는다.
- actual train mode와learn()은한번도실행하지않았다. Optimizer step0,가중치불변. 장시간학습시작전설정/계측검증이다.

## 평가·기록 지표

Training중native24step logging에속도RMSE(vx/vy/yaw)·실제속도·EE position mean/P95·timeout/실패reset·1000env-step당실패·발net force norm평균/P95/P99(pooled및발별)·모든활성reward raw/weighted/signed share/absolute share를추가했다. 원본loss/episode 로그는유지한다. ablation_metrics.jsonl과동일W&B step에저장하며project는as2piper-low다.

weighted는/100이전,abs share=E|weighted term|/ΣE|weighted term|. C의force contribution은항상0. Force통계는비접촉0도포함하는norm이고Fz/6축sensor norm과다르다. Rough net contact의보고누락가능성때문에독립실측접촉정답으로보지않는다.

후속평가는같은seed·terrain·duration·iteration으로각조건checkpoint를로드한다. 기본flat/rough각32환경,4명령조건,40초중초기2초제외,고정중앙EE궤적,평가에서만DR/push off. 학습manifest의condition/source hash가다르면거부한다. 자세한명령은README_KO.md.

## 파일·보존

- configs.py: 세조건/태스크, runtime.py: 계측환경·원본runner logging확장, run.py: prepare/validate/train/evaluate.
- experiments/contact_force_ablation/seed_1/: 공통미학습weight와전체config.
- experiments/contact_force_ablation/validation/*/result.json,samples.npz: 조건별검증결과.
- experiments/contact_force_ablation/evaluations/*/: 평가경로검사결과(미학습fixture).
- verification_summary.json,preservation_before.json: 기존B1 checkpoint·현재No-contact production code·PPO·URDF 보존검사.

기존파일hash보존을확인했다. Git status는기존dubious ownership오류라설정을변경하지않고hash로확인했다. 사용자검토전장시간학습은시작하지않는다.
'''
(ROOT/'force_ablation/VALIDATION_KO.md').write_text(report)
p=ROOT/'CHANGES_KO.md';s=p.read_text();heading='## Contact Force Reward Ablation 준비'
if heading not in s:
 s+='''\n## Contact Force Reward Ablation 준비\n\n- 기존production코드수정없이force_ablation모듈·run_force_ablation.sh추가. A40N/B200N/Cforce scale0을별도태스크/run으로분리.\n- Height0.18·scale+1, shaping walking_mask·scale−2, DR/Gait/PPO/755·737입력유지.\n- Seed1공통미학습weight strict load,초기/최종모델hash및DR tensor hash가세조건동일. 실제reward합산·정지shaping0·150/250N probe검증.\n- 속도/EE/reset/force분포/reward기여도계측및checkpoint평가경로준비. Native PPO/DAgger유지.\n- Prepare·소규모PD검증·미학습fixture평가만실행. 실제학습/optimizer step0. 상세force_ablation/VALIDATION_KO.md.\n''';p.write_text(s)
for p in (ROOT/'force_ablation').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
gid=grp.getgrnam('vbc').gr_gid
for p in [ROOT/'force_ablation',base]+list((ROOT/'force_ablation').rglob('*'))+list(base.rglob('*'))+[ROOT/'run_force_ablation.sh',ROOT/'CHANGES_KO.md']:
 if p.is_symlink():continue
 p.chmod(0o2775 if p.is_dir() else (0o775 if p.suffix=='.sh' else 0o664))
 if p.stat().st_uid==os.getuid() and p.stat().st_gid!=gid:os.chown(str(p),-1,gid)
print('All checks passed:',proof)
