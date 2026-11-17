"""Report matched reward-only comparisons; never infer gait from survival alone."""
import json,re,os
from datetime import datetime,timezone
from pathlib import Path
from summarize_gait import summarize
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/gait_reward_proxy_v2'
CASES=['stable_proxy','progress_gain2_proxy','stable_proxy_ground','progress_gain2_proxy_yaw4','progress_gain2_proxy_ground_floor','progress_gain2_proxy_smooth_yaw4','progress_gain2_proxy_smooth_yaw4_contactmatch','progress_gain2_proxy_smooth_yaw4_footground','progress_gain2_proxy_smooth_yaw4_contactmatch_deadband','progress_gain2_proxy_smooth_yaw4_footground_deadband'];GROUPS=['stand','forward','turn','forward_turn']
RUN_NAMES={'progress_gain2_proxy_smooth_yaw4':'gait_reward_smooth_yaw4_fresh_s7_v3','progress_gain2_proxy_smooth_yaw4_contactmatch':'gait_reward_smooth_yaw4_contactmatch_fresh_s7_v3','progress_gain2_proxy_smooth_yaw4_footground':'gait_reward_smooth_yaw4_footground_fresh_s7_v4','progress_gain2_proxy_smooth_yaw4_contactmatch_deadband':'gait_reward_smooth_contactmatch_vbcdeadband_fresh_s7_v5','progress_gain2_proxy_smooth_yaw4_footground_deadband':'gait_reward_smooth_yaw4_footground_deadband_fresh_s7_v6'}


def training_logs(name):
 candidates={name,RUN_NAMES.get(name,name)}
 return sorted(p for candidate in candidates for p in (OUT/'resume_logs').glob(candidate+'_*.log') if re.fullmatch(re.escape(candidate)+r'_\d{8}_\d{6}_\d+\.log',p.name))

def publish(path,contents):
 temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 temporary.write_text(contents)
 os.replace(str(temporary),str(path))

lines=['# 동일 VBC 환경 보상 비교','', '[실험 조건과 판정 기준](PROTOCOL_KO.md) · [보상 효과 분리 비교](REWARD_EFFECT_COMPARISON_KO.md)','',
'목표: 전진0.25m/s, 회전0.6rad/s. 초기2초 이후 속도를 집계하며 넘어짐은 전체64환경×20초에서 집계한다.',
'raw foot/body contact force에는 SDK 보고 누락 가능성이 있다. 접촉 bit와 정상반력 N은 구분한다.','',
'| 보상 | 완료 update | 전진 실측(m/s) | 제자리 회전 yaw(rad/s) | 혼합 vx/yaw | 넘어짐 | 무넘어짐 환경 | Baseline |',
'|---|---:|---:|---:|---|---:|---:|---|']
all_data={}
references=[('original_vbc_port',ROOT/'experiments/gait_full_env_v1/original'),('stable_net_force',ROOT/'experiments/gait_full_env_v1/stable')]
evaluations=references+[(c,OUT/c) for c in CASES]+[(c+'_resume1',OUT/(c+'_resume1')) for c in CASES]
for case,folder in evaluations:
 for point in [100,200,300,400,600,1000]:
  f=folder/('eval_%d'%point)/'policy_evaluation.json'
  if not f.exists():continue
  d=summarize(f.parent);all_data[case+'_'+str(point)]=d
  result=folder/('result_%d.json'%point)
  if result.exists():
   record=json.loads(result.read_text());record.update(completed_updates=d['completed_updates'],local_acceptance_pass=d['local_acceptance_pass'],full_environment_baseline_pass=d.get('full_environment_baseline_pass',False),passed=d.get('full_environment_baseline_pass',False))
   publish(result,json.dumps(record,indent=2)+'\n')
  fall=sum(d[k]['fall_resets_total'] for k in GROUPS);nf=sum(d[k]['no_fall_environment_fraction'] for k in GROUPS)/4
  lines.append('| %s | %s | %.3f | %.3f | %.3f / %.3f | %d | %.1f%% | %s |'%(case+(' (fresh0)' if case in RUN_NAMES else '')+(' (branch201)' if case in ('progress_gain2_proxy_yaw4','progress_gain2_proxy_ground_floor') else ''),d.get('completed_updates','?'),d['forward']['mean_vx_m_s'],d['turn']['mean_yaw_rad_s'],d['forward_turn']['mean_vx_m_s'],d['forward_turn']['mean_yaw_rad_s'],fall,100*nf,'PASS' if d.get('full_environment_baseline_pass',False) else 'FAIL'))
lines+=['','## 재개 및 비교의 한계','',
'모든 후보는 처음에 scratch로 시작했다. 2026-10-30 07:12:17 UTC에 세 실험의 로그가 동시에 중단되었다. traceback 및 현재 cgroup OOM 기록만으로 정확한 종료 원인은 확정할 수 없다.',
'stable_proxy와 progress_gain2_proxy는 201회, stable_proxy_ground는 101회 완료 checkpoint에서 optimizer/history 상태를 복원했다. 이후 결과에는 resume1을 표시한다. 물리 환경·episode·지지 높이 cache는 초기화되므로 중단 없는 학습과 완전히 동일하지 않으며, 서로 다른 재개 시점도 비교 한계다.',
'yaw4는 progress_gain2_proxy의 동일201회 checkpoint/Adam/history/seed에서 분기하며 yaw weight만2→4로 변경한다. 환경은 재초기화한다. 높이 기준은 grid를 유지한다.',
'ground_floor 분기도 동일201회 정책에서 시작하며 마지막 접촉 높이 기준과 swing 최소 높이 보상을 묶어 비교한다. stance 목표는 대칭0, 평지의4cm 대칭 목표는 유지한다. 이것은 높이 보상 그룹의 조합 효과이며 개별항의 효과를 모두 분리했다고 주장하지 않는다.',
'기존 온라인 run 및 B1 코드는 중지·수정하지 않았다.','', '## 학습 상태','',
'갱신 시각: '+datetime.now(timezone.utc).isoformat()]
runtime={}
for case in CASES:
 for suffix in ('','_resume1'):
  name=case+suffix
  paths=training_logs(name) if suffix else [OUT/(case+'_train.log')]
  if not paths or not paths[-1].exists():
   paths=training_logs(name)
  if not paths:continue
  f=paths[-1];matches=re.findall(r'Learning iteration (\d+)/(\d+)',f.read_text())
  run=RUN_NAMES.get(case,'gait_reward_'+case+'_s7_v2')+suffix
  done=ROOT/'logs/as2piper-low'/run/'study_finished.json'
  pidfile=f.with_suffix('.pid');pid=None;alive=False
  if pidfile.exists():
   pid=int(pidfile.read_text().strip())
   try:
    os.kill(pid,0);alive=run.encode() in (Path('/proc')/str(pid)/'cmdline').read_bytes()
   except (ProcessLookupError,FileNotFoundError):pass
  record={'latest_logger_iteration':matches[-1] if matches else None,'pid':pid,'alive':alive,'finished':json.loads(done.read_text()) if done.exists() else None,'log':str(f.relative_to(ROOT))}
  runtime[name]=record
  status='완료' if done.exists() else ('실행 중' if alive else '중단 / 실행 확인 안 됨')
  lines.append('- '+name+': '+status+'; logger '+(' / '.join(matches[-1]) if matches else '초기화 중'))
lines+=['','모든 후보가 통과 전에는 안정적 보행이나 장시간 수렴으로 표시하지 않는다.']
publish(OUT/'RESULTS_KO.md','\n'.join(lines)+'\n');publish(OUT/'comparison.json',json.dumps(all_data,indent=2)+'\n')
publish(OUT/'CURRENT_STATE.json',json.dumps({'updated_utc':datetime.now(timezone.utc).isoformat(),'runs':runtime},indent=2)+'\n')
print('Measured evaluations:',len(all_data))
for name,record in runtime.items():print(name,record['latest_logger_iteration'],record['alive'])
