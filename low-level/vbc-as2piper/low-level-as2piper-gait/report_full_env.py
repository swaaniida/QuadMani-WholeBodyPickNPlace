"""Summarize only measured matched full-environment evaluations."""
import json,re
from summarize_gait import summarize
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/gait_full_env_v1'
GROUPS=['stand','forward','turn','forward_turn']

def main():
    rows=[];all_results={}
    for preset in ['original','speed','stable','progress','progress_gain2','gaussian_gain2']:
        for point in [100,200,400,600]:
            file=OUT/preset/('eval_%d'%point)/'gait_summary.json'
            if not file.exists():continue
            d=summarize(file.parent);all_results[preset+'_'+str(point)]=d
            fall=sum(d[k]['fall_resets_total'] for k in GROUPS)
            mae=sum(d[k]['vx_abs_error_m_s'] for k in GROUPS)/4
            yaw=sum(d[k]['yaw_abs_error_rad_s'] for k in GROUPS)/4
            no_fall=sum(d[k]['no_fall_environment_fraction'] for k in GROUPS)/4
            rows.append('| %s | %d | %s | %.3f | %.3f | %d | %.1f%% | %s | %s |'%(preset,point,d.get('completed_updates','?'),mae,yaw,fall,100*no_fall,'PASS' if d.get('full_environment_baseline_pass',False) else 'FAIL','PASS' if d.get('strict_no_fall_pass',False) else 'FAIL'))
    doc=['# VBC 환경 보상 비교 측정 결과','',
      '공통 환경과 비교 범위는 [PROTOCOL_KO.md](PROTOCOL_KO.md)를 따른다. 평지 결과를 이 표에 섞지 않는다.',
      '속도 오차는 4개 동작의 MAE 평균이며, 전체 넘어짐은 64환경×20초에서 집계한다.',
      '넘어짐 없는 환경 비율은 전체 20초 동안 한 번도 넘어지지 않은 환경의 비율이다.',
      'episode return은 목적 함수가 다르므로 군 간 성능 비교에 사용하지 않는다.','',
      '| 보상 | checkpoint 파일 번호 | 실제 완료 update | 전진속도 MAE(m/s) | yaw MAE(rad/s) | 전체 넘어짐 | 넘어짐 없는 환경 | Full-env baseline | 전체 무넘어짐 |',
      '|---|---:|---:|---:|---:|---:|---:|---|---|' ]+rows
    if not rows:doc+=['| 평가 대기 | — | — | — | — | — | — | — | — |']
    doc+=['','## 100회 pilot: 32환경 ×20초','', '| 보상 | 전진 실측(m/s) | 회전 실측(rad/s) | 전체 넘어짐 |','|---|---:|---:|---:|']
    for preset in ['original','speed','stable','progress','progress_gain2','gaussian_gain2']:
        f=OUT/(preset+'_early')/'eval_100/gait_summary.json'
        if f.exists():
            d=summarize(f.parent);all_results[preset+'_pilot100']=d
            doc.append('| %s | %.3f | %.3f | %d |'%(preset,d['forward']['mean_vx_m_s'],d['turn']['mean_yaw_rad_s'],sum(d[k]['fall_resets_total'] for k in GROUPS)))
    doc+=['','## 학습 진행 상태','', '| 보상 | 최근 iteration | 최근 episode 길이(steps) |','|---|---:|---:|']
    for preset in ['original','speed','stable','progress','progress_gain2','gaussian_gain2']:
        path=OUT/(preset+'_train.log')
        if not path.exists():continue
        text=path.read_text();it=re.findall(r'Learning iteration (\d+)/(\d+)',text);length=re.findall(r'Mean episode length: ([\d.]+)',text)
        doc.append('| %s | %s | %s |'%(preset,'/'.join(it[-1]) if it else '준비 중',length[-1] if length else '—'))
    doc+=['','학습 중 episode 길이는 보행 성능의 대체 지표가 아니다. 지면 원점 기준 높이로 판단한 upright 비율은 rough에서 근사이며, 몸통 기울기 비율 및 실제 reset 원인과 함께 해석한다.',
      '단일 학습 seed와 제한된 평가 시간의 결과이며 실물 안전성·장시간 수렴을 증명하지 않는다.']
    (OUT/'RESULTS_KO.md').write_text('\n'.join(doc)+'\n')
    (OUT/'comparison.json').write_text(json.dumps(all_results,indent=2)+'\n')
    print('Summarized',len(all_results),'measured evaluations')

if __name__=='__main__':main()
