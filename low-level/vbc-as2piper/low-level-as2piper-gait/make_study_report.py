"""Produce a comparison table from completed paired experiments only."""
import json
from pathlib import Path
from summarize_gait import summarize

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/gait_ablation_v1'
PRESETS=['original','velocity','jerk','stop','data_stop','speed','stable']


def main():
    results={}
    for preset in PRESETS:
        folder=OUT/(preset+'_eval')
        if (folder/'policy_evaluation.json').exists():results[preset]=summarize(folder)
    rows=[]
    for preset,d in results.items():
        stand,forward,turn,mixed=[d[k] for k in ['stand','forward','turn','forward_turn']]
        rows.append('| %s | %.3f | %.3f | %.3f | %.3f | %.1f | %.3f | %d | %s |'%(
            preset,stand['mean_vx_m_s'],forward['mean_vx_m_s'],turn['mean_vx_m_s'],turn['mean_yaw_rad_s'],
            forward['contact_dominant_Hz'],forward['stance_slip_mean_m_s'],
            sum(v['fall_resets_total'] for v in [stand,forward,turn,mixed]),'PASS' if d['local_acceptance_pass'] else 'FAIL'))
    text='''# 동일 초기 정책에서 보상 원인 비교

공통 초기값: model313 + Adam/history optimizer, seed7,1024환경, 각120회 추가 학습.
공통 환경: 평지·DR off, 기존 랜덤 EE 목표/명령 및 +/-20% 관절 reset 유지.
평가: 조건별8환경×10초, 최초2초 제외한 속도·미끄러짐; 넘어짐은 초기2초를 포함한 전체 횟수.
단일 seed의 짧은 재학습 비교이며, scratch 수렴 및 거친 지면 성능을 대신하지 않는다.

| 조건 | 정지 vx | 전진 vx (목표0.25) | 회전 중 vx (목표0) | 회전 yaw (목표+0.6) | 전진 접촉 Hz | 접지 미끄러짐 m/s | 전체 넘어짐 | 명령 추종 기준 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
'''+ '\n'.join(rows)+'\n'
    (OUT/'PAIRED_COMPARISON_KO.md').write_text(text)
    (OUT/'comparison_detailed.json').write_text(json.dumps(results,indent=2))
    print(text)


if __name__=='__main__':main()
