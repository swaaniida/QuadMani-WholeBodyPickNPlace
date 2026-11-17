"""Summarize identical-protocol candidate evaluations and W&B-equivalent logs."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
FONT = ROOT / 'reports/ee_ground_compare_20261114/viewer/vendor/fonts/NotoSansKR-Medium.ttf'
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(FONT)).get_name(),
                     'axes.unicode_minus': False, 'figure.dpi': 140, 'font.size': 10})


def main():
    data = {}
    detailed = []
    for p in sorted((OUT / 'evaluations').glob('*/result.json')):
        r = json.loads(p.read_text())
        it = int(Path(r['checkpoint']).stem.split('_')[1])
        data.setdefault(it, {})[r['terrain']] = r
        for condition, metrics in r['metrics'].items():
            detailed.append({'iteration': it, 'terrain': r['terrain'], 'condition': condition, **metrics})
    complete = {it: results for it, results in data.items() if set(results) == {'flat', 'rough'}}
    aggregated = []
    for it, terrains in sorted(complete.items()):
        groups = [m for r in terrains.values() for m in r['metrics'].values()]
        v = {'iteration': it}
        v.update({k: float(np.mean([m[k] for m in groups])) for k in groups[0]})
        v['Reset/failures_sum'] = sum(m['Reset/failures'] for m in groups)
        v['forward_vx_mean'] = np.mean([r['metrics']['forward']['Tracking/vx_mean_m_s'] for r in terrains.values()])
        v['forward_vx_rmse'] = np.mean([r['metrics']['forward']['Tracking/vx_rmse_m_s'] for r in terrains.values()])
        v['forward_turn_vx_rmse'] = np.mean([r['metrics']['forward_turn']['Tracking/vx_rmse_m_s'] for r in terrains.values()])
        v['forward_only_yaw_rmse'] = np.mean([r['metrics']['forward']['Tracking/yaw_rmse_rad_s'] for r in terrains.values()])
        v['turn_only_yaw_rmse'] = np.mean([r['metrics']['turn']['Tracking/yaw_rmse_rad_s'] for r in terrains.values()])
        v['turn_only_vy_rmse'] = np.mean([r['metrics']['turn']['Tracking/vy_rmse_m_s'] for r in terrains.values()])
        v['Pose/roll_abs_max_deg_max'] = max(m['Pose/roll_abs_max_deg'] for m in groups)
        v['Pose/pitch_abs_max_deg_max'] = max(m['Pose/pitch_abs_max_deg'] for m in groups)
        aggregated.append(v)
    if not aggregated:
        print('No complete two-terrain candidates yet')
        return
    for name, rows in [('all_conditions.csv', detailed), ('candidate_metrics.csv', aggregated)]:
        with (OUT / name).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / 'evaluation_summary.json').write_text(json.dumps({'evaluations': data, 'summary': aggregated,
        'averaging': 'Equal mean across 8 terrain/command conditions; means of conditional RMSE and quantiles, not pooled quantiles. Failure counts summed. Only complete flat+rough candidates are ranked.'}, indent=2) + '\n')

    base = next((r for r in aggregated if r['iteration'] == 20000), None)
    if base:
        groups = {
            'tracking': {'Tracking/vx_rmse_m_s': .5, 'Tracking/vy_rmse_m_s': .2, 'Tracking/yaw_rmse_rad_s': .3},
            'posture': {'Pose/roll_abs_mean_deg': .4, 'Pose/roll_abs_p95_deg': .2, 'RewardRaw/leg/base_height': .2, 'RewardRaw/leg/ang_vel_xy': .2},
            'ee': {'EE/position_mean_m': .5, 'EE/position_p95_m': .5},
            'load': {'RewardRaw/leg/torques': .25, 'RewardRaw/leg/work': .25, 'Force/norm_p99_N': .25, 'RewardRaw/leg/collision': .25}}
        priorities = {'balanced': {'tracking': .5, 'posture': .25, 'ee': .15, 'load': .1},
                      'tracking_first': {'tracking': .7, 'posture': .1, 'ee': .15, 'load': .05},
                      'posture_first': {'tracking': .3, 'posture': .45, 'ee': .15, 'load': .1}}
        ranks = []
        for r in aggregated:
            score = {'iteration': r['iteration'], 'failures': r['Reset/failures_sum']}
            score.update({g: sum(w * r[k] / base[k] for k, w in metrics.items()) for g, metrics in groups.items()})
            score.update({name: sum(w * score[g] for g, w in weights.items()) for name, weights in priorities.items()})
            ranks.append(score)
        (OUT / 'selection_sensitivity.json').write_text(json.dumps({'reference': 20000, 'groups': groups,
            'priorities': priorities, 'scores': ranks,
            'note': 'Transparent heuristic, lower is better, reference20k=1; not a validated deployment safety index. Reset is a separate exclusion criterion. Pitch is reported, not scored as an unrequested flat-body objective.'}, indent=2)+'\n')
        print('WEIGHT SENSITIVITY')
        for priority in priorities:
            print(priority, [(r['iteration'], round(r[priority], 4)) for r in sorted(ranks, key=lambda r:r[priority])])

    columns = [('Tracking/vx_rmse_m_s', '전후 속도 RMSE', 1, 'm/s'),
               ('Tracking/vy_rmse_m_s', '횡방향 드리프트 RMSE', 1, 'm/s'),
               ('Tracking/yaw_rmse_rad_s', 'Yaw 속도 RMSE', 1, 'rad/s'),
               ('Pose/roll_abs_mean_deg', 'Roll 절댓값 평균', 1, '°'),
               ('Pose/roll_abs_p95_deg', 'Roll 절댓값 P95', 1, '°'),
               ('Pose/pitch_abs_p95_deg', 'Pitch 절댓값 P95', 1, '°'),
               ('EE/position_mean_m', 'EE 위치 오차 평균', 100, 'cm'),
               ('RewardRaw/leg/base_height', '몸통 높이 목표 오차', 100, 'cm'),
               ('Force/norm_p99_N', '발 접촉력 P99', 1, 'N'),
               ('RewardRaw/leg/torques', '다리 명령 토크 제곱합', 1, 'Nm²'),
               ('forward_vx_mean', '전진 실제 속도 (명령 0.4)', 1, 'm/s'),
               ('Reset/failures_sum', '실패 Reset 합계', 1, '회')]
    fig, axes = plt.subplots(4, 3, figsize=(14, 13), constrained_layout=True)
    recommended_path = OUT/'recommendation.json'
    recommended = json.loads(recommended_path.read_text())['iteration'] if recommended_path.exists() else None
    colors = ['#24846a' if r['iteration'] == recommended else ('#3565a5' if r['iteration'] != 45000 else '#d97736') for r in aggregated]
    labels = [f"{r['iteration']/1000:g}k" for r in aggregated]
    for ax, (key, title, scale, unit) in zip(axes.flat, columns):
        values = [r[key] * scale for r in aggregated]
        ax.bar(labels, values, color=colors)
        ax.set_title(title, loc='left', fontweight='bold'); ax.set_ylabel(unit)
        ax.tick_params(axis='x', labelrotation=45); ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
        for i, v in enumerate(values):
            label = f'{v:.0f}' if key == 'RewardRaw/leg/torques' else f'{v:.3g}'
            ax.text(i, v, label, ha='center', va='bottom', fontsize=8)
        if max(values)>0: ax.set_ylim(0, max(values)*1.2)
        else: ax.set_ylim(0, 1)
        if key == 'forward_vx_mean':
            ax.axhline(.4, color='#b24b3a', ls='--', lw=1.2, label='명령 0.4m/s')
            ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('AS2 + Head PiPER + Orin | 체크포인트 균형 비교\n평지·Rough × 정지·전진·회전·복합 / 각 40초 / DR·Push OFF / History 추론', fontsize=15)
    fig.savefig(OUT/'candidate_comparison.png', dpi=170); plt.close(fig)

    z = np.load(OUT/'training_metrics.npz'); a=z['values']; names=z['columns'].tolist()
    metrics = [('Tracking/vx_rmse_m_s', '전후 속도 오차 (m/s)', 1),
               ('Tracking/yaw_rmse_rad_s', 'Yaw 속도 오차 (rad/s)', 1),
               ('RewardRaw/leg/roll', 'Roll 절댓값 평균 (°)', 180/np.pi),
               ('EE/position_mean_m', 'EE 위치 오차 (cm)', 100),
               ('Reset/failures_per_1000_env_steps', '실패 Reset / 100만 env-step', 1000),
               ('Force/norm_p99_N', '학습 구간별 발 접촉력 P99 (N)', 1)]
    fig, axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for ax,(key,title,factor) in zip(axes.flat,metrics):
        block=200
        x=a[:,0].reshape(-1,block).mean(1)/1000
        y=a[:,names.index(key)].reshape(-1,block).mean(1)*factor
        ax.plot(x,y,color='#3565a5',lw=1.5);ax.set_title(title,loc='left');ax.set_xlabel('학습 Iteration (천 회)')
        ax.axvline(10,color='#b24b3a',ls='--',alpha=.65,label='20 ms 지연 시작')
        ax.grid(alpha=.15)
        if key.startswith('Reset/'):ax.set_yscale('symlog',linthresh=5)
        if key.startswith('Force/'):ax.set_ylim(270,430)
    axes.flat[0].legend()
    fig.suptitle('전체 45,000회 학습 추세 | W&B에 전송한 동일 원자료\n200회 구간 평균 · DR/Push 및 탐색 행동 포함 · 배포 성능 순위와 구분',fontsize=14)
    fig.savefig(OUT/'training_trends.png',dpi=170);plt.close(fig)
    print('COMPLETE CANDIDATES', sorted(complete))
    for r in aggregated:
        print(r['iteration'], 'vx',round(r['Tracking/vx_rmse_m_s'],5),'vy',round(r['Tracking/vy_rmse_m_s'],5),
              'yaw',round(r['Tracking/yaw_rmse_rad_s'],5),'roll',round(r['Pose/roll_abs_mean_deg'],4),
              'EEcm',round(100*r['EE/position_mean_m'],3),'failure',r['Reset/failures_sum'])


if __name__ == '__main__':
    main()
