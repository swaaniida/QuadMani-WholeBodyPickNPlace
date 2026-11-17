"""Compare unchanged 36s vy=0 motion tests; no physics or training here."""
import csv
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(directory):
    directory = Path(directory)
    meta = json.loads((directory/'metadata.json').read_text())
    z = np.load(directory/'states.npz')
    errors = z['all_env_errors']
    t = (np.arange(len(errors))+1)*.02
    commands = np.stack([np.interp(t, z['time'], z['commands'][:, 0, j]) for j in range(3)], -1)
    assert np.count_nonzero(z['commands'][:, :, 1]) == 0
    ground = np.interp(t, z['time'], z['target'][:, 0, 2]) <= .16
    valid = t > 2.
    q = z['root'][:, :, 3:7]
    x, y, zz, w = np.moveaxis(q, -1, 0)
    roll = np.arctan2(2*(w*x+y*zz), 1-2*(x*x+y*y))*180/np.pi
    pitch = np.arcsin(np.clip(2*(w*y-zz*x), -1, 1))*180/np.pi
    video_valid = z['time'] > 2.
    rows = []
    for j, it in enumerate(meta['iterations']):
        a = errors[:, j*8:(j+1)*8]
        row = {'iteration': it, 'source': str(directory),
               'ee_mean_cm': float(a[valid, :, 0].mean()*100),
               'ee_p95_cm': float(np.quantile(a[valid, :, 0], .95)*100),
               'ee_floor_mean_cm': float(a[valid & ground, :, 0].mean()*100),
               'ee_orientation_mean_deg': float(a[valid, :, 1].mean()),
               'failures': int((a[:, :, 2]*(1-a[:, :, 3])).sum()),
               'video_failures': int(z['done'][:, j].sum())}
        for k, name in [(5, 'vx'), (6, 'vy'), (7, 'yaw')]:
            row[name+'_rmse'] = float(np.sqrt(np.mean(a[valid, :, k]**2)))
        for label, mask in [('forward', commands[:,0] > .1), ('reverse', commands[:,0] < -.1),
                            ('rotation', np.abs(commands[:,2]) > .15)]:
            mask = mask & valid
            row[label+'_vx_rmse'] = float(np.sqrt(np.mean(a[mask, :, 5]**2)))
            row[label+'_yaw_rmse'] = float(np.sqrt(np.mean(a[mask, :, 7]**2)))
        for name, values in [('roll', roll[video_valid, j]), ('pitch', pitch[video_valid, j])]:
            row['video_'+name+'_abs_mean_deg'] = float(np.abs(values).mean())
            row['video_'+name+'_abs_p95_deg'] = float(np.quantile(np.abs(values), .95))
            row['video_'+name+'_abs_max_deg'] = float(np.abs(values).max())
        rows.append(row)
    return rows


rows = read(ROOT/'reports/ee_locomotion_vy0_compare_20261114/data_final')
if (OUT/'dynamic_candidates/states.npz').exists(): rows += read(OUT/'dynamic_candidates')
if (OUT/'dynamic_finalists/states.npz').exists(): rows += read(OUT/'dynamic_finalists')
with (OUT/'dynamic_metrics.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
(OUT/'dynamic_metrics.json').write_text(json.dumps({'rows': rows,
    'protocol': '36s, exact same body vx/yaw/EE schedules, vy command=0, nominal flat, no DR/push, history inference. Error metrics:8 nominal replicas,50Hz. Attitude:one video replica,25Hz,2s warmup removed. Body XY target follows actual base, ground Z fixed. OOD ground targets; no claim of trained picking skill.'}, indent=2)+'\n')
for row in sorted(rows,key=lambda r:r['iteration']):
    print(row['iteration'], {k:round(row[k],4) for k in ['vx_rmse','vy_rmse','yaw_rmse','ee_mean_cm','ee_floor_mean_cm','video_roll_abs_mean_deg','video_roll_abs_max_deg','failures']})

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font = ROOT/'reports/ee_ground_compare_20261114/viewer/vendor/fonts/NotoSansKR-Medium.ttf'
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False})
latest_by_iteration = {r['iteration']:r for r in rows}
compared = sorted([r for r in latest_by_iteration.values() if r['iteration'] >= 20000],key=lambda r:r['iteration'])
recommendation = OUT/'recommendation.json'
recommended = json.loads(recommendation.read_text())['iteration'] if recommendation.exists() else None
colors = ['#24846a' if r['iteration']==recommended else ('#d97736' if r['iteration']==45000 else '#3565a5') for r in compared]
fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
for ax,(key,title,unit) in zip(axes.flat,[('vx_rmse','전후 속도 RMSE','m/s'),('vy_rmse','횡방향 드리프트 RMSE','m/s'),('yaw_rmse','Yaw 속도 RMSE','rad/s'),('ee_floor_mean_cm','바닥 목표 EE 위치 오차','cm'),('video_roll_abs_p95_deg','Roll 절댓값 P95 (영상 대표1개)','°'),('video_roll_abs_max_deg','Roll 순간 최대 (영상 대표1개)','°')]):
    values=[r[key] for r in compared];labels=[f"{r['iteration']/1000:g}k" for r in compared]
    ax.bar(labels,values,color=colors);ax.set_title(title,loc='left');ax.set_ylabel(unit);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.tick_params(axis='x',labelrotation=40)
    ax.set_ylim(0,max(values)*1.2)
    for i,v in enumerate(values):ax.text(i,v,f'{v:.3g}',ha='center',va='bottom',fontsize=9)
fig.suptitle('36초 동적 보행 + 바닥 EE 궤적 | vy 명령0\n동일 vx·yaw·EE 시간표 · 추종 오차8replica · Roll은 영상 대표1개 · 학습 범위 밖 stress test',fontsize=14)
fig.savefig(OUT/'dynamic_comparison.png',dpi=170);plt.close(fig)
