"""Write a Korean decision report from recorded candidate evaluation results."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--recommend', type=int, required=True)
args = parser.parse_args()
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
payload = json.loads((OUT/'evaluation_summary.json').read_text())
rows = {r['iteration']: r for r in payload['summary']}
dynamic = {r['iteration']: r for r in json.loads((OUT/'dynamic_metrics.json').read_text())['rows']}
assert len(rows) == 10
selected = args.recommend
assert selected in rows and selected in dynamic
reference = rows[30000]
chosen = rows[selected]
checkpoint = ROOT/'low-level-as2piper-head-orin-nocontact/experiments/contact_force_ablation/runs/as2piper_headpiper_orin_200N_0.18m'/('model_%d.pt'%selected)
recommendation = {'iteration': selected, 'checkpoint': str(checkpoint),
    'basis': 'Full45k local W&B-equivalent diagnostics plus10 checkpoint/8 condition deterministic history-inference evaluations and36s vy0 dynamic checks.',
    'scope': 'Best practical candidate among evaluated checkpoints, not an exhaustive physics rollout of all226 models or real-world safety certification.',
    'training_changes': False, 'optimizer_updates': 0}
(OUT/'recommendation.json').write_text(json.dumps(recommendation, indent=2)+'\n')


def table(headers, values):
    return '| ' + ' | '.join(headers) + ' |\n|'+'|'.join(['---']*len(headers))+'|\n' + ''.join('| '+' | '.join(map(str,r))+' |\n' for r in values)


def percentage(key, it=30000):
    return (chosen[key]/rows[it][key]-1)*100


s = f'''# 체크포인트 선정 결과

## 권고

**기본 사용 후보: {selected:,} iteration (`model_{selected}.pt`).**

저장된 226개 체크포인트에 대응하는 전체 45,000회 학습 원자료를 조사하고, 10개 후보를 같은 조건으로 실제 추론 평가했다. 최종 선택은 W&B reward/loss의 최고점이 아니라 속도·자세·EE·접촉·관절 비용의 절충에 근거한다. 아래 표의 낮은 비용이 실제 하드웨어의 안전 인증을 의미하지는 않는다.

체크포인트: `{checkpoint}`

- 선정 후보의 30,000 대비 8조건 평균 vx RMSE 변화: **{percentage('Tracking/vx_rmse_m_s'):+.1f}%**, vy RMSE **{percentage('Tracking/vy_rmse_m_s'):+.1f}%**, yaw RMSE **{percentage('Tracking/yaw_rmse_rad_s'):+.1f}%**.
- Roll 절댓값 평균 **{percentage('Pose/roll_abs_mean_deg'):+.1f}%**, P95 **{percentage('Pose/roll_abs_p95_deg'):+.1f}%**. EE 위치 평균 **{percentage('EE/position_mean_m'):+.1f}%**.
- 어느 후보도 과속 문제를 근본적으로 해결하지는 못했다. 0.4m/s 전진 명령에서 선정 후보의 평지/rough 실제 vx 평균은 **{chosen['forward_vx_mean']:.3f}m/s**다. 현재 reward는 목표 이상 속도에서 포화하므로 checkpoint 선택만으로 정밀한 속도 추종을 보장할 수 없다.

### 사용 목적별 선택

- **보행 + 바닥 EE 동작의 기본 후보: 28,000.** 36초 동적 시험에서 vx RMSE는20k와 동급, vy RMSE는 비교 후보 중 가장 작고 EE 평균은45k와 동급이다. 대표 영상 replica의 순간 roll 최대도45k보다 작다. 같은 시험의30k 대비 EE 평균 약34%, vy RMSE 약28% 개선이 확인됐다. 20k 대비 약0.001m/s인 vx 차이를 의미 있는 개선으로 주장하지 않는다.
- **Yaw 추종·순간 흔들림 억제 우선: 20,000.** 28k보다 동적 yaw 오차와 roll P95/최대가 작다. 대신 바닥 EE 평균 오차는 더 크다.
- **EE가 학습 범위에 머무는 일반 보행: 40,000 또는40,600.** 40k는 고정 명령 속도,40.6k는 roll·접촉력에 강점이 있다. 바닥 EE와 동시 이동 시 vy/yaw 및 순간roll 악화가 확인되어 전신 조작의 기본 후보로 선정하지 않았다.
- **30,000을 기본으로 고르지 않은 이유:** nominal yaw 성능은 좋지만 동적 시험에서는28k보다 EE·횡방향 드리프트가 크다. 반면28k의 yaw 추종과 roll 평균은30k보다 나쁘므로 이를 숨긴 종합 우승 주장은 하지 않는다.
- **45,000을 기본으로 고르지 않은 이유:** EE·평균roll 장점이 있으나 동적 vx 오차와 순간roll이28k보다 크다. Training reward/loss가 좋아졌다는 이유로 최종 모델을 우선하지 않는다.

선택은 현재 목적에 대한 절충 판단이다. 바닥 목표는 학습 범위 밖이므로 모든 실제 manipulation 조건에 대한 일반화를 입증한 결과는 아니다. Yaw가 매우 중요한 작업이라면 기본 후보를20k로 바꿔 선택하는 편이 타당하다.

## 평가 범위와 재현 조건

- W&B에 `wandb.log()`로 전송한 동일 원자료: 원래 run의 `ablation_metrics.jsonl`, 45,000행. W&B 서버 API를 새로 조회한 결과가 아니다. Native loss·reward·episode 지표는 원래 학습 로그 전체 45,000회에서 추출했다.
- 저장 checkpoint 전수 목록 226개, 최근 1,000회 구간 통계 226×전체/History/Privileged 세 경로를 CSV로 보존했다. 5,000회 이전 전진 전용, 10,000회 이전 무지연, 이후 20ms action delay를 구분했다. 초기 checkpoint의 다른 조건에서 낮았던 training RMSE를 그대로 후기 모델보다 우수하다고 해석하지 않는다.
- 실제 추가 평가: **18,000 / 20,000 / 24,000 / 28,000 / 30,000 / 34,000 / 40,000 / 40,600 / 43,400 / 45,000**.
- 각 checkpoint: 평지·rough × 정지·전진(0.4m/s)·회전(0.6rad/s)·전진+회전, 조건당 8개 nominal replica, 40초 중 초기 2초 제외. Seed1, 같은 원본 지형 생성기·제어·reward·termination, DR/Push OFF, History 기반 deterministic 추론, optimizer update 0회.
- 원래 `force_ablation.run.evaluate()`를 그대로 실행하고, 평가 전용 hook으로 reset 전 실제 root quaternion의 roll/pitch와 EE orientation 오차만 추가 기록했다. 기존 학습 파일은 변경하지 않았다.
- 후보별 140개 지표×8조건, 총 11,200개 수치를 `all_conditions.csv`에 보존했다. Bodycontact termination은 기존 비활성 상태를 유지했다. 따라서 reset0을 모든 비정상 접촉이 없다는 뜻으로 해석하지 않는다.
- 표의 RMSE/P95/P99는 **8조건별 통계의 동등 평균**이며 전체 샘플 pooled 통계가 아니다. 최대값은 별도 표기한 경우 조건 전체 최대다. 같은 seed의 8개 replica는 독립된 8개 학습 seed가 아니다. GPU PhysX 반복 실행의 소폭 차이는 허용하며 20k/30k/45k도 이번에 같은 hook으로 재평가했다. 동적 평가의 첫 reset 후 실제 q/root 상태도 replica 간 bitwise 동일하지 않으며 원본 `initial_states.npz`에 보존했다.

## 속도·EE 비교

''' + table(['Iteration','vx RMSE (m/s)','vy RMSE (m/s)','yaw RMSE (rad/s)','전진 실제 vx (명령0.4)','EE 평균 (cm)','EE P95 (cm)','실패 reset'], [
    [it, f"{r['Tracking/vx_rmse_m_s']:.4f}",f"{r['Tracking/vy_rmse_m_s']:.4f}",f"{r['Tracking/yaw_rmse_rad_s']:.4f}",f"{r['forward_vx_mean']:.4f}",f"{r['EE/position_mean_m']*100:.3f}",f"{r['EE/position_p95_m']*100:.3f}",int(r['Reset/failures_sum'])]
    for it,r in sorted(rows.items())])
s += '\n## Roll·Pitch·높이\n\n' + table(['Iteration','Roll 평균 절댓값(°)','Roll P95(°)','Roll 최대(°)','Pitch P95(°)','Pitch 최대(°)','Base높이 MAE(cm)'],[
    [it,f"{r['Pose/roll_abs_mean_deg']:.3f}",f"{r['Pose/roll_abs_p95_deg']:.3f}",f"{r['Pose/roll_abs_max_deg_max']:.3f}",f"{r['Pose/pitch_abs_p95_deg']:.3f}",f"{r['Pose/pitch_abs_max_deg_max']:.3f}",f"{r['RewardRaw/leg/base_height']*100:.3f}"] for it,r in sorted(rows.items())])
s += '''
Roll은 부호 없는 절댓값이므로 한쪽으로 기울어진 방향을 뜻하지 않는다. Pitch는 EE 접근에 필요한 몸통 기울기일 수 있어 무조건0이 최적이라고 가정하지 않았으며, 별도 수치로 비교했다. 순간 최대값은 이상 동작 여부를 보는 보조 지표이며 평균/P95와 함께 판단한다.

## 접촉·다리 구동 비용

''' + table(['Iteration','발 힘 P95(N)','발 힘 P99(N)','다리 τ² 합(Nm²)','Work proxy(W)','Hip편차²','관절가속도²','Collision raw'],[
    [it,f"{r['Force/norm_p95_N']:.1f}",f"{r['Force/norm_p99_N']:.1f}",f"{r['RewardRaw/leg/torques']:.1f}",f"{r['RewardRaw/leg/work']:.2f}",f"{r['RewardRaw/leg/hip_pos']:.5f}",f"{r['RewardRaw/leg/dof_acc']:.0f}",f"{r['RewardRaw/leg/collision']:.5f}"] for it,r in sorted(rows.items())])
s += '''
- 토크 비용은 `env.torques`의 명령값이며 실제 PiPER position-drive 토크를 포함하지 않는다. Work는 `abs(sum(τ*dq))` proxy이며 전체 소비 전력이 아니다.
- 발 힘은 비접촉0을 포함한 시뮬레이터 net contact force norm. Rough의 보고 누락/과소추정 가능성이 있어 rough의 더 작은 힘을 실제 하드웨어의 더 안전한 접촉으로 해석하지 않는다. Threshold200N은 강제 상한이 아니다.
- 활성 reward 전체의 raw·weighted·absolute/signed share와 Leg/Arm 합계를 조건별 CSV에 보존했다. Reward 합계나 walking_dof 점수가 높다는 이유로 후보를 선정하지 않았다.

## 동적 EE·보행 교차 검증

현재 영상과 동일한 36초 궤적, **vy 명령 전 구간0**, vx ±0.35m/s 이내·yaw ±0.7rad/s 이내, 바닥부터 위쪽까지 움직이는 EE 목표다. XY 목표는 실제 base 위치/yaw를 따라가고 Z는 지면 기준이다. 명령 숫자·시간표·상대 EE 궤적은 기존 vy0 실험과 동일하다.

새 후보 24k/28k/34k/40k/40.6k/43.4k를 추가 실행하고 기존20k/30k/45k 기록과 비교했다. 20k/30k는 새 후보와 함께 반복해 재현성을 확인했으며 표에는 최근 반복 결과를 사용했다. 반복 전체 원자료도 JSON/CSV에 보존했다. 오차는 8개 replica·50Hz, 자세는 동일한 영상 대표 replica 1개·25Hz이다. Nominal 고정 EE 평가와 별도 태스크이므로 두 오차를 섞어 평균내지 않았다.

''' + table(['Iteration','vx RMSE','vy RMSE','yaw RMSE','EE 평균(cm)','바닥 EE 평균(cm)','Roll평균(°)','Roll P95(°)','Roll최대(°)','실패 reset(8rep)'],[
    [it,f"{r['vx_rmse']:.4f}",f"{r['vy_rmse']:.4f}",f"{r['yaw_rmse']:.4f}",f"{r['ee_mean_cm']:.3f}",f"{r['ee_floor_mean_cm']:.3f}",f"{r['video_roll_abs_mean_deg']:.3f}",f"{r['video_roll_abs_p95_deg']:.3f}",f"{r['video_roll_abs_max_deg']:.3f}",r['failures']] for it,r in sorted(dynamic.items()) if it>=20000])
s += '''
바닥 목표는 기존 좁은 학습 EE 범위 밖의 stress test다. 성공한 시뮬레이션 추종을 바닥 집기 학습 완료나 실제 grasp 성공으로 해석하지 않는다. 팔은 기존 IK/position drive를 유지하므로 EE 오차는 학습된 다리 제어와 기존 팔 제어를 합친 파이프라인의 결과다.

## 일반 보행 지표의 선정 기준 민감도

`selection_sensitivity.json`은20k를1로 정규화한 공개된 비교 heuristic이다. Tracking 그룹은 vx/vy/yaw, 자세 그룹은 roll 평균/P95·높이·몸통 angular-xy, EE 그룹은 위치 평균/P95, 부하 그룹은 다리 토크·work·접촉력P99·collision을 사용한다. 실패 reset은 별도 제외 조건이다. Pitch0을 임의의 새로운 목표로 추가하지 않았다.

- 균형형: Tracking50%, 자세25%, EE15%, 부하10%.
- 속도 우선: Tracking70%, 자세10%, EE15%, 부하5%.
- 자세 우선: Tracking30%, 자세45%, EE15%, 부하10%.

이는 일반 보행의 지표만 점수화한 heuristic이며 검증된 안전성 지수가 아니다. 이 점수에서40k/40.6k가 유리해도 동적 바닥 EE 테스트의 vy/yaw·순간roll이 악화되어 기본 후보를 바꿀 수 있다. 가중치에 따른 순위 변화와 동적 stress 결과를 함께 보고 사용 목적에 맞게 선택한다. 모든 226개 모델을 실제 physics rollout한 것은 아니므로 미평가 checkpoint보다 전역적으로 최적이라는 주장도 하지 않는다.

## W&B 지표 해석

- 학습 rollout은 20회 중19회 Privileged,1회 History 경로다. 전체 평균은 Privileged 경로가95%이므로 배포 History 추론의 순위와 같다고 볼 수 없다. 두 경로를 분리한 `training_encoding_windows.csv`와 전 checkpoint 통계를 보존했다.
- 학습 지표는 stochastic action·DR·push·무작위 명령을 포함한다. 이번 평가는 deterministic History 추론·고정 명령·nominal physics이므로 절댓값을 직접 비교하지 않는다.
- 후반부 reward/loss/History supervision은 개선되지만, 전진 reward 포화 때문에 deterministic 과속이 남을 수 있다. Value/PPO loss 감소는 속도 정확성의 지표가 아니다.
- DAgger 회차에는 PPO loss, PPO 회차에는 History loss가 직전 회차의 값으로 표시될 수 있다. 이 값을 같은 회차의 동시 업데이트로 취급하지 않는다.
- Native loss·reward·episode length·noise 전체 시계열은 `native_learning_metrics.csv`, 학습 물리 추세는 `training_metrics.npz`에 보존했다.
- 20k/30k 동적 반복에서 EE 평균 차이는 각각 약0.005/0.025cm, 속도 RMSE 차이는 약0.002 이하로 이번 후보 간 큰 차이보다 작았다. 단8개 nominal replica의 반복이며 독립된 학습 seed 검증은 아니다. 28k와45k의 약0.02cm EE 평균 차이는 실질적인 우위로 보지 않고 동급으로 해석했다.

## 산출물·보존

- [후보 비교 그래프](candidate_comparison.png), [전체 학습 추세](training_trends.png).
- [전체 조건별140지표](all_conditions.csv), [후보 집계](candidate_metrics.csv), [동적 검증](dynamic_metrics.csv), [전체 체크포인트 학습 통계](all_checkpoint_training_windows.csv).
- 평가 코드와 결과는 이 report 디렉터리에만 추가했다. 기존 source/URDF/PPO/reward/config/checkpoint 및 B1 원본은 변경하지 않았다. 학습·W&B run 생성·optimizer update 없음.
- 원래 학습 manifest의 source hash 일치를 각 평가에서 검사했다. 종료 시 원래226개 checkpoint의 크기/mtime 및 manifest·source hash 보존을 다시 확인했다. Git status는 기존 ownership 오류 때문에 사용할 수 없어 hash/metadata로 확인했다.
'''
(OUT/'REPORT_KO.md').write_text(s, encoding='utf-8')
print('Report saved:',OUT/'REPORT_KO.md')
