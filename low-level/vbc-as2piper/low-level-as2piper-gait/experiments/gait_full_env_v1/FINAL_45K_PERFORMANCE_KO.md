# AS2+PiPER 45,000 iteration 모델 성능 분석

## 결론

45k 모델은 **학습에 사용한 평지 조건에서는 보행과 속도 추종이 매우 안정적**이었다. 새 holdout seed에서 정지·전진·제자리 회전·전진+회전 64개 환경 평가 중 낙상은 없었고, 명령 속도 오차도 작았다.

그러나 저장된 학습 설정은 rough 지형이 아니라 plane이었다. 지형만 rough trimesh로 바꾼 두 holdout 평가에서는 낙상이 반복되고 회전 및 혼합 명령 추종이 나빠졌다. 따라서 이 모델은 평지 보행 baseline으로는 유효하지만, rough 지형이나 실환경 전체에서 안정하다고 판단할 수 없다. 45,000회 학습 자체가 지형 일반화를 보장하지 않았다.

## 실제 학습 run과 환경

- W&B run: `as2piper_hw_gaitfix_vbcsampler_scratch_20261029_134701`, project `as2piper-low`
- task: `as2piper_plate_walk_wbc_vbc`; 평가도 같은 `AS2PiPERWalkWBCVBC` sampler task와 일치시켰다. 저장된 asset과 학습 소스 SHA256이 개발 복사본의 것과 일치했다.
- 최종 checkpoint: `model_45000.pt`; checkpoint metadata의 algorithm counter와 next iteration 모두 45,000이다. 총 6,635,520,000 simulator timesteps가 기록됐다.
- 실제 학습 지형: `mesh_type=plane`. 설정의 terrain proportions에 rough-flat이 있더라도 이 run에서는 평면 mesh가 사용됐다.
- 랜덤화: friction 0.6–1.2, leg motor strength 0.9–1.1, arm motor strength 0.7–1.3은 켜져 있었다. mass, CoM, gripper mass, push randomization은 꺼져 있었다. 저장된 domain randomization 설정에서 범위가 정의되어 있어도 관련 `randomize_*` flag가 false면 실제 적용되지 않는다.
- 학습 command 범위: 전진 속도 ±0.4m/s, yaw 속도 ±0.5rad/s. 평가에는 범위 안의 +0.25m/s 및 +0.5rad/s를 사용했다.
- 20 DOF 모델에서 정책은 다리 12 + 팔 6개, 총 18 action을 출력한다. Gripper 2 DOF는 별도 제어다.

## 학습 로그 수렴 상태

| 완료 iteration | 평균 reward | 전진 추종 항 | yaw 추종 항 | EE world 추종 항 |
|---:|---:|---:|---:|---:|
| 100 | 17.63 | 131.32 | 37.21 | 38.24 |
| 1,000 | 20.27 | 141.69 | 46.30 | 38.19 |
| 10,000 | 21.67 | 145.39 | 47.20 | 38.33 |
| 20,000 | 22.29 | 145.95 | 47.02 | 38.29 |
| 30,000 | 21.92 | 145.75 | 47.29 | 38.31 |
| 40,000 | 22.40 | 146.63 | 47.47 | 38.31 |
| 45,000 | 22.14 | 146.46 | 47.43 | 38.28 |

평균 reward는 약 10k 부근부터 22 전후에서 완만하게 변했다. 마지막 PPO 로그는 surrogate loss −0.0014, value loss 0.0023, history latent supervision 0.0071이었다. 이는 학습 update가 진행되고 손실이 폭주하지 않았다는 뜻이지, rough 지형 성능이 수렴했다는 증거는 아니다. 마지막 rollout의 episode length는 502 step이었다.

마지막 기록에서 `rew_tracking_contacts_shaped_vel`은 −56.97, `rew_tracking_contacts_shaped_force`는 −3.83이었다. 이는 episode 누적 보상항의 부호와 크기다. 실제 접촉 수나 정상반력 자체를 뜻하지 않으므로 이 수치만으로 접촉 성능을 판정하지 않았다.

## Holdout 성능

각 평가는 명령별 16개, 총 64개 환경을 20초씩 실행했다. 같은 checkpoint·task·DR 설정에서 history policy를 사용하고 팔 EE goal을 계속 resample했다. 평지 평가는 seed 20261102, rough 평가는 seed 20261102 및 20261103을 썼다. 각 셀의 낙상은 전체 20초에서 발생한 reset event 수이고, 무낙상 환경은 평가 동안 한 번도 낙상 reset을 겪지 않은 환경의 비율이다.

| 지형 / seed | 명령 | 평균 실측 vx / yaw | 평균 절대 오차 vx / yaw | 무낙상 환경 | 낙상 event | upright 비율 |
|---|---|---|---|---:|---:|---:|
| 평지 / 20261102 | 정지 | 0.007 / −0.006 | 0.010 / 0.013 | 100% | 0 | 100% |
| 평지 / 20261102 | 전진 .25 / 0 | .253 / .004 | .011 / .023 | 100% | 0 | 100% |
| 평지 / 20261102 | 회전 0 / .5 | −.000 / .486 | .014 / .032 | 100% | 0 | 100% |
| 평지 / 20261102 | 혼합 .25 / .5 | .250 / .502 | .010 / .029 | 100% | 0 | 100% |
| Rough / 20261102 | 정지 | .004 / −.005 | .013 / .027 | 81.3% | 3 | 99.8% |
| Rough / 20261102 | 전진 .25 / 0 | .203 / .001 | .065 / .105 | 75.0% | 9 | 97.4% |
| Rough / 20261102 | 회전 0 / .5 | .000 / .366 | .044 / .163 | 93.8% | 2 | 99.6% |
| Rough / 20261102 | 혼합 .25 / .5 | .158 / .321 | .115 / .243 | 43.8% | 19 | 95.8% |
| Rough / 20261103 | 정지 | .001 / −.001 | .011 / .023 | 87.5% | 2 | 99.6% |
| Rough / 20261103 | 전진 .25 / 0 | .204 / −.004 | .067 / .103 | 68.8% | 10 | 98.0% |
| Rough / 20261103 | 회전 0 / .5 | .005 / .287 | .057 / .244 | 81.3% | 4 | 96.5% |
| Rough / 20261103 | 혼합 .25 / .5 | .171 / .343 | .098 / .213 | 68.8% | 11 | 96.7% |

평지에서는 target과 실측이 거의 일치했다. 반면 rough에서 두 seed 모두 전진 평균이 약 .20m/s, 혼합 전진이 .16–.17m/s로 낮아졌고 yaw 추종도 약해졌다. Rough seed 20261102의 낙상은 33회, seed 20261103은 27회였다. 혼합 명령은 두 seed에서 특히 취약했다.

팔 EE 추종 오차는 평지에서 평균 위치 약 1.5–1.6cm, orientation 약 3.6–3.7도였다. rough에서는 평균 위치 약 2.1–2.4cm, orientation 약 4.1–4.6도로 늘었지만, 이번 평가에서 arm-to-body contact는 관측되지 않았다. 다리 actuator는 평지에서 limit 90%를 넘은 샘플이 없었다. Rough seed 20261102에서도 limit 초과는 매우 드물었고, 혼합 명령의 최대 토크가 약 90Nm에 닿았다. 주된 문제는 지속적인 토크 포화보다는 rough 지형에서의 발 디딤과 균형 회복 부족으로 보인다.

## 실제 terrain mesh 기준 발 접촉

독립 rough seed 20261103의 실제 triangle mesh와 발 구(반경 28mm) 사이 최단 거리를 검사했다. 지형에서 2mm 이내를 기하학적 근접 접촉으로 셌다. 이는 정상반력 N이나 관통 깊이를 측정한 값이 아니다.

- 전진·회전·혼합의 발 주기는 약 2.5Hz로, 학습 config의 약 2.52Hz와 가깝다.
- 두 발 이상이 기하학적으로 접촉한 샘플 비율은 약 94–95%였지만, desired stance와 실제 기하 접촉의 일치율은 전진 88%, 회전 79%, 혼합 83%였다.
- 기존 contact proxy의 precision은 약 99.6–99.7%, recall은 약 83.8–88.6%였다. false positive는 적지만 실제 접촉을 일부 놓쳤다.
- 실제 접촉점 기준 접선 미끄러짐은 전진 .10m/s, 회전 .14m/s, 혼합 .14m/s였다.

따라서 rough 성능 저하는 보행 주파수가 사라져 생긴 현상은 아니다. 발 위상과 접촉의 불일치, 거친 면에서의 미끄러짐, 넘어짐 후 회복 문제가 함께 남아 있다. raw contact force에 대한 SDK 누락 문제 때문에 정상 지지력을 직접 측정한 결과로 해석할 수는 없다.

## 영상과 산출물

[rough seed 20261103 실제 정책 상태 영상](../gait_reward_proxy_v2/final_45k_rough_seed20261103/learned_gait.mp4)은 평가에서 저장한 PhysX 상태를 MuJoCo renderer로 재생한다. 12초, 12.5fps이며 scripted gait나 별도 PD 시연이 아니다. 영상은 외형 확인용이고 수치 판정을 대신하지 않는다.

- 평지 전체 결과: `final_45k_plane_vbc_sampler/policy_evaluation.json`
- rough seed 20261102 결과: `final_45k_rough_vbc_sampler/gait_summary.json`
- rough seed 20261103 결과: `final_45k_rough_seed20261103/gait_summary.json`
- 실제 발 mesh 분석: `final_45k_rough_seed20261103/geometric_gait.json`
- 학습 로그: [train_20261029_134701_512623.log](/data/team/users/suhwan/vbc-as2piper/low-level-as2piper-hw/launch_logs/train_20261029_134701_512623.log)
- checkpoint: [model_45000.pt](/data/team/users/suhwan/vbc-as2piper/low-level-as2piper-hw/logs/as2piper-low/as2piper_hw_gaitfix_vbcsampler_scratch_20261029_134701/model_45000.pt)

## 판단

45k 학습은 평지와 제한된 마찰·모터 랜덤화 분포에 대해 수렴했다. Rough 지형 평가는 두 holdout seed에서 같은 취약점을 재현했다. 따라서 현재 checkpoint는 평지 low-level baseline 및 추가 학습의 출발점으로 활용할 수 있으나, rough terrain 또는 실제 전신 조작 주행에 안전한 정책으로 승인할 수 없다. 이 평가는 시뮬레이션 성능이며 하드웨어 안전 검증이 아니다.
