# 병행 학습 자원 검사

판정: **6144 env 병행 학습은 메모리상 가능하지만 기존 학습 속도 저하를 감수해야 한다.**

- 할당 장치: A100 MIG 3g.40gb, 42 SM, 39.25 GiB. A100 전체80GB를 쓰는 것이 아니다.
- 단독 메모리: 7.85 GiB. 병행 관측 최대: 16.02 GiB. 최소 관측 여유: 23.23 GiB.
- 기존 FPS: 단독 37,404, 병행 23,000 (**38.5% 감소**), 종료 후 37,698.
- 새 검사 run: 6144env×24step×4iteration=589,824 transitions, 합산 wall time 기준 10,079 FPS. Iteration별 FPS는 6,257–12,971. 목표 충돌 검사 비용과 초기 워밍업에 따라 변동하며 장시간 성능으로 일반화하면 안 된다.
- 전 환경 동시 EE resampling: 1.088초. 이 검사에서 837환경이 최대 재추출 후 이전 목표를 유지했다.
- 호스트 cgroup: CPU16코어 quota, 메모리125GiB. 관측 최대 메모리 24.96GiB. memory.events의 OOM/OOM-kill=0 유지.
- MIG의 GPU-Util은 N/A이므로 GPU가 쉬고 있다는 근거가 아니다. 메모리 여유와 연산/실행시간 여유를 구분해야 한다.

검사 프로세스는 정상 종료했고 GPU 메모리는8040MiB로 돌아왔다. 기존 학습 PID220646은 그대로 실행 중이다. B1/기존AS2 코드·학습을 수정/중지하지 않았고 W&B 실험은 만들지 않았다. 새 장시간 학습도 시작하지 않았다.

동시 실험을 우선하면 새 run을 시작할 자원은 충분하다. 기존 학습 처리량을 유지해야 한다면 같은 MIG에 병행하지 않는 편이 적합하다. 본 검사는 짧은 시작 구간의 자원 확인이며 수렴·장시간 성능/안전성 검증이 아니다.

근거: `capacity_report.json`, `capacity_probe_measurements.json`, `capacity_samples.json`, `capacity_probe_stdout.log`. 기존 FPS의 정확한 iteration 구간은 JSON에 있다. PyTorch allocator 수치에는 PhysX 등 외부 CUDA 할당이 모두 포함되지 않아 전체 메모리 판정에는 MIG 장치 사용량을 썼다.
