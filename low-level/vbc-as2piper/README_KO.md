# AS2 + PiPER + UMI + Orin — VBC Low-Level 인계 패키지

이 폴더를 상위 GitHub 저장소의 `low-level/vbc-as2piper/`에 넣는다. 현재 서버의 코드·로봇 자산·모델·평가 근거를 보존한 **스냅샷**이며, 실제 push는 아직 하지 않았다.

## 먼저 읽을 문서

1. [현재 상태·미해결 문제](docs/CURRENT_STATE_KO.md)
2. [설치·학습·평가·배포 입력](docs/REPRODUCTION_KO.md)
3. [설계와 누적 지식](docs/KNOWLEDGE_KO.md)
4. [포함 파일과 과거 버전](docs/CONTENTS_KO.md)
5. [GitHub 업로드·LFS·복구 절차](docs/GITHUB_KO.md)
6. [외부 의존성과 이용 조건](docs/EXTERNAL_DEPENDENCIES_KO.md)
7. [미적용 보상안·분석 수치](docs/REWARD_DECISIONS_KO.md) · [다음 작업](docs/NEXT_STEPS_KO.md)

## 사용하는 코드

|목적|디렉터리|상태|
|---|---|---|
|현재70N 학습/후속개발|`low-level-as2piper-head-orin-deploy/`|기본/B70N,카메라 포함,확장 EE 구면/불변 가상중심|
|이전200N·45k 재현|`low-level-as2piper-head-orin-nocontact/`|export 복사본에서만 실제 학습 당시 URDF로 복원|
|원본B1+Z1 참고|`reference/vbc-legacy/`|읽기 전용 참조용 소스·45000 checkpoint 포함|
|과거AS2 변경·gait 연구|기타 `low-level-as2piper*`|역사적 코드. 최신 실행 진입점과 혼동하지 않는다|

`AGENTS.md`가 새 작업 기준이다. `docs/historical/`의 문서와 각 폴더 CHANGES는 당시 기록이며 현재 값과 다를 수 있다.

## 최소 실행

Isaac Gym을 별도 설치하고 Python3.8 학습 환경을 활성화한 뒤 이 폴더에서:

```bash
python scripts/verify_snapshot.py
bash scripts/train70.sh --seed 1 --name "as2piper_70N_$(date -u +%Y%m%dT%H%M%SZ)"
```

위 명령은 **새 초기 가중치로 학습**한다. 기존 checkpoint의 완전 동일 상태 resume 기능은 아니다. 기존 run 이름 재사용/덮어쓰기는 하지 않는다.

학습 조건은6144환경·45000iterations·policy50Hz/physics200Hz. 실제 장시간 학습은 이 패키지를 만들면서 실행하지 않았다.

정리 직전 실행의 모든 완성 checkpoint와 manifest는 각 코드 폴더의 `experiments/contact_force_ablation/runs/`에 보존한다. [목록](docs/checkpoint_index.json)의 최신 iteration은 **스냅샷 시점까지만** 포함하며 이후 서버에서 생성된 모델이 자동 반영되지는 않는다.

## 복구 한계

- Isaac Gym Preview4 SDK는 재배포 제한 때문에 포함하지 않는다. 공식 다운로드에서 별도로 확보해야 한다.
- GitHub에 **LFS 실제 객체까지 push**해야 checkpoint/mesh/data가 보존된다. pointer 파일만 올리면 복구되지 않는다.
- 현재 policy는 과속·정지 RR 접촉 부족이 남아 있어 실물 배포 완료 모델이 아니다.
- 추론 모델 복원과 평가 재현은 가능하도록 묶었지만 RNG·환경상태·history optimizer 등을 저장하지 않아 원래 훈련을 비트 단위로 동일 재개할 수는 없다.
