# QuadMani-WholeBodyPickNPlace

사족보행 로봇과 매니퓰레이터를 이용한 전신 Pick-and-Place 프로젝트입니다. Low-Level 이외의 구성도 함께 관리할 수 있도록 하위 디렉터리를 분리합니다.

## 구성

| 경로 | 내용 |
|---|---|
| [low-level/vbc-as2piper](low-level/vbc-as2piper/README_KO.md) | AS2 + PiPER + UMI + Orin의 VBC Low-Level 코드·모델·자산·실험 인계 |

## VBC Low-Level 시작점

- [현재 상태와 검증 결과](low-level/vbc-as2piper/docs/CURRENT_STATE_KO.md)
- [설치·학습·평가](low-level/vbc-as2piper/docs/REPRODUCTION_KO.md)
- [누적 지식·설계 변경·실패 사례](low-level/vbc-as2piper/docs/KNOWLEDGE_KO.md)
- [체크포인트 목록](low-level/vbc-as2piper/docs/checkpoint_index.json)
- [파일 검증 및 외부 의존성](low-level/vbc-as2piper/docs/EXTERNAL_DEPENDENCIES_KO.md)

## 내려받기

모델·메시·평가 데이터는 Git LFS를 사용합니다. Git LFS를 설치한 상태에서 clone한 뒤 실제 객체를 받습니다.

```bash
git clone https://github.com/swaaniida/QuadMani-WholeBodyPickNPlace.git
cd QuadMani-WholeBodyPickNPlace
git lfs pull
python3 low-level/vbc-as2piper/scripts/verify_snapshot.py
```

Isaac Gym Preview4는 라이선스 동의 후 별도로 확보해야 합니다. 학습에는 문서에 명시된 Python/CUDA 환경도 필요합니다. 인증정보와 SDK 바이너리는 이 저장소의 소스 스냅샷에 포함하지 않습니다.

## 스냅샷 범위

VBC 패키지는 2026-11-17 스냅샷입니다. 현재70N 계열은31,800까지, 이전200N 계열은45,000까지의 저장 모델을 포함합니다. 70N의 성능 검증은20k까지 진행됐으며, 실기 배포 완료 모델이 아닙니다. 이후 원래 서버에서 생성되는 결과는 자동으로 추가되지 않습니다.
