# GitHub에 올릴 때

정리만수행했으며git init/commit/remote설정/push는아직하지않았다. 원하는상위저장소에`low-level/vbc-as2piper/`전체를복사한다. 기존서버의`.git`은legacy를참조하는linkedworktree이므로절대복사하지않는다.

## LFS

checkpoint/mesh/npz/video/큰HTML등은동봉.gitattributes에LFS정책을정의했다. 상위저장소에서GitLFS를설치/활성화한후add해야한다. 도구가없다면설치방법은 https://git-lfs.com/ 참고.

```bash
cd <상위저장소>
git lfs install
git add low-level/vbc-as2piper
# 아직commit/push전필수확인
git lfs ls-files
python low-level/vbc-as2piper/scripts/verify_snapshot.py
# diff/stat와파일목록검토후commit/push
```

과거에binary를일반Git으로이미add/commit했다면.gitattributes추가만으로과거blob이LFS가되지는않는다. 이패키지를새로추가하기전에LFS활성화순서를지킨다. 자동historyrewrite는하지않는다.

`git push`후`git lfs push --all <remote>`의성공여부도확인한다. 저장소/LFS저장량·전송량제약에걸리면모델을Release/허가된artifactstorage로옮기되README에고정URL·SHA256·다운로드스크립트를같이남겨야한다. 이패키지에는현재로선실제weight가포함돼있다.

## 복구 검증

서버를없애기전에별도PC/폴더에서새로clone하고`git lfs pull`,`python .../scripts/verify_snapshot.py`를실행한다. `.pt`가수백byte의LFS pointer로남으면실제모델복구가아니다. GPU없는PC도정적파일검증/오프라인HTML재생은가능하다. 학습재현은별도SDK/호환CUDA환경이필요하다.

## 제외 항목

인증정보,.git,.venv,.deps/.wandb_deps,컴파일캐시,.so,원시W&B서비스로그,실행중PID/삭제restart도구,SDK바이너리는일반push대상에서제외. frozencheckpoint/runmanifest/압축numericmetrics는의도적으로포함했다.
