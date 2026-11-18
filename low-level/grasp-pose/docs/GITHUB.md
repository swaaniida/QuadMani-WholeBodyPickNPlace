# GitHub 업로드 준비

게시 대상은 `https://github.com/swaaniida/QuadMani-WholeBodyPickNPlace`의 **`low-level/grasp-pose/`** 다. 기존 monorepo에 하위 프로젝트로 포함하며 별도 원격 grasp-pose 저장소를 만들지 않는다. 원격의 다른 프로젝트는 수정하지 않는다.

```bash
# QuadMani-WholeBodyPickNPlace checkout에서:
cd low-level/grasp-pose
# 수정 후 해당 하위 프로젝트만 staging:
git add .
python3 scripts/audit_publish.py
git commit -m "Update GraspGenX RealSense PoC"
git push origin main
```

서버의 원본 작업 폴더 `/data/team/users/suhwan/grasp-pose`는 별도 초기 Git index를 가지고 있다. 최초 업로드는 검증된 index 파일만 원격 checkout의 대상 경로로 복사한다. 내부 `.git`/환경/녹화를 복사하지 않는다. 원본 폴더에서 원격 main으로 바로 push하면 monorepo와 다른 root/history가 되므로 실행하지 않는다.

## 확인할 것

```bash
# Git이 실제 포함하는 목록; 출력은 경로만이다:
git diff --cached --relative --stat
git diff --cached --relative --name-only
# 제외 확인:
git check-ignore graspgenx-realsense/.venv/bin/python graspgenx-realsense/upstream/GraspGenX/README.md graspgenx-realsense/outputs/tracking/server.log
# 선택한 소스만 별도 파일로 보관 (staging 이후):
git diff --cached --binary > /tmp/grasp-pose-initial.patch
```

`audit_publish.py`는 staged 내용의 runtime/checkpoint/녹화/인증파일 경로, secret signature, 50MiB 이상 단일 파일을 검사한다. secret 내용을 출력하지 않는다. 자동검사가 모든 비밀정보·저작권을 판별하는 것은 아니므로 publish 파일 목록과 `THIRD_PARTY_NOTICES.md`를 함께 검토한다. 외부 checkpoint·녹화는 `.gitignore`로 보존 대상에서 제외했다. `git add -f`로 무시 규칙을 우회하지 않는다.

README의 현재 실행 절차와 과거 보고서는 서로 구분했다. `configs/*freeze.txt`는 현재 OS/환경을 그대로 옮겨 설치하는 lockfile이 아니며, 설치 스크립트와 고정 revision이 재구성의 출발점이다.

현재 stage 파일과 같은 내용으로 만든 source-only ZIP 및 파일별 SHA256 manifest는 `graspgenx-realsense/dist/`와 `outputs/repository_prep/`에 로컬 생성한다. 두 경로는 Git에서 제외된다. 원본 녹화와 환경을 삭제/이동하거나 서비스를 재시작하지 않았다.
