# 설치·실행·평가·복구

## 준비

```bash
cd low-level/vbc-as2piper
python3.8 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install pip==23.3.2 setuptools==59.5.0
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r environment/requirements-train.txt -r environment/requirements-wandb.txt
# 라이선스동의후별도받은IsaacGym을third_party/isaacgym/에풀어놓은상태
python -m pip install --no-deps -e third_party/isaacgym/python
python scripts/verify_snapshot.py
```

rsl_rl은PyPI최신버전으로대체하지않고각코드의rsl_rl_local을PYTHONPATH에서사용한다. legged_gym setup.py의일반dependency를무작정업그레이드하지않는다. 기존requirements도서버실제버전이기준이다.

## W&B

```bash
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_ENTITY=quadruped_capstone
wandb login --host "$WANDB_BASE_URL"
```

로그인은각자터미널에서한다. .netrc나API키를공유/커밋하지않는다. wandb-core/gpu_stats실행권한을확인한다. requirements로설치된일반패키지에실행권한이정상이라면추가chmod는불필요하다.

## 새70N학습(처음부터)

```bash
run_name="as2piper_70N_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p local_logs
nohup setsid env WANDB_MODE=online PYTHONUNBUFFERED=1 \
  bash scripts/train70.sh --seed 1 --project as2piper-low --name "$run_name" \
  > "local_logs/$run_name.log" 2>&1 < /dev/null &
tail -f "local_logs/$run_name.log"
```

6144환경/45000iterations/200주기checkpoint가config기본이다. 신규이름을사용하고기존manifest폴더를덮어쓰지않는다. 소규모실험을원하면원본config직접변경대신별도copy에서명시적override를기록한다.

## 저장된20k평가

```bash
checkpoint="$PWD/low-level-as2piper-head-orin-deploy/experiments/contact_force_ablation/runs/as2piper_headpiper_orin_70N_20261115T133137Z/model_20000.pt"
bash scripts/evaluate70.sh --checkpoint "$checkpoint" --seed 1 --device cuda:0 --terrain flat --seconds 40 --name "20k_flat_$(date -u +%Y%m%dT%H%M%SZ)"
# rough는--terrain rough로별도실행
```

배포용historyencoder로추론,optimizerupdate0. sourcehash와runmanifest가일치하는지검사. currentsource는현재70N의기준이다. baseline200평가는nocontact코드의run_force_ablation.sh를사용한다. 같은관측차원이어도두config/URDF/좌표계를혼합하지않는다.

## 체크포인트 해석

`model_state_dict`에actor/critic/privileged/history가포함된다. mainoptimizer/iter/infos도저장된다. historyoptimizer/ROAcounter/RNG/환경/historybuffer는누락되어완전동일resume는불가. `--mode train`은resume옵션이없고seed초기가중치로시작한다.

## 실기 배포 입력

755는학습전용67+18+670이다. 실제입력은67+670=737. JITexporter가737을요구하고18privileged는필요없다. 현재action18의팔출력6은원본마스킹을유지하며실팔명령은IK별도경로다. PyTorch/JIT파일만으로AS2/PiPER실물통신이완성되는것은아니다.

## 시각화

별도Python3.11환경에서environment/requirements-viz.txt설치. OSMesa예:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 \
  python mujoco_mount_compare/run_mount_demo.py --variant plate_only --mode arm --seconds 5 --video
```

이것은역사적PD데모다. 학습policy평가는별도환경에서수행한다. 파일만보려면`reports/ee_ground_compare_20261114/viewer/downloads/as2piper_3d_offline.html`을브라우저로열면된다. 기록은이전200N45k계열이며새70N최신모델영상이아니다.

과거reports/tools의절대경로는출처기록으로보존했다. 먼저`python scripts/relocate_historical.py --dry-run`으로목록을확인하고복사본에서`--apply`한다. 현재두핵심runtime/nativePython이나run sourcehash를임의로rewrite하지않는다.
