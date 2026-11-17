# 외부 의존성/재배포 범위

## 필수 SDK

Isaac Gym Preview4는NVIDIA재배포제한SDK다. 현장python/LICENSE.txt에도express license없는재배포금지문구가있어SDK코드/바이너리/일반SDKassets를포함하지않았다. 이패키지만으로SDK를대체할수는없다.

- 공식확보: https://developer.nvidia.com/isaac-gym/download
- 독립적으로라이선스동의후확보한배포본을`third_party/isaacgym/`에풀어`python/isaacgym`이위치하도록한다.
- 서버폐기전SDK원본설치파일도별도의허가된비공개보관소에보관해야미래공식다운로드의존성을줄일수있다. GitHub소스저장소에는자동포함하지않는다.

## Python/CUDA

- 학습:Linuxx86_64/Ubuntu20.04/Python3.8,PyTorch1.13.1+cu117,torchvision0.14.1+cu117,numpy1.23.5. GPU/드라이버에의존하므로미래의다른GPU에서완전동일동작을보장하지않음.
- PyTorch공식구버전설치: https://docs.pytorch.org/get-started/previous-versions/
- 환경기록은environment/requirements-train.txt와requirements-wandb.txt. 현재설치버전기록이며깨끗한신규OS의전체재설치를이번작업에서실행한것은아님.
- 시각화는별도Python3.11/MuJoCo3.5.0환경. 오래된Py3.8Gym과같은환경으로무리하게통합하지않음.
- OSMesa시스템라이브러리(예:libosmesa6)/ffmpeg/컴파일러/CUDA호환드라이버는OS관리자가준비. 전역OS설정변경은이패키지생성중하지않음.

## 공개 소스/자산

- VBC원본LICENCE/BSDlegged_gym/rsl_rl/Three.js/RealSense출처·license는복사본에유지했다. 단일새라이선스로모든타사자산을재라이선스하지않는다.
- AS2/PiPER CAD/mesh는source bundle과원본출처를보존했다. 하드웨어제조사자산의공개재배포범위가모든파일에명시돼있다고가정하지않는다. 실제공개push전저장소공개범위와각원본이용조건을확인한다.
- 인증정보는포함하지않았다. WANDB_API_KEY는파일에쓰지말고개인로그인/CI secret로관리한다.
