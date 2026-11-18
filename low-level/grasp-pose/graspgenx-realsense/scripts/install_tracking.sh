#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/scripts/bootstrap_uv.sh"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
if [[ ! -d upstream/Tracking_SAM/.git ]]; then
  git clone https://github.com/RogerQi/Tracking_SAM.git upstream/Tracking_SAM
  git -C upstream/Tracking_SAM checkout d5e5e81a361dce4ddbe3264a2b8027b2b43ea3fd
fi
git -C upstream/Tracking_SAM submodule update --init --recursive
tools/uv venv --python "$TASK_PYTHON" .venv-tracking --allow-existing
tools/uv pip install --python .venv-tracking/bin/python 'torch==1.13.1+cu117' 'torchvision==0.14.1+cu117' --index-url https://download.pytorch.org/whl/cu117
tools/uv pip install --python .venv-tracking/bin/python -r configs/tracking-requirements.txt
tools/uv pip install --python .venv-tracking/bin/python 'git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf'
export CUDA_HOME=/usr/local/cuda-11.7
export TORCH_CUDA_ARCH_LIST=8.0
export GPU_ARCH=80
export MAX_JOBS=2
tools/uv pip install --python .venv-tracking/bin/python --no-build-isolation --no-deps \
  ./upstream/Tracking_SAM/tracking_SAM/third_party/Pytorch-Correlation-extension \
  ./upstream/Tracking_SAM/tracking_SAM/third_party/GroundingDINO
mkdir -p upstream/Tracking_SAM/pretrained_weights outputs/tracking
WEIGHTS="$PROJECT_DIR/upstream/Tracking_SAM/pretrained_weights"
if [[ ! -f "$WEIGHTS/sam_vit_h_4b8939.pth" ]]; then
  curl -L --fail https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o "$WEIGHTS/sam_vit_h_4b8939.pth.part"
  mv "$WEIGHTS/sam_vit_h_4b8939.pth.part" "$WEIGHTS/sam_vit_h_4b8939.pth"
fi
if [[ ! -f "$WEIGHTS/AOTT_PRE_YTB_DAV.pth" ]]; then
  .venv-tracking/bin/gdown 'https://drive.google.com/uc?id=1owPmwV4owd_ll6GuilzklqTyAd0ZvbCu' -O "$WEIGHTS/AOTT_PRE_YTB_DAV.pth"
fi
if [[ ! -f "$WEIGHTS/groundingdino_swint_ogc.pth" ]]; then
  curl -L --fail https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -o "$WEIGHTS/groundingdino_swint_ogc.pth.part"
  mv "$WEIGHTS/groundingdino_swint_ogc.pth.part" "$WEIGHTS/groundingdino_swint_ogc.pth"
fi
tools/uv pip freeze --python .venv-tracking/bin/python > configs/tracking-freeze.txt
