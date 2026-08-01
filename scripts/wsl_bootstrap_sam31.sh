#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "==> DNS"
printf "nameserver 8.8.8.8\nnameserver 1.1.1.1\n" > /etc/resolv.conf || true
mkdir -p /etc/apt/apt.conf.d
printf 'Acquire::ForceIPv4 "true";\nAcquire::Retries "5";\n' > /etc/apt/apt.conf.d/99force-ipv4

echo "==> apt packages"
apt-get update -y
apt-get install -y --no-install-recommends \
  python3-venv python3-pip python3-dev build-essential \
  git curl ca-certificates libgl1 libglib2.0-0

echo "==> venv"
mkdir -p /opt/visionforge
if [[ ! -x /opt/visionforge/venv/bin/python ]]; then
  python3 -m venv /opt/visionforge/venv
fi
# shellcheck disable=SC1091
source /opt/visionforge/venv/bin/activate

export PIP_CACHE_DIR=/mnt/d/caches/pip
export TORCH_HOME=/mnt/d/caches/torch
export HF_HOME=/mnt/d/caches/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/mnt/d/caches/tmp
mkdir -p "$PIP_CACHE_DIR" "$TORCH_HOME" "$HF_HOME" "$TMPDIR"

python -m pip install -U pip setuptools wheel

echo "==> PyTorch cu128 (match Windows smoke)"
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio

echo "==> pin numpy<2 (sam3 constraint) + runtime deps + sam3 editable"
python -m pip install "numpy>=1.26,<2" pillow einops pycocotools psutil
# Install package deps without network model downloads.
python -m pip install -e /mnt/d/caches/sam3_src
# Ensure triton present (usually via torch); fail loudly if missing.
python -c "import triton; print('triton', triton.__version__)"

echo "==> verify"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
import triton
print("triton", triton.__version__)
import sam3
from sam3.model_builder import build_sam3_multiplex_video_predictor
print("sam3_ok", sam3.__file__)
print("builder_ok", build_sam3_multiplex_video_predictor)
PY

echo "==> BOOTSTRAP_OK"
