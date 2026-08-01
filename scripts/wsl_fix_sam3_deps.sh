#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source /opt/visionforge/venv/bin/activate
export PIP_CACHE_DIR=/mnt/d/caches/pip
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/mnt/d/caches/torch
export HF_HOME=/mnt/d/caches/huggingface
export TMPDIR=/mnt/d/caches/tmp

python -m pip install "numpy>=1.26,<2" einops

for _ in $(seq 1 10); do
  set +e
  OUT=$(python - <<'PY' 2>&1
import sam3
from sam3.model_builder import build_sam3_multiplex_video_predictor
print("IMPORT_OK")
PY
)
  STATUS=$?
  set -e
  echo "$OUT"
  if echo "$OUT" | grep -q IMPORT_OK; then
    echo "DEPS_OK"
    break
  fi
  MOD=$(echo "$OUT" | sed -n "s/.*No module named '\([^']*\)'.*/\1/p" | head -1)
  if [[ -z "${MOD}" ]]; then
    echo "Non-missing-module failure (status=$STATUS)"
    exit 1
  fi
  echo "Installing missing: $MOD"
  python -m pip install "$MOD"
done

python - <<'PY'
import torch
import triton
import sam3
import numpy as np
from sam3.model_builder import build_sam3_multiplex_video_predictor
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("triton", triton.__version__)
print("numpy", np.__version__)
print("builder", build_sam3_multiplex_video_predictor)
print("VERIFY_OK")
PY
