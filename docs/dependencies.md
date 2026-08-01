# VisionForge dependencies & licenses

## ProPainter (object removal / video inpainting)

- **Source:** https://github.com/sczhou/ProPainter (NTU S-Lab)
- **Install root (isolated, Windows):** `D:\caches\visionforge\propainter`
  - Clone: `D:\caches\visionforge\propainter\src`
  - Venv: `D:\caches\visionforge\propainter\.venv` (Python **3.11** via `py -3.11`; do **not** install into `D:\project\.venvs\smoke`)
  - Weights: `D:\caches\visionforge\propainter\weights` (also hard-linked into `src\weights`)
- **Runner:** `D:\project\visionforge\inpaint\propainter_runner.py` (subprocess into isolated venv Python)
- **Weights (official release v0.1.0):**
  - `https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth`
  - `https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth`
  - `https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth`
- **Torch / CUDA (what works on this machine):**
  - GPU: NVIDIA GeForce RTX 5060 Laptop GPU (driver reports CUDA 13.2)
  - Working stack in propainter venv: **torch 2.11.0+cu128** + **torchvision 0.26.0+cu128** (`https://download.pytorch.org/whl/cu128`)
  - `torch.cuda.is_available()` → **True**
  - Note: `pip install` of the ~2.7 GB wheel can stall on Windows; downloading the wheel with `curl` then `pip install <local.whl>` is reliable
  - Other deps from ProPainter `requirements.txt` installed into the same venv (`numpy` kept at 2.2.x for compatibility; do not mutate smoke)
- **License:** **NTU S-Lab License 1.0 — NON-COMMERCIAL**
  - Redistribution and use for **non-commercial** purposes in source/binary form are permitted with copyright notice retention and disclaimer.
  - **Commercial use / redistribution requires contacting the authors** (Dr. Shangchen Zhou / Prof. Chen Change Loy).
  - Software is provided **AS IS**, without warranties; liability is disclaimed.
  - Full text: `D:\caches\visionforge\propainter\src\LICENSE`

## RT-DETR (detection)

- **Model:** `PekingU/rtdetr_r18vd` (Transformers `RTDetrForObjectDetection`)
- **Local cache:** `D:\caches\visionforge\models\rtdetr_r18vd`
- **Code:** `visionforge/detection/rtdetr.py` (default detector); OWL-ViT remains text-prompt fallback
- **License:** **Apache-2.0** (PekingU / Apache Software Foundation terms)
  - Permissive use, modification, and distribution with attribution and LICENSE notice
  - Patent grant with defensive termination; AS-IS warranty disclaimer
- **Labels:** Official `config.id2label` COCO mapping; confidence threshold + class-aware NMS; low-confidence → “Unknown object”

## Cache policy

Large downloads (pip, torch, Hugging Face, weights) must stay on `D:\caches\…` — see `visionforge/_cache_paths.py`.
