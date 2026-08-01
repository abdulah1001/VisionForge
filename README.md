# VisionForge

**Local AI Video Object Remover**

VisionForge is a localhost studio for removing a selected object from a short video and exporting a cleaned MP4. You upload a clip, pause on a clear frame, detect or draw a box, confirm the target, then run a single GPU job that tracks the object, rebuilds the background, and remuxes audio when it exists.

The product goal is practical local use on consumer NVIDIA hardware (about 8GB VRAM), not cloud inference or a research dashboard. Defaults favor completion over maximum resolution: Standard (~720p), Optimized retry (~640p), and a soft duration budget so long phone clips do not collapse the worker.

```
Upload → Analyze objects → Select target → Remove → Preview / Download
```

---

## Why this exists

Most object-removal demos either ship as notebooks, require cloud GPUs, or hide failure modes. VisionForge packages a full path you can run on one machine:

1. A React Studio that only exposes product steps (no model-config clutter in the main flow)
2. A FastAPI worker that owns one CUDA job at a time
3. A removal pipeline that connects detection → tracking → inpainting → H.264 export
4. Explicit limits and recovery when VRAM runs out, instead of a silent blank result

Footage stays on disk under local artifact paths. The API binds to `127.0.0.1` by default.

---

## What this project delivers

### Product surface

- **Studio workflow** for upload, object analysis, selection (auto or manual box), quality mode, remove, processing, failure, and result preview
- **In-page cleaned MP4 preview** plus download; original file is never overwritten
- **Failure screen** with clear codes; GPU OOM maps to **Retry Optimized** instead of dumping back to an empty upload state
- **Jobs** and **System** pages for queue visibility and tracker capability status
- **Landing** positioned as a local remover (privacy, three-step story, Studio CTA)

### Pipeline

| Stage | Implementation |
|-------|----------------|
| Detect | RT-DETR (`PekingU/rtdetr_r18vd`) with class-aware NMS; OWL-ViT remains a text-prompt fallback |
| Track | EdgeTAM on native Windows CUDA; optional SAM 3.1 via WSL path |
| Mask prep | Dilated masks for inpaint stability |
| Inpaint | ProPainter in an **isolated** venv/cache (does not mutate the main smoke env) |
| Encode | Frame sequence → H.264; audio copied/muxed when the source has an audio stream |

### Engineering for 8GB-class GPUs

- Processing long-side caps (Standard / High / Optimized)
- Soft **~60s** duration budget and hard frame safety caps
- EdgeTAM **chunked tracking** with video/state offload to CPU on long clips
- ProPainter chunked inference with smaller chunk retry on OOM
- Friendly error mapping so CUDA OOM surfaces as `GPU_OOM`, not a generic pipeline failure

### Platform hygiene

- Weights and large caches stay **out of git** (`models/`, `artifacts/`, `D:\caches\…`)
- Offline-friendly loading when packs are already present (`HF_HUB_OFFLINE` / local dirs)
- Unit coverage for API job rules, geometry, detection helpers, tracker capability contracts
- Dependency and license notes for ProPainter (non-commercial) and RT-DETR in `docs/dependencies.md`

---

## Architecture (short)

```
frontend/ (Vite + React)
    │  REST
    ▼
visionforge.api  →  job queue (1 GPU worker)
    │
    ▼
visionforge.cli.remove_object
    │
    ├─ detection (RT-DETR)
    ├─ tracking (EdgeTAM / SAM 3.1)
    ├─ inpaint (ProPainter subprocess)
    └─ encode + audio mux → cleaned.mp4
```

| Layer | Role |
|-------|------|
| `frontend/` | Studio UI: `layout/`, `remover/`, `jobs/` |
| `visionforge/api/` | FastAPI, media upload/probe, job store, worker |
| `visionforge/pipeline/` | `remove_object` orchestration |
| `visionforge/detection/` | RT-DETR + box ops |
| `visionforge/tracking/` | Tracker backends |
| `visionforge/inpaint/` | ProPainter runner |
| `tests/` | Pytest unit + integration |
| `scripts/` | Cache / WSL helpers |

---

## Operating limits

These are product constraints for a single local GPU worker:

| Constraint | Detail |
|------------|--------|
| VRAM | Full-res or long clips can still OOM on 8GB. Prefer Standard (~720p) or Optimized (~640p). |
| Duration | Soft process cap ~**60 seconds**. Longer inputs are truncated. |
| Hardware | NVIDIA CUDA required for removal. CPU-only is unsupported. |
| Quality | Fast motion, thin structures (hair, poles, leashes), and heavy occlusion can leave residue. |
| Concurrency | One active GPU job by design. |
| Network | Localhost studio, not multi-tenant SaaS. |
| Weights | Not in the repository. Place packs under `models/` and/or cache roots. |
| License | ProPainter is **NTU S-Lab 1.0 — non-commercial**. Commercial use needs upstream permission. |

On OOM, use **Retry Optimized** in Studio for the same selection at a smaller footprint.

---

## Stack

- **Backend:** Python 3.12+, FastAPI, Pydantic, OpenCV/Pillow, PyTorch (CUDA) in the working env
- **Frontend:** React 19, Vite, TanStack Query, Zustand, Tailwind CSS 4, Framer Motion
- **Models (local packs):** RT-DETR, EdgeTAM, optional SAM 3.1, ProPainter weights in isolated cache
- **Target OS for the documented path:** Windows + optional WSL2 for SAM 3.1

---

## Quick start

```powershell
# API
cd D:\project
$env:PYTHONPATH="D:\project"; $env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
& "D:\project\.venvs\smoke\Scripts\python.exe" -m visionforge.api.server --host 127.0.0.1 --port 8000

# UI
cd D:\project\frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

| Service | URL |
|---------|-----|
| Studio | http://127.0.0.1:5173/studio |
| API | http://127.0.0.1:8000 |

Caches, ProPainter venv, and license notes: `docs/dependencies.md`, `scripts/use_d_caches.ps1`.

---

## Roadmap

1. Pre-job VRAM policy (auto resolution / frame budget from measured memory)
2. In-pipeline OOM recovery without a full job restart
3. Stronger masks and multi-object remove in one pass
4. Before/after compare and clearer duration / VRAM estimates in Studio
5. One-shot Windows setup (env, weight check, ports)
6. Public regression fixtures (duration, audio preserve, OOM paths)

---

## Contributing

Human contributions are welcome when they improve the local remover path.

**Please:**

- Keep the product local-first (`127.0.0.1`, no silent cloud upload)
- Do not commit weights, videos, or `artifacts/` outputs
- Respect upstream licenses (especially ProPainter non-commercial terms)
- Prefer small, reviewable changes with tests for API/pipeline behavior where practical
- Match existing structure: `visionforge/` for backend, `frontend/src/components/{layout,remover,jobs}` for UI

**Pull requests:** open against `master` with a short summary of the user-visible change and how you verified it (unit test, manual Studio run, or both).

---

## Licensing

Application code in this repository follows the license you attach when publishing.

Upstream models (EdgeTAM, SAM, DINOv3, MobileCLIP2, RT-DETR, ProPainter) keep their own terms. This README is not commercial clearance.
