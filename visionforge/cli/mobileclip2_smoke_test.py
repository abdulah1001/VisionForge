"""CLI: local MobileCLIP2-S0 CUDA proof-of-life."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from visionforge._cache_paths import apply_d_drive_caches
from visionforge.model_adapters.mobileclip2_adapter import MobileCLIP2Adapter
from visionforge.model_registry import LocalModelRegistry, ModelId

apply_d_drive_caches()


def _synthetic_rgb(seed: int = 11, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(20, 180, size=(size, size, 3), dtype=np.uint8)
    img[40:120, 40:200] = (30, 180, 40)  # green-ish region
    return img


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("D:/project/artifacts/mobileclip2_smoke"),
    )
    args = p.parse_args(argv)

    reg = LocalModelRegistry()
    pkg = reg.validate(ModelId.MOBILECLIP2_S0)
    adapter = MobileCLIP2Adapter(pkg.primary_checkpoint, device="cuda")
    labels = ["a red circle", "a green rectangle", "a blue sky"]
    try:
        adapter.load()
        result = adapter.encode_and_compare(_synthetic_rgb(), labels)
    finally:
        adapter.close()

    if not result.finite:
        print("MOBILECLIP2_FAILED: non-finite values")
        return 1

    out = {
        "status": "ok",
        "checkpoint_path": result.checkpoint_path,
        "image_embedding_shape": list(result.image_embedding_shape),
        "text_embedding_shape": list(result.text_embedding_shape),
        "embedding_dim": result.embedding_dim,
        "labels": result.labels,
        "similarity_scores": result.similarity_scores,
        "finite": result.finite,
        "load_time_sec": result.load_time_sec,
        "inference_time_sec": result.inference_time_sec,
        "peak_cuda_allocated_mb": None
        if result.peak_allocated_bytes is None
        else round(result.peak_allocated_bytes / (1024 * 1024), 2),
        "peak_cuda_reserved_mb": None
        if result.peak_reserved_bytes is None
        else round(result.peak_reserved_bytes / (1024 * 1024), 2),
        "warnings": list(pkg.warnings),
    }
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = args.artifacts_dir / "metrics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
