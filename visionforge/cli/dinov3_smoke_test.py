"""CLI: local DINOv3 ViT-S/16 CUDA proof-of-life."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from visionforge._cache_paths import apply_d_drive_caches
from visionforge.model_adapters.dinov3_adapter import DINOv3Adapter
from visionforge.model_registry import LocalModelRegistry, ModelId

apply_d_drive_caches()


def _synthetic_rgb(seed: int = 7, size: int = 224) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(30, 200, size=(size, size, 3), dtype=np.uint8)
    # Bright square for a stable visual structure.
    img[70:150, 70:150] = (240, 40, 40)
    return img


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("D:/project/artifacts/dinov3_smoke"),
    )
    args = p.parse_args(argv)

    reg = LocalModelRegistry()
    pkg = reg.validate(ModelId.DINOV3_VITS16)
    adapter = DINOv3Adapter(pkg.package_dir, device="cuda")
    try:
        adapter.load()
        result = adapter.encode_rgb_image(_synthetic_rgb())
    finally:
        adapter.close()

    if not result.finite:
        print("DINOV3_FAILED: non-finite embedding")
        return 1

    out = {
        "status": "ok",
        "checkpoint_path": result.checkpoint_path,
        "package_dir": result.package_dir,
        "input_resolution": list(result.input_resolution),
        "output_shape": list(result.shape),
        "dtype": result.dtype,
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
