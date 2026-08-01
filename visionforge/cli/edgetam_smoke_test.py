"""CLI: local EdgeTAM CUDA video tracking proof-of-life."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from visionforge._cache_paths import apply_d_drive_caches
from visionforge.model_adapters.edgetam_adapter import EdgeTAMAdapter
from visionforge.model_registry import LocalModelRegistry, ModelId
from visionforge.observability.gpu_metrics import empty_cuda_cache
from visionforge.preprocessing.synthetic_video import SyntheticVideoSpec, generate_synthetic_video

apply_d_drive_caches()


def _save_masks(masks, masks_dir: Path, object_id: int) -> list[str]:
    masks_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    by_frame = {}
    for fm in masks:
        if int(fm.object_id) != int(object_id):
            continue
        by_frame[int(fm.frame_index)] = fm.mask
    for idx, mask in sorted(by_frame.items()):
        arr = np.asarray(mask).astype(np.uint8) * 255
        out = masks_dir / f"mask_{idx:05d}_obj{object_id}.png"
        Image.fromarray(arr, mode="L").save(out)
        paths.append(str(out))
    return paths


def _run_once(
    *,
    artifacts_dir: Path,
    num_frames: int,
    width: int,
    height: int,
) -> dict:
    reg = LocalModelRegistry()
    pkg = reg.validate(ModelId.EDGETAM)
    video = generate_synthetic_video(
        artifacts_dir / "frames",
        SyntheticVideoSpec(
            num_frames=num_frames,
            width=width,
            height=height,
            object_size=max(24, min(width, height) // 5),
        ),
    )
    adapter = EdgeTAMAdapter(pkg.primary_checkpoint, device="cuda")
    try:
        adapter.load()
        x0, y0, x1, y1 = video.first_frame_box_xyxy
        result = adapter.track_video_dir(
            video.frames_dir,
            object_id=1,
            box_xyxy=(x0, y0, x1, y1),
            frame_width=video.width,
            frame_height=video.height,
        )
        saved = _save_masks(result.masks, artifacts_dir / "masks", object_id=1)
        processed = result.num_frames
        infer = result.inference_time_sec
        metrics = {
            "status": "ok",
            "checkpoint_path": result.checkpoint_path,
            "num_input_frames": video.num_frames,
            "num_processed_frames": processed,
            "masks_saved": len(saved),
            "input_resolution": [video.width, video.height],
            "load_time_sec": result.load_time_sec,
            "total_inference_time_sec": infer,
            "avg_time_per_frame_sec": round(infer / max(processed, 1), 4),
            "approx_fps": round(processed / infer, 3) if infer > 0 else None,
            "peak_cuda_allocated_mb": None
            if result.peak_allocated_bytes is None
            else round(result.peak_allocated_bytes / (1024 * 1024), 2),
            "peak_cuda_reserved_mb": None
            if result.peak_reserved_bytes is None
            else round(result.peak_reserved_bytes / (1024 * 1024), 2),
            "mask_paths": saved,
            "warnings": result.warnings,
            "config": {"num_frames": num_frames, "width": width, "height": height},
        }
        return metrics
    finally:
        adapter.close()
        empty_cuda_cache()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("D:/project/artifacts/edgetam_smoke"),
    )
    args = p.parse_args(argv)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.artifacts_dir / "metrics.json"

    try:
        metrics = _run_once(
            artifacts_dir=args.artifacts_dir,
            num_frames=8,
            width=256,
            height=256,
        )
        metrics["configuration"] = "original"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return 0
    except Exception as exc:
        original_error = {
            "status": "error",
            "configuration": "original",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        is_oom = "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"
        empty_cuda_cache()
        if not is_oom:
            metrics_path.write_text(json.dumps(original_error, indent=2), encoding="utf-8")
            print(json.dumps(original_error, indent=2))
            return 1

        try:
            reduced = _run_once(
                artifacts_dir=args.artifacts_dir / "reduced",
                num_frames=4,
                width=192,
                height=192,
            )
            reduced["configuration"] = "reduced"
            reduced["original_error"] = original_error
            metrics_path.write_text(json.dumps(reduced, indent=2), encoding="utf-8")
            print(json.dumps(reduced, indent=2))
            return 0
        except Exception as exc2:
            payload = {
                "status": "error",
                "configuration": "reduced_failed",
                "original_error": original_error,
                "reduced_error_type": type(exc2).__name__,
                "reduced_error": str(exc2),
                "reduced_traceback": traceback.format_exc(),
            }
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps(payload, indent=2))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
