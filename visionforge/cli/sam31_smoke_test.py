"""CLI smoke test: synthetic video -> real SAM 3.1 Object Multiplex tracking."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from visionforge._cache_paths import apply_d_drive_caches

apply_d_drive_caches()


def _bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024 * 1024), 2)


def _save_masks_and_overlays(
    *,
    masks: list[Any],
    frames_dir: Path,
    masks_dir: Path,
    overlays_dir: Path | None,
    object_id: int,
) -> list[str]:
    import numpy as np
    from PIL import Image

    masks_dir.mkdir(parents=True, exist_ok=True)
    if overlays_dir is not None:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    # Keep one mask per frame for the selected object (last write wins if duplicates).
    by_frame: dict[int, Any] = {}
    for fm in masks:
        if int(fm.object_id) != int(object_id):
            continue
        by_frame[int(fm.frame_index)] = fm.mask

    for frame_index, mask in sorted(by_frame.items()):
        arr = np.asarray(mask).astype(np.uint8) * 255
        out_path = masks_dir / f"mask_{frame_index:05d}_obj{object_id}.png"
        Image.fromarray(arr, mode="L").save(out_path)
        saved.append(str(out_path))

        if overlays_dir is not None:
            frame_path = frames_dir / f"{frame_index:05d}.jpg"
            if frame_path.exists():
                frame = np.asarray(Image.open(frame_path).convert("RGB"))
                overlay = frame.copy()
                m = np.asarray(mask).astype(bool)
                overlay[m] = (
                    (0.45 * overlay[m] + 0.55 * np.array([255, 64, 64])).astype(np.uint8)
                )
                Image.fromarray(overlay).save(
                    overlays_dir / f"overlay_{frame_index:05d}_obj{object_id}.jpg",
                    quality=90,
                )
    return saved


def run_smoke_test(
    *,
    artifacts_dir: Path,
    num_frames: int = 16,
    width: int = 320,
    height: int = 240,
    object_id: int = 1,
    use_box: bool = True,
    save_overlays: bool = True,
) -> dict[str, Any]:
    from visionforge.model_adapters.sam31_adapter import (
        BoxPrompt,
        PointPrompt,
        SAM31Adapter,
        SAM31AdapterError,
    )
    from visionforge.observability.gpu_metrics import (
        empty_cuda_cache,
        reset_peak_stats,
        snapshot_gpu,
    )
    from visionforge.preprocessing.synthetic_video import (
        SyntheticVideoSpec,
        generate_synthetic_video,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = artifacts_dir / "frames"
    masks_dir = artifacts_dir / "masks"
    overlays_dir = artifacts_dir / "overlays" if save_overlays else None
    metrics_path = artifacts_dir / "metrics.json"

    video = generate_synthetic_video(
        frames_dir,
        SyntheticVideoSpec(
            num_frames=num_frames,
            width=width,
            height=height,
        ),
    )

    metrics: dict[str, Any] = {
        "status": "started",
        "num_input_frames": video.num_frames,
        "input_resolution": [video.width, video.height],
        "checkpoint_repo": SAM31Adapter.APPROVED_REPO,
        "checkpoint_name": SAM31Adapter.APPROVED_CHECKPOINT_NAME,
        "warnings": [],
        "output_paths": {
            "frames_dir": str(frames_dir),
            "masks_dir": str(masks_dir),
            "overlays_dir": str(overlays_dir) if overlays_dir else None,
            "metrics_path": str(metrics_path),
        },
    }

    adapter = SAM31Adapter(compile_model=False, use_fa3=False, use_rope_real=False)
    try:
        import torch

        if not torch.cuda.is_available():
            raise SAM31AdapterError("CUDA unavailable")

        empty_cuda_cache()
        reset_peak_stats()
        t0 = time.perf_counter()
        with torch.inference_mode():
            adapter.load()
            load_s = time.perf_counter() - t0
            metrics["model_load_time_sec"] = round(load_s, 3)
            metrics["checkpoint_path"] = (
                str(adapter.resolved_checkpoint) if adapter.resolved_checkpoint else None
            )
            if adapter.resolved_checkpoint is not None:
                metrics["checkpoint_verified_name"] = adapter.resolved_checkpoint.name

            adapter.start_video_session(video.frames_dir)
            if use_box:
                x0, y0, x1, y1 = video.first_frame_box_xyxy
                adapter.add_prompt(
                    frame_index=0,
                    object_id=object_id,
                    box=BoxPrompt(x0=x0, y0=y0, x1=x1, y1=y1, absolute=True),
                )
                metrics["prompt"] = {
                    "kind": "box",
                    "xyxy": [x0, y0, x1, y1],
                    "object_id": object_id,
                }
            else:
                px, py = video.first_frame_point_xy
                adapter.add_prompt(
                    frame_index=0,
                    object_id=object_id,
                    points=[PointPrompt(x=px, y=py, label=1, absolute=True)],
                )
                metrics["prompt"] = {
                    "kind": "point",
                    "xy": [px, py],
                    "object_id": object_id,
                }

            t_inf0 = time.perf_counter()
            result = adapter.track()
            infer_s = time.perf_counter() - t_inf0

        # Prefer selected object; if upstream renumbered, take the first observed id.
        obj_ids = sorted({int(m.object_id) for m in result.masks})
        selected = object_id if object_id in obj_ids else (obj_ids[0] if obj_ids else object_id)
        saved = _save_masks_and_overlays(
            masks=result.masks,
            frames_dir=video.frames_dir,
            masks_dir=masks_dir,
            overlays_dir=overlays_dir,
            object_id=selected,
        )

        frames_with_masks = sorted(
            {
                int(m.frame_index)
                for m in result.masks
                if int(m.object_id) == selected
            }
        )
        selected_masks = [m for m in result.masks if int(m.object_id) == selected]
        adapter.validate_masks(
            selected_masks,
            expected_hw=(video.height, video.width),
        )

        gpu = snapshot_gpu()
        processed = len(frames_with_masks)
        metrics.update(
            {
                "status": "ok",
                "num_processed_frames": processed,
                "masks_saved": len(saved),
                "mask_paths": saved,
                "total_inference_time_sec": round(infer_s, 3),
                "avg_time_per_frame_sec": round(infer_s / max(processed, 1), 4),
                "approx_fps": round(processed / infer_s, 3) if infer_s > 0 else None,
                "peak_cuda_allocated_mb": _bytes_to_mb(gpu.max_allocated_bytes),
                "peak_cuda_reserved_mb": _bytes_to_mb(gpu.max_reserved_bytes),
                "gpu_name": gpu.device_name,
                "gpu_total_memory_mb": _bytes_to_mb(gpu.total_memory_bytes),
                "all_masks_expected_shape": True,
                "masks_non_empty": True,
                "warnings": list(result.warnings) + list(adapter.warnings),
                "selected_object_id": selected,
            }
        )
    except Exception as exc:
        metrics["status"] = "error"
        metrics["error_type"] = type(exc).__name__
        metrics["error"] = str(exc)
        raise
    finally:
        adapter.close()
        empty_cuda_cache()
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VisionForge SAM 3.1 video proof-of-life")
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("D:/project/artifacts/sam31_smoke"),
        help="Ignored artifact output directory (must stay on D:)",
    )
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--object-id", type=int, default=1)
    p.add_argument("--point-prompt", action="store_true", help="Use point instead of box")
    p.add_argument("--no-overlays", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metrics = run_smoke_test(
            artifacts_dir=args.artifacts_dir,
            num_frames=args.num_frames,
            width=args.width,
            height=args.height,
            object_id=args.object_id,
            use_box=not args.point_prompt,
            save_overlays=not args.no_overlays,
        )
    except Exception as exc:
        print(f"SMOKE_TEST_FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(metrics, indent=2))
    return 0 if metrics.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
