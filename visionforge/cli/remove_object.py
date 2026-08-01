"""CLI: remove a selected object from video (track → ProPainter → cleaned MP4)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from visionforge._cache_paths import apply_d_drive_caches
from visionforge.pipeline.remove_object import RemoveConfig, RemoveError, run_remove_object

apply_d_drive_caches()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VisionForge object removal pipeline")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--box", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"), required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--tracker", type=str, default="edgetam")
    p.add_argument("--label", type=str, default="")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-side", type=int, default=960)
    p.add_argument("--quality", choices=("standard", "high"), default="standard")
    p.add_argument("--mask-dilate", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=48)
    p.add_argument("--progress-file", type=Path, default=None)
    p.add_argument("--run-id", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_remove_object(
            RemoveConfig(
                input_path=args.input,
                box_xyxy=(args.box[0], args.box[1], args.box[2], args.box[3]),
                output_root=args.output_root,
                tracker=args.tracker,
                selected_label=args.label or None,
                max_frames=args.max_frames,
                max_side=args.max_side,
                quality_mode=args.quality,
                mask_dilate_px=args.mask_dilate,
                chunk_size=args.chunk_size,
                progress_file=args.progress_file,
                run_id=args.run_id,
            )
        )
    except RemoveError as exc:
        msg = str(exc)
        code = "PIPELINE_FAILED"
        user = "Object removal could not be completed. Your original video was not changed."
        if msg.startswith("TRACK_UNRELIABLE:"):
            code = "TRACK_UNRELIABLE"
            user = msg.split(":", 1)[1]
        elif msg.startswith("GPU_OOM:"):
            code = "GPU_OOM"
            user = msg.split(":", 1)[1]
        elif msg.startswith("INPAINT_INCOMPLETE:"):
            code = "INPAINT_INCOMPLETE"
            user = msg.split(":", 1)[1]
        print(json.dumps({"status": "error", "error_code": code, "error": user, "detail": msg[:500]}))
        return 1
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        low = detail.lower()
        if "out of memory" in low or "outofmemory" in low:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_code": "GPU_OOM",
                        "error": (
                            "This video is too large for the available GPU. "
                            "Optimize it to 640p and try again."
                        ),
                        "detail": detail[:500],
                    }
                )
            )
            return 1
        print(json.dumps({"status": "error", "error_code": "PIPELINE_FAILED", "error": detail}))
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest_path": str(result.manifest_path),
                "run_dir": str(result.run_dir),
                "cleaned_mp4": str(result.cleaned_mp4),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
