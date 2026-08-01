"""CLI for the native-Windows EdgeTAM → DINOv3 → MobileCLIP2 pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from visionforge._cache_paths import apply_d_drive_caches
from visionforge.pipeline.runner import PipelineConfig, run_pipeline

apply_d_drive_caches()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VisionForge native Windows E2E core pipeline (EdgeTAM default)"
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input video file or ordered image-frame directory",
    )
    p.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        required=True,
        help="First-frame bounding box in source image coordinates",
    )
    p.add_argument(
        "--labels",
        type=str,
        default="",
        help="Comma-separated text labels for MobileCLIP2 ranking (optional)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("D:/project/artifacts/e2e_runs"),
        help="Parent directory for unique run artifacts",
    )
    p.add_argument(
        "--tracker",
        type=str,
        default="edgetam",
        help="Tracker backend: edgetam (native Windows CUDA) or sam31 (WSL2 VisionForge-SAM31)",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--width", type=int, default=None, help="Optional process width")
    p.add_argument("--height", type=int, default=None, help="Optional process height")
    p.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow writing into an existing run directory id",
    )
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--parent-job-id", type=str, default=None)
    p.add_argument("--revision-id", type=str, default=None)
    p.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="Optional JSONL file for structured progress events (API/job worker)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = [x.strip() for x in str(args.labels).split(",") if x.strip()]
    try:
        result = run_pipeline(
            PipelineConfig(
                input_path=args.input,
                box_xyxy=(args.box[0], args.box[1], args.box[2], args.box[3]),
                text_labels=labels,
                output_root=args.output_root,
                tracker=args.tracker,
                max_frames=args.max_frames,
                start_frame=int(args.start_frame or 0),
                process_width=args.width,
                process_height=args.height,
                allow_existing=args.allow_existing,
                run_id=args.run_id,
                progress_file=args.progress_file,
                parent_job_id=args.parent_job_id,
                revision_id=args.revision_id,
            )
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest_path": str(result.manifest_path),
                "run_dir": str(result.run_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
