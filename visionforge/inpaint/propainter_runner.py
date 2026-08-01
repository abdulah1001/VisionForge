"""VisionForge ProPainter object-removal runner (isolated venv).

Invokes the official ProPainter inference entrypoint from the cached clone
using the isolated Python at D:\\caches\\visionforge\\propainter\\.venv.
Does not install into or mutate D:\\project\\.venvs\\smoke.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


PROPAINTER_ROOT = Path(r"D:\caches\visionforge\propainter")
PROPAINTER_SRC = PROPAINTER_ROOT / "src"
PROPAINTER_VENV_PYTHON = PROPAINTER_ROOT / ".venv" / "Scripts" / "python.exe"
PROPAINTER_WEIGHTS = PROPAINTER_ROOT / "weights"
INFERENCE_SCRIPT = PROPAINTER_SRC / "inference_propainter.py"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_progress(
    fh: TextIO | None,
    stage: str,
    *,
    completed: int | None = None,
    total: int | None = None,
    overall_percent: float | None = None,
    message: str | None = None,
    extra: dict[str, Any] | None = None,
    event: str = "progress",
) -> None:
    payload: dict[str, Any] = {
        "event": event,
        "stage": stage,
        "timestamp": _utc_now_iso(),
    }
    if completed is not None:
        payload["completed"] = int(completed)
    if total is not None:
        payload["total"] = int(total)
    if overall_percent is not None:
        payload["overall_percent"] = round(float(overall_percent), 2)
    if message:
        payload["message"] = message
    if extra:
        payload["extra"] = extra
    line = json.dumps(payload, ensure_ascii=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()
    print(line, flush=True)


def _count_frames(frames_dir: Path) -> int:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sum(1 for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _ensure_weights_linked() -> None:
    """Point ProPainter src/weights at the VisionForge weights cache when possible."""
    src_weights = PROPAINTER_SRC / "weights"
    src_weights.mkdir(parents=True, exist_ok=True)
    PROPAINTER_WEIGHTS.mkdir(parents=True, exist_ok=True)
    for name in ("raft-things.pth", "ProPainter.pth", "recurrent_flow_completion.pth"):
        cached = PROPAINTER_WEIGHTS / name
        dest = src_weights / name
        if cached.is_file() and not dest.is_file():
            try:
                os.link(cached, dest)
            except OSError:
                shutil.copy2(cached, dest)


def _normalize_output_frames(pp_out_dir: Path, output_dir: Path) -> list[Path]:
    """Copy ProPainter frame_XXXX.png (or ####.png) to frame_00000.png sequence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_subdir = pp_out_dir / "frames"
    sources: list[Path] = []
    if frames_subdir.is_dir():
        sources = sorted(
            p for p in frames_subdir.iterdir() if p.suffix.lower() == ".png"
        )
    written: list[Path] = []
    for idx, src in enumerate(sources):
        dest = output_dir / f"frame_{idx:05d}.png"
        shutil.copy2(src, dest)
        written.append(dest)
    # Also copy video if present
    video = pp_out_dir / "inpaint_out.mp4"
    if video.is_file():
        shutil.copy2(video, output_dir / "inpaint_out.mp4")
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ProPainter object removal for VisionForge."
    )
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--masks-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--width", type=int, default=-1)
    p.add_argument("--height", type=int, default=-1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=80,
        help="Mapped to ProPainter --subvideo_length (long-video chunks).",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=10,
        help="Mapped to ProPainter --neighbor_length (local temporal window).",
    )
    p.add_argument("--progress-file", type=Path, default=None)
    p.add_argument(
        "--save-video",
        action="store_true",
        help="Keep ProPainter mp4 in addition to PNG sequence (always requests --save_frames).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    progress_fh: TextIO | None = None
    if args.progress_file is not None:
        args.progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_fh = args.progress_file.open("a", encoding="utf-8")

    try:
        if not PROPAINTER_VENV_PYTHON.is_file():
            emit_progress(
                progress_fh,
                "error",
                message=f"ProPainter venv python missing: {PROPAINTER_VENV_PYTHON}",
                event="error",
            )
            return 2
        if not INFERENCE_SCRIPT.is_file():
            emit_progress(
                progress_fh,
                "error",
                message=f"ProPainter inference script missing: {INFERENCE_SCRIPT}",
                event="error",
            )
            return 2
        if not args.frames_dir.is_dir():
            emit_progress(
                progress_fh,
                "error",
                message=f"frames-dir not found: {args.frames_dir}",
                event="error",
            )
            return 2
        if not args.masks_dir.is_dir():
            emit_progress(
                progress_fh,
                "error",
                message=f"masks-dir not found: {args.masks_dir}",
                event="error",
            )
            return 2

        n_frames = _count_frames(args.frames_dir)
        emit_progress(
            progress_fh,
            "propainter_init",
            completed=0,
            total=max(n_frames, 1),
            overall_percent=0.0,
            message="Starting ProPainter inference",
            extra={
                "frames_dir": str(args.frames_dir),
                "masks_dir": str(args.masks_dir),
                "output_dir": str(args.output_dir),
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "fp16": bool(args.fp16),
            },
        )

        _ensure_weights_linked()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vf_propainter_") as tmp:
            # ProPainter writes under output/<video_name>/
            pp_output_root = Path(tmp) / "results"
            cmd = [
                str(PROPAINTER_VENV_PYTHON),
                str(INFERENCE_SCRIPT),
                "--video",
                str(args.frames_dir.resolve()),
                "--mask",
                str(args.masks_dir.resolve()),
                "--output",
                str(pp_output_root),
                "--subvideo_length",
                str(args.chunk_size),
                "--neighbor_length",
                str(args.overlap),
                "--save_frames",
            ]
            if args.width > 0 and args.height > 0:
                cmd.extend(["--width", str(args.width), "--height", str(args.height)])
            if args.fp16:
                cmd.append("--fp16")

            emit_progress(
                progress_fh,
                "propainter_run",
                completed=0,
                total=max(n_frames, 1),
                overall_percent=5.0,
                message="Invoking official inference_propainter.py",
                extra={"cmd": cmd},
            )

            env = os.environ.copy()
            env["PYTHONPATH"] = str(PROPAINTER_SRC) + os.pathsep + env.get("PYTHONPATH", "")
            # Prefer cached weights; inference also auto-downloads into src/weights
            env.setdefault("PIP_CACHE_DIR", r"D:\caches\pip")

            proc = subprocess.run(
                cmd,
                cwd=str(PROPAINTER_SRC),
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.stdout:
                print(proc.stdout, flush=True)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, flush=True)

            if proc.returncode != 0:
                emit_progress(
                    progress_fh,
                    "error",
                    message="ProPainter inference failed",
                    event="error",
                    extra={
                        "returncode": proc.returncode,
                        "stderr_tail": (proc.stderr or "")[-4000:],
                    },
                )
                return proc.returncode or 1

            # Locate ProPainter result folder (results/<video_name>/)
            candidates = list(pp_output_root.glob("*"))
            result_dirs = [c for c in candidates if c.is_dir()]
            if not result_dirs:
                emit_progress(
                    progress_fh,
                    "error",
                    message="No ProPainter output directory found",
                    event="error",
                    extra={"stdout_tail": (proc.stdout or "")[-2000:]},
                )
                return 1
            pp_out = result_dirs[0]
            written = _normalize_output_frames(pp_out, args.output_dir)
            if not args.save_video:
                vid = args.output_dir / "inpaint_out.mp4"
                if vid.is_file():
                    # keep video by default as useful artifact; user asked PNG OR video
                    pass

            emit_progress(
                progress_fh,
                "propainter_done",
                completed=len(written),
                total=max(n_frames, len(written), 1),
                overall_percent=100.0,
                message=f"Wrote {len(written)} frames to {args.output_dir}",
                extra={
                    "frames": [str(p.name) for p in written[:5]],
                    "frame_count": len(written),
                    "has_video": (args.output_dir / "inpaint_out.mp4").is_file(),
                },
                event="complete",
            )
        return 0
    finally:
        if progress_fh is not None:
            progress_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
