"""Input frame discovery and safe output directory helpers."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


class PipelineIOError(Exception):
    pass


_FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class FrameSequence:
    source: Path
    frame_paths: list[Path]
    width: int
    height: int
    is_extracted_video: bool = False


def _sort_frame_paths(paths: list[Path]) -> list[Path]:
    def key(p: Path) -> tuple:
        stem = p.stem
        m = re.search(r"(\d+)$", stem)
        if m:
            return (0, int(m.group(1)), p.name.lower())
        return (1, stem.lower(), p.name.lower())

    return sorted(paths, key=key)


def load_frame_sequence(
    input_path: str | Path,
    *,
    max_frames: int | None = None,
    start_frame: int = 0,
    extract_dir: Path | None = None,
) -> FrameSequence:
    path = Path(input_path)
    if not path.exists():
        raise PipelineIOError(f"Input does not exist: {path}")

    if path.is_dir():
        frames = [
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in _FRAME_EXTS
        ]
        frames = _sort_frame_paths(frames)
        if not frames:
            raise PipelineIOError(f"No readable image frames in directory: {path}")
        start = max(0, int(start_frame or 0))
        if start:
            frames = frames[start:]
        if max_frames is not None:
            frames = frames[: max(0, int(max_frames))]
        with Image.open(frames[0]) as im:
            w, h = im.size
        # Verify readable
        for fp in frames:
            with Image.open(fp) as im:
                im.load()
        return FrameSequence(
            source=path.resolve(),
            frame_paths=[p.resolve() for p in frames],
            width=int(w),
            height=int(h),
            is_extracted_video=False,
        )

    if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        if extract_dir is None:
            raise PipelineIOError("extract_dir required when input is a video file")
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            import cv2
        except ImportError as exc:
            raise PipelineIOError("opencv is required to read video files") from exc
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise PipelineIOError(f"Unable to open video: {path}")
        frames: list[Path] = []
        start = max(0, int(start_frame or 0))
        src_idx = 0
        kept = 0
        try:
            while True:
                if max_frames is not None and kept >= int(max_frames):
                    break
                ok, bgr = cap.read()
                if not ok:
                    break
                if src_idx < start:
                    src_idx += 1
                    continue
                rgb = bgr[:, :, ::-1]
                out = extract_dir / f"{kept:05d}.jpg"
                Image.fromarray(rgb).save(out, quality=95)
                frames.append(out.resolve())
                kept += 1
                src_idx += 1
        finally:
            cap.release()
        if not frames:
            raise PipelineIOError(f"No frames decoded from video: {path}")
        with Image.open(frames[0]) as im:
            w, h = im.size
        return FrameSequence(
            source=path.resolve(),
            frame_paths=frames,
            width=int(w),
            height=int(h),
            is_extracted_video=True,
        )

    raise PipelineIOError(
        f"Unsupported input (need image directory or video file): {path}"
    )


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def maybe_resize_frames(
    frames: FrameSequence,
    *,
    process_width: int | None,
    process_height: int | None,
    out_dir: Path,
) -> tuple[FrameSequence, float, float]:
    """Optionally resize frames for processing. Returns (seq, scale_x, scale_y)."""
    if process_width is None and process_height is None:
        return frames, 1.0, 1.0
    if process_width is None or process_height is None:
        raise PipelineIOError("Provide both --width and --height to resize, or neither")

    out_dir.mkdir(parents=True, exist_ok=True)
    new_paths: list[Path] = []
    for i, fp in enumerate(frames.frame_paths):
        img = Image.open(fp).convert("RGB")
        resized = img.resize((int(process_width), int(process_height)), Image.BILINEAR)
        dest = out_dir / f"{i:05d}.jpg"
        resized.save(dest, quality=95)
        new_paths.append(dest.resolve())
    sx = int(process_width) / float(frames.width)
    sy = int(process_height) / float(frames.height)
    return (
        FrameSequence(
            source=frames.source,
            frame_paths=new_paths,
            width=int(process_width),
            height=int(process_height),
            is_extracted_video=frames.is_extracted_video,
        ),
        sx,
        sy,
    )


def create_run_directory(
    output_root: str | Path,
    *,
    allow_existing: bool = False,
    run_id: str | None = None,
) -> tuple[Path, str]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    rid = run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    run_dir = root / rid
    if run_dir.exists():
        if not allow_existing:
            raise PipelineIOError(
                f"Run directory already exists: {run_dir}. "
                "Pass --allow-existing or choose another output root."
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir.resolve(), rid
