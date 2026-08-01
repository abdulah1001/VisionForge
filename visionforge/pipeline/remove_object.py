"""Object-removal pipeline: track masks → ProPainter → cleaned MP4."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from visionforge.observability.gpu_metrics import empty_cuda_cache
from visionforge.pipeline.annotate import _ffmpeg_exe, _mux_audio
from visionforge.pipeline.geometry import scale_bounding_box, validate_bounding_box
from visionforge.pipeline.io import (
    create_run_directory,
    load_frame_sequence,
    maybe_resize_frames,
)
from visionforge.pipeline.manifest import write_json
from visionforge.pipeline.progress import open_progress
from visionforge.pipeline.quality import diagnose_mask
from visionforge.tracking import require_available, select_tracker_backend

PROPAINTER_MODULE = "visionforge.inpaint.propainter_runner"


@dataclass
class RemoveConfig:
    input_path: Path
    box_xyxy: tuple[float, float, float, float]
    output_root: Path
    tracker: str = "edgetam"
    selected_label: str | None = None
    max_frames: int | None = None
    max_side: int = 720
    quality_mode: str = "standard"  # standard | high
    mask_dilate_px: int = 3
    chunk_size: int = 48
    overlap: int = 10
    fp16: bool = True
    allow_existing: bool = False
    run_id: str | None = None
    progress_file: Path | None = None
    max_duration_sec: float = 60.0


@dataclass
class RemoveResult:
    run_dir: Path
    manifest_path: Path
    cleaned_mp4: Path
    manifest: dict = field(default_factory=dict)


class RemoveError(Exception):
    pass


USER_STAGE = {
    "preparing": "Preparing video",
    "tracking": "Following selected object",
    "inpainting": "Rebuilding background",
    "encoding": "Finalizing result",
    "completed": "Done",
}


def _fit_side(w: int, h: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, float(max_side) / float(max(w, h)))
    nw = max(2, int(round(w * scale / 2) * 2))
    nh = max(2, int(round(h * scale / 2) * 2))
    return nw, nh


def _emit(progress, stage: str, *, frac: float = 1.0, completed=None, total=None, message=None):
    bands = {
        "preparing": (0.0, 10.0),
        "tracking": (10.0, 40.0),
        "inpainting": (40.0, 90.0),
        "encoding": (90.0, 100.0),
        "completed": (100.0, 100.0),
    }
    lo, hi = bands.get(stage, (0.0, 100.0))
    pct = lo + (hi - lo) * max(0.0, min(1.0, frac))
    progress.emit(
        stage,
        completed=completed,
        total=total,
        overall_percent=100.0 if stage == "completed" else min(99.0, round(pct, 2)),
        message=message or USER_STAGE.get(stage),
        extra={"user_stage": USER_STAGE.get(stage, stage)},
    )


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if px <= 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(m, k, iterations=1)


def _probe_fps(path: Path) -> float:
    try:
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        cap.release()
        if fps > 1:
            return min(fps, 30.0)
    except Exception:
        pass
    return 24.0


def _encode_frames_mp4(frame_paths: list[Path], output: Path, fps: float) -> None:
    if not frame_paths:
        raise RemoveError("No frames to encode")
    first = np.asarray(Image.open(frame_paths[0]).convert("RGB"))
    h, w = first.shape[:2]
    enc_w, enc_h = w - (w % 2), h - (h % 2)
    fps = float(fps) if fps and fps > 0 else 24.0
    ffmpeg = _ffmpeg_exe()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not ffmpeg:
        raise RemoveError("ffmpeg unavailable for cleaned MP4 encode")
    cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{enc_w}x{enc_h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "15",
        "-movflags",
        "+faststart",
        str(output),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        for p in frame_paths:
            arr = np.asarray(Image.open(p).convert("RGB"))
            if arr.shape[0] != h or arr.shape[1] != w:
                arr = np.asarray(Image.fromarray(arr).resize((w, h), Image.Resampling.BILINEAR))
            if enc_w != w or enc_h != h:
                arr = arr[:enc_h, :enc_w]
            proc.stdin.write(arr.tobytes())
        proc.stdin.close()
        err = proc.stderr.read() if proc.stderr else b""
        rc = proc.wait(timeout=600)
    except Exception as exc:
        proc.kill()
        raise RemoveError(f"encode failed: {exc}") from exc
    if rc != 0 or not output.is_file():
        raise RemoveError(f"ffmpeg encode failed: {err[-400:]!r}")


def _run_propainter(
    *,
    frames_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    chunk_size: int,
    overlap: int,
    fp16: bool,
    progress_file: Path | None,
) -> None:
    cmd = [
        sys.executable, "-m", PROPAINTER_MODULE,
        "--frames-dir", str(frames_dir),
        "--masks-dir", str(masks_dir),
        "--output-dir", str(output_dir),
        "--width", str(width),
        "--height", str(height),
        "--chunk-size", str(chunk_size),
        "--overlap", str(overlap),
    ]
    if fp16:
        cmd.append("--fp16")
    if progress_file is not None:
        cmd.extend(["--progress-file", str(progress_file)])
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RemoveError("Inpainting timed out") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[-800:]
        low = err.lower()
        if "out of memory" in low or "outofmemoryerror" in low:
            raise RemoveError("GPU_OOM:" + err[:200])
        raise RemoveError(f"Inpainting failed: {err}")
    if not list(output_dir.glob("frame_*.png")) and not list(output_dir.glob("*.png")):
        raise RemoveError("Inpainting produced no frames")


def run_remove_object(config: RemoveConfig) -> RemoveResult:
    progress = open_progress(config.progress_file)
    _emit(progress, "preparing", frac=0.05)

    run_dir, run_id = create_run_directory(
        config.output_root,
        allow_existing=config.allow_existing,
        run_id=config.run_id,
    )
    extract_dir = run_dir / "video_frames"
    frames_work = run_dir / "frames_proc"
    masks_dir = run_dir / "masks"
    masks_inpaint = run_dir / "masks_inpaint"
    inpaint_out = run_dir / "inpaint_frames"
    pp_frames = run_dir / "pp_frames"
    for d in (extract_dir, frames_work, masks_dir, masks_inpaint, inpaint_out, pp_frames):
        d.mkdir(parents=True, exist_ok=True)

    # Honor client max_side (including Optimized ~640). High quality floors at 960 only
    # when the client asked for at least that; never force 720 over a lower request.
    if config.quality_mode == "high":
        max_side = max(960, int(config.max_side))
    else:
        max_side = max(320, int(config.max_side))
    src_path = Path(config.input_path)
    fps = _probe_fps(src_path) if src_path.is_file() else 24.0
    # Prefer full clip up to max_duration; only truncate if max_frames set.
    max_frames = config.max_frames
    if config.max_duration_sec and fps > 0:
        by_dur = max(2, int(config.max_duration_sec * fps))
        if max_frames is None:
            max_frames = by_dur
        else:
            max_frames = min(int(max_frames), by_dur)
    # Hard safety cap (~60s @ 30fps). Prefer duration limit above.
    if max_frames is not None:
        max_frames = min(int(max_frames), 1800)

    seq = load_frame_sequence(
        config.input_path,
        max_frames=max_frames,
        start_frame=0,
        extract_dir=extract_dir,
    )
    original_box = validate_bounding_box(
        config.box_xyxy,
        image_width=seq.width,
        image_height=seq.height,
    )
    # Long clips: keep resolution modest so ProPainter + tracker fit in 8GB.
    n_est = len(seq.frame_paths)
    if n_est > 400 and max_side > 640:
        max_side = 640
    elif n_est > 240 and max_side > 720:
        max_side = 720
    pw, ph = _fit_side(seq.width, seq.height, max_side)
    proc_seq, _sx, _sy = maybe_resize_frames(
        seq,
        process_width=pw,
        process_height=ph,
        out_dir=frames_work,
    )
    proc_box = scale_bounding_box(
        original_box,
        src_width=seq.width,
        src_height=seq.height,
        dst_width=proc_seq.width,
        dst_height=proc_seq.height,
    )
    n = len(proc_seq.frame_paths)
    if n < 2:
        raise RemoveError("Need at least 2 frames for removal")

    _emit(progress, "preparing", frac=1.0, completed=n, total=n)

    backend = select_tracker_backend(config.tracker)
    require_available(backend)
    _emit(progress, "tracking", frac=0.0, completed=0, total=n)
    backend.load()
    try:
        try:
            track = backend.track(
                frames_work,
                box_xyxy=proc_box.as_tuple(),
                frame_width=proc_seq.width,
                frame_height=proc_seq.height,
                object_id=1,
            )
        except Exception as exc:
            low = f"{type(exc).__name__}: {exc}".lower()
            if (
                "out of memory" in low
                or "outofmemory" in low
                or str(exc).startswith("GPU_OOM")
            ):
                raise RemoveError(
                    "GPU_OOM:This video is too large for the available GPU. "
                    "Optimize it to 640p and try again."
                ) from exc
            raise
    finally:
        backend.close()
        empty_cuda_cache()

    valid = 0
    lost = 0
    for i, fm in enumerate(track.frames):
        mask = np.asarray(fm.mask).astype(bool)
        diag = diagnose_mask(
            mask,
            frame_index=i,
            frame_w=proc_seq.width,
            frame_h=proc_seq.height,
            box_xyxy=proc_box.as_tuple(),
        )
        if diag.valid and not diag.empty:
            valid += 1
        else:
            lost += 1
        raw = (mask.astype(np.uint8) * 255)
        Image.fromarray(raw).save(masks_dir / f"mask_{i:05d}.png")
        Image.fromarray(dilate_mask(raw, config.mask_dilate_px)).save(
            masks_inpaint / f"{i:05d}.png"
        )
        # ProPainter frame copy
        shutil.copy2(proc_seq.frame_paths[i], pp_frames / f"{i:05d}.jpg")
        if i % 4 == 0 or i == n - 1:
            _emit(progress, "tracking", frac=(i + 1) / n, completed=i + 1, total=n)

    if valid / max(1, n) < 0.5:
        raise RemoveError(
            "TRACK_UNRELIABLE:We couldn't follow this object reliably. "
            "Try selecting it from a clearer frame."
        )

    _emit(progress, "inpainting", frac=0.05, completed=0, total=n)
    chunk = config.chunk_size if config.quality_mode == "high" else min(config.chunk_size, 40)
    try:
        _run_propainter(
            frames_dir=pp_frames,
            masks_dir=masks_inpaint,
            output_dir=inpaint_out,
            width=proc_seq.width,
            height=proc_seq.height,
            chunk_size=chunk,
            overlap=config.overlap,
            fp16=config.fp16,
            progress_file=config.progress_file,
        )
    except RemoveError as exc:
        if str(exc).startswith("GPU_OOM"):
            empty_cuda_cache()
            try:
                _run_propainter(
                    frames_dir=pp_frames,
                    masks_dir=masks_inpaint,
                    output_dir=inpaint_out,
                    width=proc_seq.width,
                    height=proc_seq.height,
                    chunk_size=max(10, chunk // 2),
                    overlap=max(4, config.overlap // 2),
                    fp16=True,
                    progress_file=config.progress_file,
                )
            except RemoveError as exc2:
                raise RemoveError(
                    "GPU_OOM:This video is too large for the available GPU. "
                    "Optimize it to 720p and try again."
                ) from exc2
        else:
            raise

    _emit(progress, "inpainting", frac=1.0, completed=n, total=n)
    empty_cuda_cache()

    painted = sorted(inpaint_out.glob("frame_*.png"))
    if len(painted) < n:
        painted = sorted(
            p for p in inpaint_out.rglob("*.png")
            if p.stem.isdigit() or p.name.startswith("frame_")
        )
    if len(painted) < n:
        raise RemoveError(
            "INPAINT_INCOMPLETE:Object removal could not be completed. "
            "Your original video was not changed."
        )
    painted = painted[:n]

    _emit(progress, "encoding", frac=0.2)
    video_only = run_dir / "cleaned_video_only.mp4"
    cleaned = run_dir / "cleaned.mp4"
    _encode_frames_mp4(painted, video_only, fps)
    audio_meta = {"audio_preserved": False}
    if src_path.is_file():
        audio_meta = _mux_audio(video_only=video_only, source=src_path, output=cleaned)
    else:
        shutil.copy2(video_only, cleaned)

    _emit(progress, "completed", frac=1.0)
    manifest = {
        "status": "succeeded",
        "operation": "remove_object",
        "run_id": run_id,
        "tracker": config.tracker,
        "selected_label": config.selected_label,
        "frames": {"processed": n, "successful": valid, "lost_masks": lost},
        "processing": {
            "width": proc_seq.width,
            "height": proc_seq.height,
            "fps": fps,
            "quality_mode": config.quality_mode,
            "max_side": max_side,
        },
        "inpainting": {
            "engine": "propainter",
            "licence": "NTU S-Lab License 1.0 NON-COMMERCIAL",
            "chunk_size": chunk,
            "mask_dilate_px": config.mask_dilate_px,
        },
        "audio": audio_meta,
        "artifacts": {"cleaned_mp4": "cleaned.mp4"},
        "mock_or_fallback_used": False,
        "user_message": "Object removed successfully.",
    }
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return RemoveResult(
        run_dir=run_dir,
        manifest_path=manifest_path,
        cleaned_mp4=cleaned,
        manifest=manifest,
    )
