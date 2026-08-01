"""Encode annotated tracking videos (playable MP4) with optional audio."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


class AnnotateError(Exception):
    pass


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def overlay_mask_rgba(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (101, 221, 244),
    opacity: float = 0.45,
    draw_box: bool = True,
) -> np.ndarray:
    out = frame_rgb.astype(np.float32).copy()
    m = np.asarray(mask).astype(bool)
    if m.shape != frame_rgb.shape[:2]:
        raise AnnotateError(f"Mask shape {m.shape} != frame {frame_rgb.shape[:2]}")
    if m.any():
        for c in range(3):
            channel = out[:, :, c]
            channel[m] = (1.0 - opacity) * channel[m] + opacity * float(color[c])
            out[:, :, c] = channel
        if draw_box:
            ys, xs = np.where(m)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            out[y1 : y1 + 2, x1 : x2 + 1] = color
            out[y2 - 1 : y2 + 1, x1 : x2 + 1] = color
            out[y1 : y2 + 1, x1 : x1 + 2] = color
            out[y1 : y2 + 1, x2 - 1 : x2 + 1] = color
    return np.clip(out, 0, 255).astype(np.uint8)


def probe_audio(path: Path) -> dict:
    """Return audio stream metadata via ffprobe when available."""
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg or not Path(path).is_file():
        return {"audio_present": False, "streams": []}
    probe_name = "ffprobe.exe" if Path(ffmpeg).suffix == ".exe" else "ffprobe"
    ffprobe = str(Path(ffmpeg).with_name(probe_name))
    if not Path(ffprobe).is_file():
        # imageio-ffmpeg may only ship ffmpeg; try PATH ffprobe, else ffmpeg -i
        ffprobe = shutil.which("ffprobe") or ""
    if not ffprobe:
        return _probe_audio_via_ffmpeg(Path(path), ffmpeg)

    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0:
            return {"audio_present": False, "streams": [], "error": "ffprobe_failed"}
        data = json.loads(proc.stdout or "{}")
        streams = data.get("streams") or []
        audio = [s for s in streams if s.get("codec_type") == "audio"]
        video = [s for s in streams if s.get("codec_type") == "video"]
        return {
            "audio_present": bool(audio),
            "audio_codec": (audio[0].get("codec_name") if audio else None),
            "audio_duration": float(audio[0]["duration"])
            if audio and audio[0].get("duration")
            else None,
            "video_codec": (video[0].get("codec_name") if video else None),
            "width": int(video[0]["width"]) if video and video[0].get("width") else None,
            "height": int(video[0]["height"]) if video and video[0].get("height") else None,
            "streams": [
                {
                    "codec_type": s.get("codec_type"),
                    "codec_name": s.get("codec_name"),
                    "duration": s.get("duration"),
                }
                for s in streams
            ],
        }
    except Exception as exc:
        return {"audio_present": False, "streams": [], "error": str(exc)[:120]}


def _probe_audio_via_ffmpeg(path: Path, ffmpeg: str) -> dict:
    try:
        proc = subprocess.run(
            [ffmpeg, "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        err = (proc.stderr or "") + (proc.stdout or "")
        has_audio = "Audio:" in err
        codec = None
        if has_audio:
            for line in err.splitlines():
                if "Audio:" in line:
                    parts = line.split("Audio:")[-1].strip().split()
                    codec = parts[0].rstrip(",") if parts else None
                    break
        return {"audio_present": has_audio, "audio_codec": codec, "streams": []}
    except Exception:
        return {"audio_present": False, "streams": []}


def _mux_audio(
    *,
    video_only: Path,
    source: Path,
    output: Path,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
) -> dict:
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        return {
            "audio_preserved": False,
            "audio_present_in_source": False,
            "warning": "ffmpeg_unavailable",
        }
    src_probe = probe_audio(source)
    if not src_probe.get("audio_present"):
        shutil.copy2(video_only, output)
        return {
            "audio_preserved": False,
            "audio_present_in_source": False,
            "audio_present": False,
            "note": "Source has no audio stream.",
        }

    # Prefer stream copy for AAC (browser-native in MP4); otherwise transcode to AAC.
    codec = str(src_probe.get("audio_codec") or "").lower()
    copy_ok = codec in {"aac", "mp4a"}
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_only),
        "-ss",
        str(max(0.0, start_sec)),
        "-i",
        str(source),
    ]
    if duration_sec is not None and duration_sec > 0:
        cmd.extend(["-t", str(duration_sec)])
    cmd.extend(["-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy"])
    if copy_ok:
        cmd.extend(["-c:a", "copy"])
    else:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    cmd.extend(["-shortest", "-movflags", "+faststart", str(output)])

    proc = subprocess.run(cmd, capture_output=True, timeout=300, check=False)
    if proc.returncode != 0 or not output.is_file() or output.stat().st_size < 64:
        # Fall back: keep video-only, record warning
        shutil.copy2(video_only, output)
        return {
            "audio_preserved": False,
            "audio_present_in_source": True,
            "audio_present": False,
            "warning": "audio_mux_failed",
            "note": "Annotated video kept without audio after mux failure.",
        }

    out_probe = probe_audio(output)
    ok = bool(out_probe.get("audio_present"))
    return {
        "audio_preserved": ok,
        "audio_present_in_source": True,
        "audio_present": ok,
        "audio_codec": out_probe.get("audio_codec"),
        "method": "copy" if copy_ok else "aac_transcode",
        "warning": None if ok else "audio_missing_after_mux",
        "ffprobe": {
            "audio_present": ok,
            "audio_codec": out_probe.get("audio_codec"),
        },
    }


def encode_annotated_mp4(
    *,
    frame_paths: list[Path],
    mask_paths: list[Path | None],
    output_path: Path,
    fps: float = 24.0,
    opacity: float = 0.45,
    source_video: Path | None = None,
    audio_start_sec: float = 0.0,
) -> dict:
    if not frame_paths:
        raise AnnotateError("No frames to encode")
    if len(mask_paths) != len(frame_paths):
        raise AnnotateError("mask_paths length must match frame_paths")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(Image.open(frame_paths[0]).convert("RGB"))
    height, width = first.shape[:2]
    # Ensure even dimensions for yuv420p
    enc_w = width - (width % 2)
    enc_h = height - (height % 2)
    fps = float(fps) if fps and fps > 0 else 24.0

    ffmpeg = _ffmpeg_exe()
    writer = None
    use_ffmpeg = False
    tmp_video = output_path.with_suffix(".video_only.mp4")
    encode_target = tmp_video if source_video else output_path

    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{enc_w}x{enc_h}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(encode_target),
        ]
        writer = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        use_ffmpeg = True
    else:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(encode_target), fourcc, fps, (enc_w, enc_h))
        if not writer.isOpened():
            raise AnnotateError("OpenCV VideoWriter failed to open")

    written = 0
    try:
        for fp, mp in zip(frame_paths, mask_paths):
            rgb = np.asarray(Image.open(fp).convert("RGB"))
            if rgb.shape[0] != height or rgb.shape[1] != width:
                rgb = np.asarray(
                    Image.fromarray(rgb).resize((width, height), Image.BILINEAR)
                )
            if enc_w != width or enc_h != height:
                rgb = np.asarray(
                    Image.fromarray(rgb).resize((enc_w, enc_h), Image.BILINEAR)
                )
            if mp is not None and Path(mp).is_file():
                mask = np.asarray(Image.open(mp).convert("L")) > 127
                if mask.shape != (height, width):
                    mask = np.asarray(
                        Image.fromarray(mask.astype(np.uint8) * 255).resize(
                            (width, height), Image.NEAREST
                        )
                    ).astype(bool)
                if enc_w != width or enc_h != height:
                    mask = np.asarray(
                        Image.fromarray(mask.astype(np.uint8) * 255).resize(
                            (enc_w, enc_h), Image.NEAREST
                        )
                    ).astype(bool)
                frame = overlay_mask_rgba(rgb, mask, opacity=opacity)
            else:
                frame = rgb
            if use_ffmpeg:
                assert writer.stdin is not None
                writer.stdin.write(frame.tobytes())
            else:
                import cv2

                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            written += 1
    finally:
        if use_ffmpeg:
            if writer.stdin:
                writer.stdin.close()
            stderr = writer.stderr.read() if writer.stderr else b""
            rc = writer.wait(timeout=120)
            if rc != 0 and not encode_target.is_file():
                raise AnnotateError(
                    f"ffmpeg encode failed: {stderr[-500:].decode(errors='replace')}"
                )
        else:
            writer.release()

    if not encode_target.is_file() or encode_target.stat().st_size < 64:
        raise AnnotateError("Annotated video was not created or is empty")

    audio_meta: dict = {
        "audio_preserved": False,
        "audio_present": False,
        "audio_present_in_source": False,
    }
    duration_sec = round(written / fps, 3)
    if source_video and Path(source_video).is_file() and use_ffmpeg:
        audio_meta = _mux_audio(
            video_only=encode_target,
            source=Path(source_video),
            output=output_path,
            start_sec=audio_start_sec,
            duration_sec=duration_sec,
        )
        try:
            encode_target.unlink(missing_ok=True)
        except Exception:
            pass
    else:
        if encode_target != output_path:
            shutil.move(str(encode_target), str(output_path))
        if source_video is None:
            audio_meta["note"] = "No source video supplied for audio mux."
        elif not use_ffmpeg:
            audio_meta["warning"] = "audio_requires_ffmpeg"

    final_probe = probe_audio(output_path)
    # Never claim audio preserved unless probe confirms
    claimed = bool(audio_meta.get("audio_preserved")) and bool(
        final_probe.get("audio_present")
    )
    audio_meta["audio_preserved"] = claimed
    audio_meta["audio_present"] = bool(final_probe.get("audio_present"))
    if claimed:
        audio_meta["ffprobe_confirmed"] = True
    elif audio_meta.get("audio_present_in_source"):
        audio_meta.setdefault(
            "warning",
            "Source had audio but annotated output has no confirmed audio stream.",
        )

    return {
        "path": str(output_path.name),
        "width": enc_w,
        "height": enc_h,
        "fps": fps,
        "frames": written,
        "duration_sec": duration_sec,
        "size_bytes": output_path.stat().st_size,
        "codec": "libx264" if use_ffmpeg else "mp4v",
        "pixel_format": "yuv420p" if use_ffmpeg else "unknown",
        "faststart": bool(use_ffmpeg),
        "audio": audio_meta,
        "ffprobe": {
            "audio_present": final_probe.get("audio_present"),
            "audio_codec": final_probe.get("audio_codec"),
            "video_codec": final_probe.get("video_codec"),
            "width": final_probe.get("width"),
            "height": final_probe.get("height"),
        },
    }
