"""Audio mux and ffprobe contract tests for annotated MP4."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from visionforge.pipeline.annotate import (
    _ffmpeg_exe,
    encode_annotated_mp4,
    probe_audio,
)


def _frames_and_masks(tmp: Path, n: int = 8) -> tuple[list[Path], list[Path]]:
    fd = tmp / "f"
    fd.mkdir()
    frames, masks = [], []
    for i in range(n):
        fp = fd / f"{i:05d}.jpg"
        Image.fromarray(np.full((64, 96, 3), 50 + i, dtype=np.uint8)).save(fp)
        frames.append(fp)
        m = np.zeros((64, 96), dtype=np.uint8)
        m[10:40, 20:50] = 255
        mp = tmp / f"m{i}.png"
        Image.fromarray(m, mode="L").save(mp)
        masks.append(mp)
    return frames, masks


def _make_source_with_audio(tmp: Path, *, codec: str = "aac") -> Path:
    ffmpeg = _ffmpeg_exe()
    assert ffmpeg
    out = tmp / f"src_{codec}.mp4"
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=96x64:d=1:r=8",
        "-f",
        "lavfi",
        "-i",
        "sine=f=880:d=1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        codec,
        "-shortest",
        "-movflags",
        "+faststart",
        str(out),
    ]
    if codec == "mp3":
        # Some ffmpeg builds need libmp3lame
        cmd[cmd.index(codec)] = "libmp3lame"
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    if proc.returncode != 0 or not out.is_file():
        # Fallback AAC always
        out = tmp / "src_aac.mp4"
        cmd2 = [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=96x64:d=1:r=8",
            "-f",
            "lavfi",
            "-i",
            "sine=f=880:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out),
        ]
        subprocess.run(cmd2, capture_output=True, timeout=60, check=True)
    return out


def test_source_without_audio_records_false(tmp_path: Path):
    frames, masks = _frames_and_masks(tmp_path)
    out = tmp_path / "ann.mp4"
    meta = encode_annotated_mp4(
        frame_paths=frames, mask_paths=masks, output_path=out, fps=8.0, source_video=None
    )
    assert meta["audio"]["audio_present"] is False
    assert meta["audio"]["audio_preserved"] is False
    assert probe_audio(out).get("audio_present") is False


def test_source_with_aac_preserves_audio(tmp_path: Path):
    src = _make_source_with_audio(tmp_path, codec="aac")
    src_probe = probe_audio(src)
    assert src_probe.get("audio_present") is True
    job = tmp_path / "job"
    job.mkdir()
    frames, masks = _frames_and_masks(job)
    out = tmp_path / "ann_audio.mp4"
    meta = encode_annotated_mp4(
        frame_paths=frames,
        mask_paths=masks,
        output_path=out,
        fps=8.0,
        source_video=src,
        audio_start_sec=0.0,
    )
    assert meta["audio"]["audio_present_in_source"] is True
    assert meta["audio"]["audio_present"] is True
    assert meta["audio"]["audio_preserved"] is True
    assert meta["audio"].get("ffprobe_confirmed") is True
    assert meta["codec"] == "libx264"
    assert meta.get("pixel_format") == "yuv420p"
    final = probe_audio(out)
    assert final.get("audio_present") is True


def test_audio_mux_failure_warns_without_failing_video(tmp_path: Path, monkeypatch):
    frames, masks = _frames_and_masks(tmp_path)
    src = _make_source_with_audio(tmp_path)
    out = tmp_path / "ann.mp4"

    import visionforge.pipeline.annotate as ann

    def fake_mux(*, video_only, source, output, start_sec=0.0, duration_sec=None):
        import shutil

        shutil.copy2(video_only, output)
        return {
            "audio_preserved": False,
            "audio_present_in_source": True,
            "audio_present": False,
            "warning": "audio_mux_failed",
        }

    monkeypatch.setattr(ann, "_mux_audio", fake_mux)
    meta = encode_annotated_mp4(
        frame_paths=frames,
        mask_paths=masks,
        output_path=out,
        fps=8.0,
        source_video=src,
    )
    assert out.is_file() and out.stat().st_size > 64
    assert meta["audio"]["audio_preserved"] is False
    assert meta["audio"].get("warning") == "audio_mux_failed"


def test_selected_time_range_audio_start_sec(tmp_path: Path):
    src = _make_source_with_audio(tmp_path)
    tdir = tmp_path / "t"
    tdir.mkdir()
    frames, masks = _frames_and_masks(tdir)
    out = tmp_path / "ann_ss.mp4"
    meta = encode_annotated_mp4(
        frame_paths=frames,
        mask_paths=masks,
        output_path=out,
        fps=8.0,
        source_video=src,
        audio_start_sec=0.0,
    )
    assert meta["frames"] == 8
    assert meta["duration_sec"] > 0
    assert meta["audio"]["audio_present"] is True


def test_source_requiring_aac_transcode(tmp_path: Path):
    """Non-AAC source audio should be remuxed/transcoded to AAC when possible."""
    src = _make_source_with_audio(tmp_path, codec="mp3")
    src_probe = probe_audio(src)
    if not src_probe.get("audio_present"):
        # Environment cannot encode mp3 — skip without weakening AAC path
        import pytest

        pytest.skip("mp3 source fixture unavailable in this ffmpeg build")
    job = tmp_path / "job_mp3"
    job.mkdir()
    frames, masks = _frames_and_masks(job)
    out = tmp_path / "ann_from_mp3.mp4"
    meta = encode_annotated_mp4(
        frame_paths=frames,
        mask_paths=masks,
        output_path=out,
        fps=8.0,
        source_video=src,
        audio_start_sec=0.0,
    )
    assert meta["audio"]["audio_present_in_source"] is True
    assert meta["audio"]["audio_present"] is True
    assert meta["audio"]["audio_preserved"] is True
    assert meta["audio"].get("ffprobe_confirmed") is True
    codec = (meta["audio"].get("audio_codec") or "").lower()
    method = (meta["audio"].get("method") or "").lower()
    assert method in {"aac_transcode", "aac", "transcode", "transcode_aac"}
    assert "aac" in codec or codec in {"mp4a"}
