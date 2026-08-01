"""Annotated MP4 audio mux unit tests (ffmpeg when available)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from visionforge.pipeline.annotate import encode_annotated_mp4, probe_audio


def _write_frames(dir: Path, n: int = 8) -> list[Path]:
    paths = []
    for i in range(n):
        p = dir / f"{i:05d}.jpg"
        Image.fromarray(np.full((64, 96, 3), 40 + i * 5, dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


def test_encode_annotated_without_source_audio(tmp_path: Path):
    frame_dir = tmp_path / "f"
    frame_dir.mkdir()
    frames = _write_frames(frame_dir)
    masks = []
    for i, _fp in enumerate(frames):
        m = np.zeros((64, 96), dtype=np.uint8)
        m[10:40, 20:50] = 255
        mp = tmp_path / f"m{i}.png"
        Image.fromarray(m, mode="L").save(mp)
        masks.append(mp)
    out = tmp_path / "annotated.mp4"
    meta = encode_annotated_mp4(
        frame_paths=frames,
        mask_paths=masks,
        output_path=out,
        fps=8.0,
        source_video=None,
    )
    assert out.is_file()
    assert meta["frames"] == 8
    assert meta["audio"]["audio_preserved"] is False
    probe = probe_audio(out)
    assert meta["ffprobe"]["audio_present"] is False or probe.get("audio_present") is False


def test_probe_audio_missing_file():
    meta = probe_audio(Path("definitely_missing_xyz.mp4"))
    assert meta["audio_present"] is False
