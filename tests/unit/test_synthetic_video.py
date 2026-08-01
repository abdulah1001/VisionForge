"""Unit tests for synthetic video generation (no model download)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visionforge.preprocessing.synthetic_video import (
    SyntheticVideoSpec,
    generate_synthetic_video,
)


def test_generate_synthetic_video_deterministic(tmp_path: Path) -> None:
    spec = SyntheticVideoSpec(num_frames=12, width=160, height=120, seed=3)
    a = generate_synthetic_video(tmp_path / "a", spec)
    b = generate_synthetic_video(tmp_path / "b", spec)
    assert a.num_frames == 12
    assert len(a.frame_paths) == 12
    assert a.width == 160 and a.height == 120
    for pa, pb in zip(a.frame_paths, b.frame_paths, strict=True):
        assert np.array_equal(np.asarray(Image.open(pa)), np.asarray(Image.open(pb)))


def test_first_frame_prompt_inside_object(tmp_path: Path) -> None:
    result = generate_synthetic_video(
        tmp_path / "frames",
        SyntheticVideoSpec(num_frames=8, width=200, height=150, object_size=40),
    )
    x0, y0, x1, y1 = result.first_frame_box_xyxy
    px, py = result.first_frame_point_xy
    assert x0 < px < x1 and y0 < py < y1
    frame0 = np.asarray(Image.open(result.frame_paths[0]))
    # Object fill color is bright red-ish.
    assert int(frame0[py, px, 0]) > 200


def test_rejects_invalid_spec(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_synthetic_video(tmp_path, SyntheticVideoSpec(num_frames=1))
