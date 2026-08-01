"""Unit tests for pipeline geometry helpers."""
from __future__ import annotations

import numpy as np
import pytest

from visionforge.pipeline.geometry import (
    BoundingBox,
    GeometryError,
    extract_masked_crop,
    scale_bounding_box,
    validate_bounding_box,
)


def test_validate_bounding_box_ok() -> None:
    box = validate_bounding_box((10, 20, 40, 50), image_width=100, image_height=80)
    assert box.as_int_tuple() == (10, 20, 40, 50)


def test_validate_bounding_box_rejects_inverted() -> None:
    with pytest.raises(GeometryError):
        validate_bounding_box((40, 20, 10, 50), image_width=100, image_height=80)


def test_validate_bounding_box_outside() -> None:
    with pytest.raises(GeometryError):
        validate_bounding_box((-1, 0, 10, 10), image_width=100, image_height=80)


def test_scale_bounding_box() -> None:
    src = BoundingBox(10, 20, 30, 40)
    scaled = scale_bounding_box(
        src, src_width=100, src_height=100, dst_width=200, dst_height=50
    )
    assert scaled.x1 == pytest.approx(20)
    assert scaled.y1 == pytest.approx(10)
    assert scaled.x2 == pytest.approx(60)
    assert scaled.y2 == pytest.approx(20)


def test_extract_masked_crop_and_empty() -> None:
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    frame[10:20, 15:25] = (200, 10, 10)
    mask = np.zeros((40, 50), dtype=bool)
    mask[10:20, 15:25] = True
    crop = extract_masked_crop(frame, mask)
    assert crop.area_px == 100
    assert crop.crop_rgb.shape[0] == 10
    assert crop.crop_rgb.shape[1] == 10
    assert crop.crop_rgb[0, 0, 0] == 200

    empty = np.zeros((40, 50), dtype=bool)
    with pytest.raises(GeometryError, match="empty mask"):
        extract_masked_crop(frame, empty)
