"""Unit tests for tracking quality heuristics."""
from __future__ import annotations

import numpy as np

from visionforge.pipeline.quality import detect_sequence_issues, diagnose_mask


def test_empty_mask_invalid() -> None:
    m = np.zeros((64, 64), dtype=bool)
    d = diagnose_mask(m, frame_index=0, frame_w=64, frame_h=64, box_xyxy=(10, 10, 30, 30))
    assert d.empty and not d.valid


def test_valid_mask_inside_box() -> None:
    m = np.zeros((64, 64), dtype=bool)
    m[12:28, 12:28] = True
    d = diagnose_mask(m, frame_index=0, frame_w=64, frame_h=64, box_xyxy=(10, 10, 30, 30))
    assert d.valid and not d.empty


def test_partial_not_clean_success() -> None:
    diags = []
    for i in range(8):
        m = np.zeros((64, 64), dtype=bool)
        if i < 5:
            m[10:20, 10:20] = True
        diags.append(diagnose_mask(m, frame_index=i, frame_w=64, frame_h=64))
    report = detect_sequence_issues(diags, frame_w=64, frame_h=64)
    assert report["valid_masks"] == 5
    assert report["empty_masks"] == 3
    assert report["recommended_status"] != "succeeded"
