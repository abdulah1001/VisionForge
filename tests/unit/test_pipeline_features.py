"""Unit tests for feature normalization and cosine similarity."""
from __future__ import annotations

import numpy as np
import pytest

from visionforge.pipeline.features import (
    FeatureError,
    l2_normalize,
    pairwise_cosine,
    summarize_similarities,
)


def test_l2_normalize_and_cosine() -> None:
    x = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    n = l2_normalize(x)
    assert n.shape == (2, 2)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    vs_first, consec = pairwise_cosine(x)
    assert vs_first.shape == (2,)
    assert np.isclose(vs_first[0], 1.0)
    assert np.isclose(consec[0], 1.0)


def test_nonfinite_rejected() -> None:
    with pytest.raises(FeatureError):
        l2_normalize(np.array([[np.nan, 1.0]], dtype=np.float32))


def test_summarize() -> None:
    s = summarize_similarities(np.array([0.2, 0.4, 0.6]))
    assert s["min"] == pytest.approx(0.2)
    assert s["max"] == pytest.approx(0.6)
    assert s["mean"] == pytest.approx(0.4)
