"""Unit tests for GPU metric helpers (no checkpoint download)."""
from __future__ import annotations

from visionforge.observability import gpu_metrics


def test_snapshot_gpu_returns_dataclass_fields() -> None:
    snap = gpu_metrics.snapshot_gpu()
    data = snap.to_dict()
    assert "cuda_available" in data
    assert "allocated_bytes" in data
    assert "max_allocated_bytes" in data
    assert "max_reserved_bytes" in data
    assert isinstance(snap.cuda_available, bool)


def test_reset_and_empty_do_not_raise() -> None:
    gpu_metrics.reset_peak_stats()
    gpu_metrics.empty_cuda_cache()


def test_cuda_is_available_bool() -> None:
    assert isinstance(gpu_metrics.cuda_is_available(), bool)
