"""Real SAM 3.1 Object Multiplex integration smoke test (explicit opt-in)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _auth_ready() -> bool:
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return False


@pytest.mark.skipif(
    not _auth_ready(),
    reason="Hugging Face auth not available (hf auth login required)",
)
def test_sam31_video_integration(tmp_path: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    from visionforge.cli.sam31_smoke_test import run_smoke_test

    # Keep artifacts on D: even if pytest tmp is elsewhere.
    artifacts = Path(os.environ.get("VISIONFORGE_ARTIFACTS_DIR", "D:/project/artifacts"))
    out = artifacts / "sam31_integration"
    metrics = run_smoke_test(
        artifacts_dir=out,
        num_frames=12,
        width=320,
        height=240,
        object_id=1,
        use_box=True,
        save_overlays=True,
    )
    assert metrics["status"] == "ok"
    assert metrics["num_processed_frames"] == metrics["num_input_frames"]
    assert metrics["masks_saved"] == metrics["num_input_frames"]
    assert metrics.get("checkpoint_verified_name") == "sam3.1_multiplex.pt"
    assert metrics["peak_cuda_allocated_mb"] is not None
    metrics_path = Path(metrics["output_paths"]["metrics_path"])
    assert metrics_path.is_file()
    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert loaded["status"] == "ok"
