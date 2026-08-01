"""Real end-to-end EdgeTAM → DINOv3 → MobileCLIP2 CUDA smoke test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_e2e_pipeline_real_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    from visionforge.pipeline.runner import PipelineConfig, run_pipeline
    from visionforge.preprocessing.synthetic_video import (
        SyntheticVideoSpec,
        generate_synthetic_video,
    )
    from visionforge.tracking import CapabilityStatus, select_tracker_backend

    fixtures = Path("D:/project/artifacts/e2e_fixtures")
    fixtures.mkdir(parents=True, exist_ok=True)
    video = generate_synthetic_video(
        fixtures / "frames_8x256",
        SyntheticVideoSpec(num_frames=8, width=256, height=256, object_size=40),
    )
    labels = ["a red circle", "a green rectangle", "a blue sky"]
    result = run_pipeline(
        PipelineConfig(
            input_path=video.frames_dir,
            box_xyxy=tuple(float(v) for v in video.first_frame_box_xyxy),
            text_labels=labels,
            output_root=Path("D:/project/artifacts/e2e_runs"),
            tracker="edgetam",
            max_frames=8,
            run_id=None,
        )
    )
    manifest = result.manifest
    assert manifest["selected_tracker_backend"] == "edgetam"
    assert manifest["real_cuda_inference"] is True
    assert manifest["mock_or_fallback_used"] is False
    assert manifest["offline_local_only"] is True
    assert manifest["frames"]["processed"] == 8
    assert manifest["frames"]["successful"] == 8
    sam_status = select_tracker_backend("sam31").capability().status
    assert sam_status in (
        CapabilityStatus.AVAILABLE_WSL2,
        CapabilityStatus.BLOCKED_WSL_MISSING,
        CapabilityStatus.BLOCKED_NATIVE_WINDOWS,
    )
    assert manifest["sam31_capability_status"] == sam_status.value
    # EdgeTAM path must remain available regardless of SAM31/WSL state.
    assert (
        select_tracker_backend("edgetam").capability().status
        == CapabilityStatus.AVAILABLE
    )

    run_dir = Path(manifest["outputs"]["run_dir"])
    assert len(list((run_dir / "masks").glob("mask_*.png"))) == 8
    assert len(list((run_dir / "overlays").glob("overlay_*.jpg"))) == 8
    assert len(list((run_dir / "crops").glob("crop_*.jpg"))) == 8

    emb = np.load(run_dir / "dinov3_embeddings.npy")
    assert emb.shape == (8, 384)
    assert np.isfinite(emb).all()

    img_f = np.load(run_dir / "mobileclip2_image_embeddings.npy")
    txt_f = np.load(run_dir / "mobileclip2_text_embeddings.npy")
    assert img_f.shape == (8, 512)
    assert txt_f.shape == (len(labels), 512)
    assert np.isfinite(img_f).all() and np.isfinite(txt_f).all()

    mclip = json.loads((run_dir / "mobileclip2_similarities.json").read_text(encoding="utf-8"))
    assert set(mclip["mean_scores"]) == set(labels)
    assert "highest_scoring_aggregate_label" in mclip

    # Required manifest fields
    for key in (
        "run_id",
        "selected_tracker_backend",
        "model_checkpoint_paths",
        "bounding_box_original_xyxy",
        "bounding_box_processed_xyxy",
        "stages",
        "timing",
    ):
        assert key in manifest
    assert "tracker" in manifest["stages"]
    assert "dinov3" in manifest["stages"]
    assert "mobileclip2" in manifest["stages"]
