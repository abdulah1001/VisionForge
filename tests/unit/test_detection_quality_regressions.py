"""Regression tests for prior root-cause failures and detector/recovery."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visionforge.api.schemas import JobSpec
from visionforge.pipeline.io import load_frame_sequence
from visionforge.pipeline.quality import (
    MaskDiagnostics,
    detect_sequence_issues,
    diagnose_mask,
)
from visionforge.pipeline.recovery import evaluate_recovery


def test_job_spec_rejects_unconfirmed_mask():
    with pytest.raises(Exception):
        JobSpec.model_validate(
            {
                "tracker": "edgetam",
                "box": [10, 10, 50, 50],
                "labels": [],
                "mask_confirmed": False,
            }
        )


def test_job_spec_full_mode_allows_null_max_frames():
    spec = JobSpec.model_validate(
        {
            "tracker": "edgetam",
            "box": [10, 10, 50, 50],
            "labels": ["cup"],
            "mask_confirmed": True,
            "analysis_mode": "full",
            "max_frames": None,
        }
    )
    assert spec.max_frames is None
    assert spec.analysis_mode == "full"


def test_job_spec_sampled_requires_max_frames():
    with pytest.raises(Exception):
        JobSpec.model_validate(
            {
                "tracker": "edgetam",
                "box": [10, 10, 50, 50],
                "mask_confirmed": True,
                "analysis_mode": "sampled",
                "max_frames": None,
            }
        )


def test_stale_sky_box_not_hardcoded_in_frontend_store():
    root = Path(__file__).resolve().parents[2]
    store_path = root / "frontend" / "src" / "store" / "studioStore.ts"
    remover_path = root / "frontend" / "src" / "store" / "removerStore.ts"
    text = ""
    if store_path.is_file():
        text += store_path.read_text(encoding="utf-8")
    if remover_path.is_file():
        text += remover_path.read_text(encoding="utf-8")
    assert text, "expected remover/studio store sources"
    assert "[20, 60, 60, 100]" not in text
    assert "20,60,60,100" not in text.replace(" ", "")
    assert "box: null" in text or "selectedId: null" in text


def test_five_of_eight_with_empty_not_succeeded():
    diags = []
    for i in range(8):
        if i < 5:
            m = np.zeros((64, 64), dtype=bool)
            m[20:40, 20:40] = True
            diags.append(
                diagnose_mask(m, frame_index=i, frame_w=64, frame_h=64, box_xyxy=(15, 15, 45, 45))
            )
        else:
            diags.append(
                diagnose_mask(
                    np.zeros((64, 64), dtype=bool),
                    frame_index=i,
                    frame_w=64,
                    frame_h=64,
                )
            )
    report = detect_sequence_issues(diags, frame_w=64, frame_h=64)
    assert report["empty_masks"] == 3
    assert report["valid_masks"] == 5
    assert report["recommended_status"] != "succeeded"


def test_quality_counts_from_diagnostics_not_file_globs():
    diags = [
        MaskDiagnostics(
            frame_index=0,
            area_px=100,
            empty=False,
            near_empty=False,
            full_frame=False,
            outside_frame=False,
            valid=True,
            reasons=[],
        ),
        MaskDiagnostics(
            frame_index=1,
            area_px=0,
            empty=True,
            near_empty=True,
            full_frame=False,
            outside_frame=False,
            valid=False,
            reasons=["empty_mask"],
        ),
    ]
    report = detect_sequence_issues(diags, frame_w=32, frame_h=32)
    assert report["valid_masks"] == 1
    assert report["empty_masks"] == 1
    assert report["invalid_masks"] == 1


def test_recovery_rejects_ambiguous_candidates():
    ref = np.ones(8, dtype=np.float32)
    ref = ref / np.linalg.norm(ref)
    feat_a = ref.copy()
    feat_b = ref.copy()
    decision = evaluate_recovery(
        frame_index=3,
        frame_w=200,
        frame_h=200,
        last_valid_box=(40, 40, 80, 80),
        reference_feature=ref,
        last_feature=ref,
        candidate_boxes=[(42, 42, 82, 82), (45, 45, 85, 85)],
        candidate_features=[feat_a, feat_b],
        thresholds={"ambiguity_gap": 0.2, "min_combined_score": 0.1, "min_appearance_sim": 0.1},
    )
    assert decision.recovered is False
    assert decision.require_review is True
    assert "ambiguous_similar_candidates" in decision.reasons


def test_recovery_accepts_clear_winner():
    ref = np.zeros(8, dtype=np.float32)
    ref[0] = 1.0
    good = ref.copy()
    bad = np.zeros(8, dtype=np.float32)
    bad[1] = 1.0
    decision = evaluate_recovery(
        frame_index=2,
        frame_w=200,
        frame_h=200,
        last_valid_box=(40, 40, 80, 80),
        reference_feature=ref,
        last_feature=ref,
        candidate_boxes=[(41, 41, 81, 81), (150, 150, 190, 190)],
        candidate_features=[good, bad],
        thresholds={
            "min_appearance_sim": 0.5,
            "min_combined_score": 0.5,
            "ambiguity_gap": 0.05,
            "max_center_jump_frac": 0.9,
            "search_expand_frac": 0.8,
        },
    )
    assert decision.recovered is True
    assert decision.chosen_box is not None


def test_full_mode_loads_all_frames(tmp_path: Path):
    d = tmp_path / "frames"
    d.mkdir()
    from PIL import Image

    for i in range(16):
        Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8) + i).save(d / f"{i:05d}.jpg")
    seq = load_frame_sequence(d, max_frames=None)
    assert len(seq.frame_paths) == 16


def test_sampled_mode_respects_max_frames(tmp_path: Path):
    d = tmp_path / "frames"
    d.mkdir()
    from PIL import Image

    for i in range(16):
        Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8) + i).save(d / f"{i:05d}.jpg")
    seq = load_frame_sequence(d, max_frames=8)
    assert len(seq.frame_paths) == 8


def test_start_frame_slices_sequence(tmp_path: Path):
    d = tmp_path / "frames"
    d.mkdir()
    from PIL import Image

    for i in range(10):
        Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8) + i).save(d / f"{i:05d}.jpg")
    seq = load_frame_sequence(d, start_frame=4, max_frames=3)
    assert len(seq.frame_paths) == 3


def test_stale_sky_box_not_hardcoded_in_frontend_store():
    root = Path(__file__).resolve().parents[2]
    store_path = root / "frontend" / "src" / "store" / "studioStore.ts"
    remover_path = root / "frontend" / "src" / "store" / "removerStore.ts"
    text = ""
    if store_path.is_file():
        text += store_path.read_text(encoding="utf-8")
    if remover_path.is_file():
        text += remover_path.read_text(encoding="utf-8")
    assert text, "expected remover/studio store sources"
    assert "2160" not in text or "box:" not in text.lower()
    # No baked fixture boxes
    assert "LEGACY_DEFAULT" in text or "selectedId: null" in text or "box: null" in text


def test_detector_available_when_weights_present():
    from visionforge.detection import get_default_detector

    det = get_default_detector()
    cap = det.capabilities()
    assert cap.status == "AVAILABLE"
    assert any(n in cap.name.lower() for n in ("rtdetr", "owlvit"))


@pytest.mark.slow
def test_detector_real_inference_smoke():
    from visionforge.detection.owlvit import OwlVitDetector

    det = OwlVitDetector(score_threshold=0.01, max_candidates=8)
    # Synthetic bright blob + dark background — may or may not fire "object"
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[80:160, 100:200] = 220
    cands = det.detect(frame, text_prompt="object")
    assert isinstance(cands, list)
    for c in cands:
        x1, y1, x2, y2 = c.box_xyxy
        assert x2 > x1 and y2 > y1
        assert 0 <= x1 < 320 and 0 <= x2 <= 320
