"""Similar-object / ambiguous recovery regressions."""
from __future__ import annotations

import numpy as np

from visionforge.pipeline.quality import detect_sequence_issues, diagnose_mask
from visionforge.pipeline.recovery import evaluate_recovery


def test_ambiguous_near_equal_scores_require_review():
    ref = np.zeros(16, dtype=np.float32)
    ref[0] = 1.0
    a = ref.copy()
    b = ref.copy()
    # Two strong in-region candidates with near-equal appearance must not silently recover.
    boxes2 = [(41, 81, 81, 121), (55, 85, 95, 125)]
    decision2 = evaluate_recovery(
        frame_index=5,
        frame_w=320,
        frame_h=240,
        last_valid_box=(40, 80, 80, 120),
        reference_feature=ref,
        last_feature=ref,
        candidate_boxes=boxes2,
        candidate_features=[a, b],
        thresholds={
            "min_appearance_sim": 0.2,
            "min_combined_score": 0.2,
            "ambiguity_gap": 0.12,
            "max_center_jump_frac": 0.95,
            "search_expand_frac": 1.0,
            "max_size_ratio": 5.0,
        },
    )
    assert decision2.recovered is False
    assert decision2.require_review is True
    assert "ambiguous_similar_candidates" in decision2.reasons


def test_clear_winner_recovers_without_identity_jump_claim():
    ref = np.zeros(16, dtype=np.float32)
    ref[0] = 1.0
    good = ref.copy()
    bad = np.zeros(16, dtype=np.float32)
    bad[1] = 1.0
    decision = evaluate_recovery(
        frame_index=2,
        frame_w=320,
        frame_h=240,
        last_valid_box=(40, 80, 80, 120),
        reference_feature=ref,
        last_feature=ref,
        candidate_boxes=[(42, 82, 82, 122), (200, 80, 240, 120)],
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
    assert decision.chosen_box[0] < 100  # stayed with left object


def test_material_invalid_frames_not_clean_succeeded():
    diags = []
    for i in range(12):
        if i < 9:
            m = np.zeros((64, 64), dtype=bool)
            m[10:30, 10:30] = True
            diags.append(
                diagnose_mask(m, frame_index=i, frame_w=64, frame_h=64, box_xyxy=(8, 8, 32, 32))
            )
        else:
            diags.append(
                diagnose_mask(np.zeros((64, 64), dtype=bool), frame_index=i, frame_w=64, frame_h=64)
            )
    report = detect_sequence_issues(diags, frame_w=64, frame_h=64)
    assert report["empty_masks"] == 3
    assert report["recommended_status"] != "succeeded"
