"""Conservative identity recovery using DINOv3 + detector proposals."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from visionforge.pipeline.quality import DEFAULT_THRESHOLDS

# Conservative recovery heuristics — not identity certainty.
DEFAULT_RECOVERY: dict[str, float] = {
    "search_expand_frac": 0.55,  # expand last box by this fraction of diagonal
    "min_appearance_sim": 0.55,
    "min_combined_score": 0.62,
    "ambiguity_gap": 0.08,  # top1-top2 must exceed this
    "max_size_ratio": 3.0,
    "max_center_jump_frac": 0.45,
}


@dataclass
class RecoveryDecision:
    frame_index: int
    recovered: bool
    lost: bool
    require_review: bool
    chosen_box: tuple[float, float, float, float] | None = None
    chosen_score: float | None = None
    candidates_considered: int = 0
    reasons: list[str] = field(default_factory=list)
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(1.0, (x2 - x1) * (y2 - y1))


def search_region(
    last_box: tuple[float, float, float, float],
    *,
    frame_w: int,
    frame_h: int,
    expand_frac: float,
) -> tuple[float, float, float, float]:
    diag = (frame_w**2 + frame_h**2) ** 0.5
    pad = diag * expand_frac
    x1, y1, x2, y2 = last_box
    return (
        max(0.0, x1 - pad),
        max(0.0, y1 - pad),
        min(float(frame_w), x2 + pad),
        min(float(frame_h), y2 + pad),
    )


def box_in_region(
    box: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
) -> bool:
    cx, cy = _box_center(box)
    rx1, ry1, rx2, ry2 = region
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def evaluate_recovery(
    *,
    frame_index: int,
    frame_w: int,
    frame_h: int,
    last_valid_box: tuple[float, float, float, float],
    reference_feature: np.ndarray,
    last_feature: np.ndarray | None,
    candidate_boxes: list[tuple[float, float, float, float]],
    candidate_features: list[np.ndarray | None],
    thresholds: dict[str, float] | None = None,
) -> RecoveryDecision:
    """Score detector candidates against appearance + spatial priors.

    Rejects weak and ambiguous matches. Never jumps silently when unsafe.
    """
    th = {**DEFAULT_RECOVERY, **(thresholds or {})}
    region = search_region(
        last_valid_box,
        frame_w=frame_w,
        frame_h=frame_h,
        expand_frac=th["search_expand_frac"],
    )
    last_area = _box_area(last_valid_box)
    last_c = _box_center(last_valid_box)
    diag = (frame_w**2 + frame_h**2) ** 0.5

    scored: list[dict[str, Any]] = []
    for i, box in enumerate(candidate_boxes):
        feat = candidate_features[i] if i < len(candidate_features) else None
        reasons: list[str] = []
        if not box_in_region(box, region):
            reasons.append("outside_search_region")
        area = _box_area(box)
        size_ratio = max(area, last_area) / max(1.0, min(area, last_area))
        if size_ratio > th["max_size_ratio"]:
            reasons.append("size_inconsistent")
        cx, cy = _box_center(box)
        jump = ((cx - last_c[0]) ** 2 + (cy - last_c[1]) ** 2) ** 0.5 / max(1.0, diag)
        if jump > th["max_center_jump_frac"]:
            reasons.append("center_jump")

        app = 0.0
        if feat is not None:
            app_ref = cosine_sim(feat, reference_feature)
            app_last = (
                cosine_sim(feat, last_feature) if last_feature is not None else app_ref
            )
            app = 0.65 * app_ref + 0.35 * app_last
        else:
            reasons.append("missing_appearance")

        if app < th["min_appearance_sim"]:
            reasons.append("weak_appearance")

        spatial = max(0.0, 1.0 - jump / max(th["max_center_jump_frac"], 1e-6))
        size_score = max(0.0, 1.0 - (size_ratio - 1.0) / max(th["max_size_ratio"] - 1.0, 1e-6))
        combined = 0.55 * app + 0.30 * spatial + 0.15 * size_score
        scored.append(
            {
                "index": i,
                "box": [float(x) for x in box],
                "appearance": round(app, 4),
                "spatial": round(spatial, 4),
                "size_score": round(size_score, 4),
                "combined": round(combined, 4),
                "rejected": bool(reasons),
                "reasons": reasons,
            }
        )

    accepted = [s for s in scored if not s["rejected"]]
    accepted.sort(key=lambda s: s["combined"], reverse=True)

    if not accepted:
        return RecoveryDecision(
            frame_index=frame_index,
            recovered=False,
            lost=True,
            require_review=True,
            candidates_considered=len(scored),
            reasons=["no_safe_candidate"],
            candidate_scores=scored,
        )

    top = accepted[0]
    if top["combined"] < th["min_combined_score"]:
        return RecoveryDecision(
            frame_index=frame_index,
            recovered=False,
            lost=True,
            require_review=True,
            candidates_considered=len(scored),
            reasons=["top_score_below_threshold"],
            candidate_scores=scored,
        )

    if len(accepted) > 1:
        gap = top["combined"] - accepted[1]["combined"]
        if gap < th["ambiguity_gap"]:
            return RecoveryDecision(
                frame_index=frame_index,
                recovered=False,
                lost=True,
                require_review=True,
                candidates_considered=len(scored),
                reasons=["ambiguous_similar_candidates"],
                candidate_scores=scored,
            )

    bx = tuple(float(x) for x in top["box"])  # type: ignore[misc]
    return RecoveryDecision(
        frame_index=frame_index,
        recovered=True,
        lost=False,
        require_review=False,
        chosen_box=(bx[0], bx[1], bx[2], bx[3]),
        chosen_score=float(top["combined"]),
        candidates_considered=len(scored),
        reasons=["recovered"],
        candidate_scores=scored,
    )


def recovery_thresholds_for_manifest() -> dict[str, Any]:
    return {
        "recovery": dict(DEFAULT_RECOVERY),
        "quality": dict(DEFAULT_THRESHOLDS),
        "note": (
            "Conservative heuristics for appearance/spatial recovery. "
            "Not a guaranteed identity proof."
        ),
    }
