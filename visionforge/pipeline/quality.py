"""Per-frame mask quality heuristics for VisionForge tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Conservative defaults — quality heuristics, not scientific certainty.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "empty_area_px": 1.0,
    "near_empty_area_px": 32.0,
    "near_empty_ratio": 0.0005,  # fraction of frame
    "full_frame_ratio": 0.92,
    "sudden_area_change": 4.0,  # max(a,b)/min(a,b)
    "sudden_centroid_jump_frac": 0.35,  # of diagonal
    "implausible_size_change": 6.0,
    "min_box_overlap_ratio": 0.05,
    "dino_similarity_drop": 0.35,  # absolute drop vs reference
    "consecutive_invalid_for_partial": 3,
    "min_valid_ratio_for_succeeded": 0.85,
    "min_valid_ratio_for_partial": 0.15,
}


@dataclass
class MaskDiagnostics:
    frame_index: int
    area_px: int
    empty: bool
    near_empty: bool
    full_frame: bool
    outside_frame: bool
    valid: bool
    reasons: list[str] = field(default_factory=list)
    centroid: tuple[float, float] | None = None
    bbox_xyxy: tuple[int, int, int, int] | None = None
    mask_to_box_ratio: float | None = None
    overlaps_box: bool | None = None


def diagnose_mask(
    mask: np.ndarray,
    *,
    frame_index: int,
    frame_w: int,
    frame_h: int,
    box_xyxy: tuple[float, float, float, float] | None = None,
    thresholds: dict[str, float] | None = None,
) -> MaskDiagnostics:
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    m = np.asarray(mask)
    if m.ndim != 2:
        return MaskDiagnostics(
            frame_index=frame_index,
            area_px=0,
            empty=True,
            near_empty=True,
            full_frame=False,
            outside_frame=True,
            valid=False,
            reasons=["corrupted_shape"],
        )
    if m.shape != (frame_h, frame_w):
        return MaskDiagnostics(
            frame_index=frame_index,
            area_px=0,
            empty=True,
            near_empty=True,
            full_frame=False,
            outside_frame=True,
            valid=False,
            reasons=[f"shape_mismatch:{m.shape}"],
        )

    binary = m.astype(bool)
    area = int(binary.sum())
    frame_area = max(1, frame_w * frame_h)
    reasons: list[str] = []
    empty = area < th["empty_area_px"]
    near_empty = (not empty) and (
        area < th["near_empty_area_px"] or (area / frame_area) < th["near_empty_ratio"]
    )
    full_frame = (area / frame_area) >= th["full_frame_ratio"]
    outside_frame = False  # mask is frame-sized; check edges later

    centroid = None
    bbox = None
    if area > 0:
        ys, xs = np.where(binary)
        centroid = (float(xs.mean()), float(ys.mean()))
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    mask_to_box_ratio = None
    overlaps_box = None
    if box_xyxy is not None and area > 0 and bbox is not None:
        bx1, by1, bx2, by2 = box_xyxy
        box_area = max(1.0, (bx2 - bx1) * (by2 - by1))
        # Intersection of mask bbox with selection box (coarse)
        ix1 = max(bbox[0], int(bx1))
        iy1 = max(bbox[1], int(by1))
        ix2 = min(bbox[2], int(bx2))
        iy2 = min(bbox[3], int(by2))
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        # Pixel overlap inside box
        x1i, x2i = max(0, int(bx1)), min(frame_w, int(bx2))
        y1i, y2i = max(0, int(by1)), min(frame_h, int(by2))
        inside = int(binary[y1i:y2i, x1i:x2i].sum()) if x2i > x1i and y2i > y1i else 0
        overlaps_box = (inside / area) >= th["min_box_overlap_ratio"] if area else False
        mask_to_box_ratio = area / box_area
        if not overlaps_box:
            reasons.append("mask_outside_selected_box")
        _ = inter  # reserved for finer diagnostics

    if empty:
        reasons.append("empty_mask")
    if near_empty:
        reasons.append("near_empty_mask")
    if full_frame:
        reasons.append("full_frame_mask")

    valid = not empty and not near_empty and not full_frame and (
        overlaps_box is not False
    )
    return MaskDiagnostics(
        frame_index=frame_index,
        area_px=area,
        empty=empty,
        near_empty=near_empty,
        full_frame=full_frame,
        outside_frame=outside_frame,
        valid=valid,
        reasons=reasons,
        centroid=centroid,
        bbox_xyxy=bbox,
        mask_to_box_ratio=mask_to_box_ratio,
        overlaps_box=overlaps_box,
    )


def detect_sequence_issues(
    diagnostics: list[MaskDiagnostics],
    *,
    frame_w: int,
    frame_h: int,
    dino_vs_first: list[float] | None = None,
    valid_frame_indices: list[int] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    diagonal = float(np.hypot(frame_w, frame_h))
    empty_count = sum(1 for d in diagnostics if d.empty)
    near_empty_count = sum(1 for d in diagnostics if d.near_empty)
    invalid_count = sum(1 for d in diagnostics if not d.valid)
    valid_count = sum(1 for d in diagnostics if d.valid)
    total = len(diagnostics)
    drift_frames: list[int] = []
    warnings: list[str] = []

    prev_valid: MaskDiagnostics | None = None
    for d in diagnostics:
        if not d.valid or d.centroid is None or d.area_px <= 0:
            continue
        if prev_valid is not None and prev_valid.centroid is not None:
            area_ratio = max(d.area_px, prev_valid.area_px) / max(
                1, min(d.area_px, prev_valid.area_px)
            )
            if area_ratio >= th["sudden_area_change"]:
                drift_frames.append(d.frame_index)
                warnings.append(f"sudden_area_change:frame_{d.frame_index}")
            if area_ratio >= th["implausible_size_change"]:
                warnings.append(f"implausible_size_change:frame_{d.frame_index}")
            jump = np.hypot(
                d.centroid[0] - prev_valid.centroid[0],
                d.centroid[1] - prev_valid.centroid[1],
            )
            if jump >= th["sudden_centroid_jump_frac"] * diagonal:
                drift_frames.append(d.frame_index)
                warnings.append(f"sudden_displacement:frame_{d.frame_index}")
        prev_valid = d

    if dino_vs_first and valid_frame_indices:
        for i, sim in enumerate(dino_vs_first):
            if i == 0:
                continue
            drop = float(dino_vs_first[0]) - float(sim)
            if drop >= th["dino_similarity_drop"]:
                fi = valid_frame_indices[i] if i < len(valid_frame_indices) else i
                drift_frames.append(int(fi))
                warnings.append(f"dino_similarity_drop:frame_{fi}")

    # Longest consecutive invalid run
    longest_fail = 0
    run = 0
    for d in diagnostics:
        if not d.valid:
            run += 1
            longest_fail = max(longest_fail, run)
        else:
            run = 0

    valid_ratio = valid_count / max(1, total)
    material_failure = (
        empty_count > 0
        or invalid_count > 0
        or len(set(drift_frames)) > 0
        or valid_ratio < th["min_valid_ratio_for_succeeded"]
    )

    if total == 0 or valid_count == 0:
        status = "failed"
    elif valid_ratio >= th["min_valid_ratio_for_succeeded"] and not material_failure:
        status = "succeeded"
    elif valid_ratio >= th["min_valid_ratio_for_succeeded"] and (
        len(set(drift_frames)) > 0 or near_empty_count > 0
    ):
        status = "review_required"
    elif valid_ratio >= th["min_valid_ratio_for_partial"]:
        # Has usable frames but material gaps — 5/8 empty case lands here
        if material_failure:
            status = "review_required" if valid_ratio >= 0.5 else "partial"
        else:
            status = "partial"
    else:
        status = "failed"

    # Explicit: 5 valid of 8 with empty masks must not be clean succeeded
    if invalid_count > 0 and status == "succeeded":
        status = "review_required"

    return {
        "thresholds": th,
        "note": (
            "Quality heuristics only — not guaranteed identity or tracking truth."
        ),
        "total_frames": total,
        "valid_masks": valid_count,
        "invalid_masks": invalid_count,
        "empty_masks": empty_count,
        "near_empty_masks": near_empty_count,
        "suspected_drift_frames": sorted(set(drift_frames)),
        "suspected_drift_count": len(set(drift_frames)),
        "longest_failure_sequence": longest_fail,
        "valid_ratio": round(valid_ratio, 4),
        "recommended_status": status,
        "warnings": warnings[:100],
        "frames_requiring_review": sorted(
            {
                *[d.frame_index for d in diagnostics if not d.valid],
                *drift_frames,
            }
        )[:200],
        "per_frame": [
            {
                "frame_index": d.frame_index,
                "area_px": d.area_px,
                "valid": d.valid,
                "empty": d.empty,
                "near_empty": d.near_empty,
                "full_frame": d.full_frame,
                "reasons": d.reasons,
                "centroid": d.centroid,
                "bbox_xyxy": d.bbox_xyxy,
                "mask_to_box_ratio": d.mask_to_box_ratio,
                "overlaps_box": d.overlaps_box,
            }
            for d in diagnostics
        ],
    }
