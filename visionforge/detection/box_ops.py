"""Box utilities: IoU and class-aware duplicate suppression."""
from __future__ import annotations

from typing import Sequence


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(x) for x in a)
    bx1, by1, bx2, by2 = (float(x) for x in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def suppress_duplicates(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    labels: list[str],
    *,
    iou_threshold: float = 0.55,
) -> list[int]:
    """Return keep indices (class-aware greedy NMS). Higher score first."""
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    suppressed = [False] * len(boxes)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if suppressed[j] or j == i:
                continue
            if labels[i] != labels[j]:
                continue
            if box_iou(boxes[i], boxes[j]) >= iou_threshold:
                suppressed[j] = True
    return keep


def title_case_label(raw: str | None) -> str:
    if not raw or not str(raw).strip():
        return "Unknown object"
    s = str(raw).strip().replace("_", " ")
    # COCO uses lowercase; present Title Case to users.
    return " ".join(w.capitalize() for w in s.split())
