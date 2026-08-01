"""Default detector selection: prefer RT-DETR (closed-set), OWL-ViT optional fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DetectionCandidate:
    candidate_id: str
    box_xyxy: tuple[float, float, float, float]
    score: float | None
    label: str | None
    has_mask: bool


@dataclass(frozen=True)
class DetectorCapability:
    status: str  # AVAILABLE | BLOCKED_DEPENDENCY | UNAVAILABLE
    name: str
    detail: str
    supports_text_prompt: bool
    supports_class_agnostic: bool


class Detector(Protocol):
    def capabilities(self) -> DetectorCapability: ...

    def detect(
        self,
        frame_rgb,
        *,
        text_prompt: str | None = None,
    ) -> list[DetectionCandidate]: ...


class BlockedDetector:
    def capabilities(self) -> DetectorCapability:
        return DetectorCapability(
            status="BLOCKED_DEPENDENCY",
            name="none",
            detail="No detector installed",
            supports_text_prompt=False,
            supports_class_agnostic=False,
        )

    def detect(self, frame_rgb, *, text_prompt: str | None = None) -> list[DetectionCandidate]:
        return []


def get_default_detector() -> Detector:
    """Prefer RT-DETR for automatic common-object labels; OWL-ViT for text prompts."""
    try:
        from visionforge.detection.rtdetr import get_rtdetr_detector

        det = get_rtdetr_detector()
        if det.capabilities().status == "AVAILABLE":
            return det
    except Exception:
        pass
    try:
        from visionforge.detection.owlvit import get_owlvit_detector

        det = get_owlvit_detector()
        if det.capabilities().status == "AVAILABLE":
            return det
    except Exception:
        pass
    return BlockedDetector()


def get_text_prompt_detector() -> Detector:
    """Open-vocab path for optional text prompts outside COCO classes."""
    try:
        from visionforge.detection.owlvit import get_owlvit_detector

        det = get_owlvit_detector()
        if det.capabilities().status == "AVAILABLE":
            return det
    except Exception:
        pass
    return get_default_detector()
