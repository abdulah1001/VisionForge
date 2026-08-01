"""RT-DETR closed-set detector (Apache-2.0 PekingU weights via Transformers)."""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from visionforge.detection import DetectionCandidate, DetectorCapability
from visionforge.detection.box_ops import suppress_duplicates, title_case_label

DEFAULT_MODEL_ID = "PekingU/rtdetr_r18vd"
DEFAULT_MODEL_DIR = Path(r"D:\caches\visionforge\models\rtdetr_r18vd")
# Confidence below this → label as Unknown object (still selectable).
UNKNOWN_BELOW = 0.25
# Drop detections weaker than this entirely.
MIN_SCORE = 0.35
MAX_CANDIDATES = 24


class RtDetrDetector:
    """Real RT-DETR inference using official id2label mapping (COCO classes)."""

    def __init__(
        self,
        model_dir: Path | None = None,
        *,
        score_threshold: float = MIN_SCORE,
        unknown_below: float = UNKNOWN_BELOW,
        max_candidates: int = MAX_CANDIDATES,
        device: str | None = None,
    ) -> None:
        env_dir = os.environ.get("VISIONFORGE_RTDETR_DIR", DEFAULT_MODEL_DIR)
        self.model_dir = Path(model_dir or env_dir)
        self.score_threshold = float(score_threshold)
        self.unknown_below = float(unknown_below)
        self.max_candidates = int(max_candidates)
        self._device = device
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._id2label: dict[int, str] = {}

    def capabilities(self) -> DetectorCapability:
        if not self.model_dir.is_dir():
            return DetectorCapability(
                status="BLOCKED_DEPENDENCY",
                name="rtdetr",
                detail=f"RT-DETR weights not found at {self.model_dir}",
                supports_text_prompt=False,
                supports_class_agnostic=False,
            )
        try:
            self._ensure_loaded()
        except Exception as exc:
            return DetectorCapability(
                status="UNAVAILABLE",
                name="rtdetr",
                detail=f"RT-DETR load failed: {exc}"[:240],
                supports_text_prompt=False,
                supports_class_agnostic=False,
            )
        return DetectorCapability(
            status="AVAILABLE",
            name="rtdetr",
            detail="PekingU RT-DETR R18vd COCO · Apache-2.0",
            supports_text_prompt=False,
            supports_class_agnostic=False,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = device
            self._processor = RTDetrImageProcessor.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            self._model = RTDetrForObjectDetection.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            self._model.to(device)
            self._model.eval()
            raw = getattr(self._model.config, "id2label", None) or {}
            self._id2label = {int(k): str(v) for k, v in raw.items()}

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def detect(self, frame_rgb, *, text_prompt: str | None = None) -> list[DetectionCandidate]:
        # text_prompt ignored for closed-set RT-DETR (OWL-ViT remains optional fallback).
        _ = text_prompt
        self._ensure_loaded()
        assert self._model is not None and self._processor is not None
        import torch

        if isinstance(frame_rgb, np.ndarray):
            image = Image.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
        else:
            image = frame_rgb.convert("RGB")
        w, h = image.size
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([[h, w]], device=self._device)
        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=min(self.score_threshold, self.unknown_below),
        )[0]

        boxes_t = results["boxes"].detach().cpu().numpy()
        scores_t = results["scores"].detach().cpu().numpy()
        labels_t = results["labels"].detach().cpu().numpy()

        boxes: list[tuple[float, float, float, float]] = []
        scores: list[float] = []
        labels: list[str] = []
        for box, score, lab_id in zip(boxes_t, scores_t, labels_t, strict=False):
            sc = float(score)
            if sc < self.unknown_below:
                continue
            raw_name = self._id2label.get(int(lab_id))
            if raw_name is None:
                display = "Unknown object"
            elif sc < self.score_threshold:
                display = "Unknown object"
            else:
                display = title_case_label(raw_name)
            x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            x1 = max(0.0, min(float(w - 1), x1))
            y1 = max(0.0, min(float(h - 1), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append((x1, y1, x2, y2))
            scores.append(sc)
            labels.append(display)

        keep = suppress_duplicates(boxes, scores, labels, iou_threshold=0.55)
        keep = keep[: self.max_candidates]
        out: list[DetectionCandidate] = []
        for i in keep:
            digest = hashlib.sha1(
                f"{labels[i]}:{boxes[i][0]:.1f}:{boxes[i][1]:.1f}:{scores[i]:.4f}".encode()
            ).hexdigest()[:12]
            out.append(
                DetectionCandidate(
                    candidate_id=f"rtdetr-{digest}",
                    box_xyxy=boxes[i],
                    score=scores[i],
                    label=labels[i],
                    has_mask=False,
                )
            )
        return out


_DETECTOR: RtDetrDetector | None = None
_DET_LOCK = threading.Lock()


def get_rtdetr_detector() -> RtDetrDetector:
    global _DETECTOR
    with _DET_LOCK:
        if _DETECTOR is None:
            _DETECTOR = RtDetrDetector()
        return _DETECTOR
