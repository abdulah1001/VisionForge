"""OWL-ViT open-vocabulary detector for VisionForge automatic candidates."""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from visionforge.detection import (
    DetectionCandidate,
    DetectorCapability,
)

# Default vocabulary for automatic (non-text) candidate proposals.
# These are detector query strings, not claimed semantic classifications of the scene.
DEFAULT_VOCAB = [
    "person",
    "face",
    "man",
    "woman",
    "child",
    "dog",
    "cat",
    "bird",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "boat",
    "airplane",
    "train",
    "traffic light",
    "bench",
    "backpack",
    "handbag",
    "suitcase",
    "umbrella",
    "bottle",
    "cup",
    "bowl",
    "banana",
    "apple",
    "orange",
    "pizza",
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "ball",
    "skateboard",
    "surfboard",
    "tennis racket",
    "sports ball",
    "frisbee",
    "kite",
    "baseball bat",
    "baseball glove",
    "object",
    "thing",
]

DEFAULT_MODEL_DIR = Path(r"D:\caches\visionforge\models\owlvit-base-patch32")


class OwlVitDetector:
    """Real OWL-ViT inference. Supports text prompts and vocabulary proposals."""

    def __init__(
        self,
        model_dir: Path | None = None,
        *,
        score_threshold: float = 0.03,
        max_candidates: int = 24,
        device: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.score_threshold = float(score_threshold)
        self.max_candidates = int(max_candidates)
        self._device = device
        self._processor = None
        self._model = None
        self._lock = threading.Lock()

    def capabilities(self) -> DetectorCapability:
        if not self.model_dir.is_dir():
            return DetectorCapability(
                status="BLOCKED_DEPENDENCY",
                name="owlvit-base-patch32",
                detail=f"OWL-ViT weights missing under {self.model_dir.name}",
                supports_text_prompt=True,
                supports_class_agnostic=True,
            )
        return DetectorCapability(
            status="AVAILABLE",
            name="owlvit-base-patch32",
            detail=(
                "google/owlvit-base-patch32 open-vocabulary detector "
                "(Apache-2.0). Automatic mode queries a COCO-like vocabulary; "
                "optional text prompts use the same model."
            ),
            supports_text_prompt=True,
            supports_class_agnostic=True,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir.is_dir():
            raise RuntimeError("OWL-ViT weights not installed")
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._processor = OwlViTProcessor.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        self._model = OwlViTForObjectDetection.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        self._model.to(device)
        self._model.eval()

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            try:
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass

    def detect(
        self,
        frame_rgb,
        *,
        text_prompt: str | None = None,
        unload_after: bool = False,
    ) -> list[DetectionCandidate]:
        import torch

        with self._lock:
            self._ensure_loaded()
            assert self._processor is not None and self._model is not None

            h, w = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
            image = Image.fromarray(np.asarray(frame_rgb).astype(np.uint8), mode="RGB")
            if text_prompt and text_prompt.strip():
                queries = [text_prompt.strip()]
                mode = "text_prompt"
            else:
                queries = list(DEFAULT_VOCAB)
                mode = "vocabulary"

            # OWL-ViT expects list-of-lists for batched text
            inputs = self._processor(
                text=[queries], images=image, return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = self._model(**inputs)

            target_sizes = torch.tensor([[h, w]], device=self._device)
            results = self._processor.post_process_grounded_object_detection(
                outputs=outputs,
                threshold=self.score_threshold,
                target_sizes=target_sizes,
                text_labels=[queries],
            )[0]

            boxes = results["boxes"].detach().float().cpu().numpy()
            scores = results["scores"].detach().float().cpu().numpy()
            # Prefer text labels from grounded post-process when present
            text_labels = results.get("text_labels") or results.get("labels")
            raw: list[tuple[float, DetectionCandidate]] = []
            for i, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = [float(v) for v in box]
                x1 = max(0.0, min(float(w), x1))
                y1 = max(0.0, min(float(h), y1))
                x2 = max(0.0, min(float(w), x2))
                y2 = max(0.0, min(float(h), y2))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                label = None
                if text_labels is not None:
                    try:
                        lab = text_labels[i]
                        label = str(lab) if lab is not None else None
                    except Exception:
                        label = None
                if label is None and "labels" in results:
                    lab_t = results["labels"][i]
                    if hasattr(lab_t, "item"):
                        li = int(lab_t.item())
                    else:
                        li = int(lab_t)
                    label = queries[li] if 0 <= li < len(queries) else None
                cid = hashlib.sha1(
                    f"{x1:.1f}:{y1:.1f}:{x2:.1f}:{y2:.1f}:{label}:{score:.4f}".encode()
                ).hexdigest()[:16]
                raw.append(
                    (
                        float(score),
                        DetectionCandidate(
                            candidate_id=cid,
                            box_xyxy=(x1, y1, x2, y2),
                            score=float(score),
                            label=label,
                            has_mask=False,
                        ),
                    )
                )

            raw.sort(key=lambda t: t[0], reverse=True)
            kept = _nms([c for _, c in raw], iou_thresh=0.5)
            _ = mode
            out = kept[: self.max_candidates]
        if unload_after:
            self.close()
        return out


def _iou(a: DetectionCandidate, b: DetectionCandidate) -> float:
    ax1, ay1, ax2, ay2 = a.box_xyxy
    bx1, by1, bx2, by2 = b.box_xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(cands: list[DetectionCandidate], *, iou_thresh: float) -> list[DetectionCandidate]:
    kept: list[DetectionCandidate] = []
    for c in cands:
        if all(_iou(c, k) < iou_thresh for k in kept):
            kept.append(c)
    return kept


_DETECTOR: OwlVitDetector | None = None
_DETECTOR_LOCK = threading.Lock()


def get_owlvit_detector() -> OwlVitDetector:
    global _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            _DETECTOR = OwlVitDetector()
        return _DETECTOR
