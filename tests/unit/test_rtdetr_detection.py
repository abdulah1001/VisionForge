"""Unit tests for detection box ops and RT-DETR label mapping helpers."""
from __future__ import annotations

import numpy as np

from visionforge.detection.box_ops import box_iou, suppress_duplicates, title_case_label


def test_title_case_label():
    assert title_case_label("person") == "Person"
    assert title_case_label("sports_ball") == "Sports Ball"
    assert title_case_label("") == "Unknown object"
    assert title_case_label(None) == "Unknown object"


def test_suppress_duplicates_class_aware():
    boxes = [
        (0.0, 0.0, 10.0, 10.0),
        (1.0, 1.0, 11.0, 11.0),  # overlaps person
        (50.0, 50.0, 80.0, 80.0),  # dog
    ]
    scores = [0.9, 0.8, 0.95]
    labels = ["Person", "Person", "Dog"]
    keep = suppress_duplicates(boxes, scores, labels, iou_threshold=0.5)
    assert 0 in keep
    assert 1 not in keep
    assert 2 in keep


def test_box_iou_basic():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_rtdetr_id2label_person_dog_cat_regression():
    """Official COCO id2label must map common classes correctly when model present."""
    from pathlib import Path

    model_dir = Path(r"D:\caches\visionforge\models\rtdetr_r18vd")
    if not (model_dir / "config.json").is_file():
        return
    from transformers import RTDetrForObjectDetection

    model = RTDetrForObjectDetection.from_pretrained(str(model_dir), local_files_only=True)
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    vals = set(id2label.values())
    assert "person" in vals
    assert "dog" in vals
    assert "cat" in vals
    # Never invent labels from filename
    assert "thing" not in vals


def test_dilate_mask_limits():
    from visionforge.pipeline.remove_object import dilate_mask

    m = np.zeros((64, 64), dtype=np.uint8)
    m[20:30, 20:30] = 255
    out = dilate_mask(m, 3)
    assert out.sum() > m.sum()
    # Conservative: not entire frame
    assert out.mean() < 200
