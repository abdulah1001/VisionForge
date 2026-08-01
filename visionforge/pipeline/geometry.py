"""Bounding-box validation, coordinate scaling, and mask crop helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class GeometryError(Exception):
    pass


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(self.x2)),
            int(round(self.y2)),
        )


def validate_bounding_box(
    box: tuple[float, float, float, float] | BoundingBox,
    *,
    image_width: int,
    image_height: int,
    min_side: float = 1.0,
) -> BoundingBox:
    if isinstance(box, BoundingBox):
        x1, y1, x2, y2 = box.as_tuple()
    else:
        if len(box) != 4:
            raise GeometryError("Bounding box must have 4 values: x1 y1 x2 y2")
        x1, y1, x2, y2 = (float(v) for v in box)

    if not all(np.isfinite([x1, y1, x2, y2])):
        raise GeometryError("Bounding box contains non-finite values")
    if x2 <= x1 or y2 <= y1:
        raise GeometryError("Bounding box must satisfy x2>x1 and y2>y1")
    if (x2 - x1) < min_side or (y2 - y1) < min_side:
        raise GeometryError(f"Bounding box sides must be >= {min_side}")
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        raise GeometryError(
            f"Bounding box {[x1, y1, x2, y2]} outside image "
            f"{image_width}x{image_height}"
        )
    # Require non-zero overlap with the image (already fully inside).
    if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
        raise GeometryError("Bounding box does not overlap the image")
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def scale_bounding_box(
    box: BoundingBox,
    *,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> BoundingBox:
    if src_width <= 0 or src_height <= 0 or dst_width <= 0 or dst_height <= 0:
        raise GeometryError("Invalid dimensions for box scaling")
    sx = dst_width / float(src_width)
    sy = dst_height / float(src_height)
    scaled = BoundingBox(
        x1=box.x1 * sx,
        y1=box.y1 * sy,
        x2=box.x2 * sx,
        y2=box.y2 * sy,
    )
    return validate_bounding_box(
        scaled,
        image_width=dst_width,
        image_height=dst_height,
    )


@dataclass(frozen=True)
class MaskCrop:
    crop_rgb: np.ndarray  # HxWx3 uint8, background zeroed outside mask
    x1: int
    y1: int
    x2: int
    y2: int
    area_px: int


def extract_masked_crop(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    padding: int = 0,
) -> MaskCrop:
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise GeometryError("frame_rgb must be HxWx3")
    m = np.asarray(mask).astype(bool)
    if m.shape != frame_rgb.shape[:2]:
        raise GeometryError(
            f"Mask shape {m.shape} does not match frame {frame_rgb.shape[:2]}"
        )
    if not m.any():
        raise GeometryError("Cannot extract crop from empty mask")

    ys, xs = np.where(m)
    y1 = max(int(ys.min()) - padding, 0)
    x1 = max(int(xs.min()) - padding, 0)
    y2 = min(int(ys.max()) + 1 + padding, frame_rgb.shape[0])
    x2 = min(int(xs.max()) + 1 + padding, frame_rgb.shape[1])

    region = frame_rgb[y1:y2, x1:x2].copy()
    region_mask = m[y1:y2, x1:x2]
    region[~region_mask] = 0
    return MaskCrop(
        crop_rgb=region.astype(np.uint8),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        area_px=int(m.sum()),
    )


def overlay_mask(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.45,
) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    out = frame_rgb.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32)
    out[m] = (1.0 - alpha) * out[m] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)
