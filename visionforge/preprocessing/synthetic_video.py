"""Deterministic synthetic short-video fixtures for SAM 3.1 smoke tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class SyntheticVideoSpec:
    num_frames: int = 16
    width: int = 320
    height: int = 240
    object_size: int = 48
    start_x: int = 40
    start_y: int = 80
    velocity_x: int = 12
    velocity_y: int = 3
    seed: int = 7


@dataclass(frozen=True)
class SyntheticVideoResult:
    frames_dir: Path
    frame_paths: list[Path]
    width: int
    height: int
    num_frames: int
    # Absolute XYXY box on frame 0 (inclusive of object disk bounds).
    first_frame_box_xyxy: tuple[int, int, int, int]
    # Absolute positive click at object center on frame 0.
    first_frame_point_xy: tuple[int, int]


def _object_center(frame_idx: int, spec: SyntheticVideoSpec) -> tuple[int, int]:
    cx = spec.start_x + frame_idx * spec.velocity_x
    cy = spec.start_y + frame_idx * spec.velocity_y
    # Keep fully inside the frame.
    half = spec.object_size // 2
    cx = int(np.clip(cx, half, spec.width - half - 1))
    cy = int(np.clip(cy, half, spec.height - half - 1))
    return cx, cy


def generate_synthetic_video(
    output_dir: str | Path,
    spec: SyntheticVideoSpec | None = None,
) -> SyntheticVideoResult:
    """Write numbered JPEG frames with one clearly moving bright disk."""
    if spec is None:
        spec = SyntheticVideoSpec()
    if spec.num_frames < 2:
        raise ValueError("num_frames must be >= 2")
    if spec.width < 64 or spec.height < 64:
        raise ValueError("resolution too small")
    if spec.object_size < 8:
        raise ValueError("object_size too small")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(spec.seed)
    frame_paths: list[Path] = []
    first_box: tuple[int, int, int, int] | None = None
    first_point: tuple[int, int] | None = None

    for i in range(spec.num_frames):
        # Mild textured background for non-trivial segmentation.
        noise = rng.integers(20, 60, size=(spec.height, spec.width, 3), dtype=np.uint8)
        img = Image.fromarray(noise, mode="RGB")
        draw = ImageDraw.Draw(img)

        cx, cy = _object_center(i, spec)
        half = spec.object_size // 2
        xyxy = (cx - half, cy - half, cx + half, cy + half)
        draw.ellipse(xyxy, fill=(240, 60, 40))

        if i == 0:
            first_box = xyxy
            first_point = (cx, cy)

        path = out / f"{i:05d}.jpg"
        img.save(path, format="JPEG", quality=95)
        frame_paths.append(path)

    assert first_box is not None and first_point is not None
    return SyntheticVideoResult(
        frames_dir=out,
        frame_paths=frame_paths,
        width=spec.width,
        height=spec.height,
        num_frames=spec.num_frames,
        first_frame_box_xyxy=first_box,
        first_frame_point_xy=first_point,
    )
