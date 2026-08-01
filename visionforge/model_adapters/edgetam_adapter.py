"""EdgeTAM local checkpoint adapter around official SAM2-style video predictor."""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class EdgeTAMAdapterError(Exception):
    pass


@dataclass
class EdgeTAMFrameMask:
    frame_index: int
    object_id: int
    mask: np.ndarray


@dataclass
class EdgeTAMTrackResult:
    masks: list[EdgeTAMFrameMask] = field(default_factory=list)
    checkpoint_path: str = ""
    load_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    num_frames: int = 0
    resolution: tuple[int, int] = (0, 0)
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    warnings: list[str] = field(default_factory=list)


_FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Bound memory: EdgeTAM materializes the whole clip as model-sized tensors.
_CHUNK_FRAMES = 96
_CHUNK_OVERLAP = 8


def _is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "outofmemory" in name
        or "out of memory" in text
        or "cuda out of memory" in text
    )


def _list_frames(frames_dir: Path) -> list[Path]:
    frames = [
        p
        for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _FRAME_EXTS
    ]
    return sorted(frames, key=lambda p: p.name.lower())


def _mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if xs.size == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _link_or_copy(src: Path, dest: Path) -> None:
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


class EdgeTAMAdapter:
    """Thin wrapper over facebookresearch/EdgeTAM video predictor."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        model_cfg: str = "configs/edgetam.yaml",
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model_cfg = model_cfg
        self.device = device
        self._predictor: Any | None = None
        self.load_time_sec = 0.0
        self.warnings: list[str] = []

    def load(self) -> None:
        import time
        from unittest.mock import patch

        import torch

        if not self.checkpoint_path.is_file():
            raise EdgeTAMAdapterError(f"Missing checkpoint: {self.checkpoint_path}")
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            raise EdgeTAMAdapterError("EdgeTAM smoke test requires CUDA")

        try:
            from sam2.build_sam import build_sam2_video_predictor
            from timm.models import create_model as timm_create_model
        except ImportError as exc:
            raise EdgeTAMAdapterError(
                "Official EdgeTAM/sam2 package is not installed. "
                "Install facebookresearch/EdgeTAM as an external dependency."
            ) from exc

        def _create_model_local(*args, **kwargs):
            # Upstream TimmBackbone hardcodes pretrained=True, which tries to
            # download ImageNet init weights. EdgeTAM's local edgetam.pt already
            # contains the trained trunk weights, so force offline construction.
            kwargs = dict(kwargs)
            kwargs["pretrained"] = False
            self.warnings.append(
                "timm backbone constructed with pretrained=False; "
                "weights come from local edgetam.pt"
            )
            return timm_create_model(*args, **kwargs)

        t0 = time.perf_counter()
        with patch("sam2.modeling.backbones.timm.create_model", side_effect=_create_model_local):
            self._predictor = build_sam2_video_predictor(
                self.model_cfg,
                str(self.checkpoint_path),
            )
        self.load_time_sec = time.perf_counter() - t0

    def track_video_dir(
        self,
        frames_dir: str | Path,
        *,
        object_id: int = 1,
        box_xyxy: tuple[int, int, int, int] | None = None,
        point_xy: tuple[int, int] | None = None,
        frame_width: int,
        frame_height: int,
        require_nonempty: bool = True,
    ) -> EdgeTAMTrackResult:
        if self._predictor is None:
            raise EdgeTAMAdapterError("Model not loaded")
        if (box_xyxy is None) == (point_xy is None):
            raise EdgeTAMAdapterError("Provide exactly one of box_xyxy or point_xy")

        import time

        from visionforge.observability.gpu_metrics import (
            empty_cuda_cache,
            reset_peak_stats,
            snapshot_gpu,
        )

        frames_dir = Path(frames_dir)
        if not frames_dir.is_dir():
            raise EdgeTAMAdapterError(f"frames_dir missing: {frames_dir}")

        frame_paths = _list_frames(frames_dir)
        if not frame_paths:
            raise EdgeTAMAdapterError(f"No frames in {frames_dir}")

        empty_cuda_cache()
        reset_peak_stats()
        t0 = time.perf_counter()

        try:
            if len(frame_paths) <= _CHUNK_FRAMES:
                masks = self._track_dir(
                    frames_dir,
                    object_id=object_id,
                    box_xyxy=box_xyxy,
                    point_xy=point_xy,
                    frame_offset=0,
                )
            else:
                seed_box = box_xyxy
                if seed_box is None and point_xy is not None:
                    px, py = point_xy
                    pad = 12
                    seed_box = (
                        max(0, px - pad),
                        max(0, py - pad),
                        min(frame_width, px + pad),
                        min(frame_height, py + pad),
                    )
                    self.warnings.append("point prompt expanded to box for chunked tracking")
                masks = self._track_chunked(
                    frame_paths,
                    object_id=object_id,
                    box_xyxy=seed_box,  # type: ignore[arg-type]
                )
                self.warnings.append(
                    f"chunked tracking: {len(frame_paths)} frames "
                    f"(chunk={_CHUNK_FRAMES}, overlap={_CHUNK_OVERLAP})"
                )
        except Exception as exc:
            if _is_oom(exc) or (
                isinstance(exc, EdgeTAMAdapterError) and str(exc).startswith("GPU_OOM")
            ):
                empty_cuda_cache()
                if isinstance(exc, EdgeTAMAdapterError) and str(exc).startswith("GPU_OOM"):
                    raise
                raise EdgeTAMAdapterError(
                    "GPU_OOM:EdgeTAM ran out of GPU memory tracking this clip. "
                    "Retry Optimized or use a shorter / lower-resolution video."
                ) from exc
            raise

        infer_s = time.perf_counter() - t0
        snap = snapshot_gpu()

        for fm in masks:
            if fm.mask.ndim != 2:
                raise EdgeTAMAdapterError(f"Mask not HxW at frame {fm.frame_index}")
            if fm.mask.shape != (frame_height, frame_width):
                raise EdgeTAMAdapterError(
                    f"Mask shape {fm.mask.shape} != expected {(frame_height, frame_width)}"
                )
            if fm.mask.dtype != bool and set(np.unique(fm.mask.astype(np.uint8))).issubset({0, 1}):
                fm.mask = fm.mask.astype(bool)
            uniq = set(np.unique(fm.mask.astype(np.uint8)).tolist())
            if not uniq.issubset({0, 1}):
                raise EdgeTAMAdapterError(f"Mask not binary at frame {fm.frame_index}: {uniq}")
            if require_nonempty and not fm.mask.any():
                raise EdgeTAMAdapterError(f"Empty mask at frame {fm.frame_index}")

        return EdgeTAMTrackResult(
            masks=masks,
            checkpoint_path=str(self.checkpoint_path),
            load_time_sec=round(self.load_time_sec, 4),
            inference_time_sec=round(infer_s, 4),
            num_frames=len({m.frame_index for m in masks}),
            resolution=(frame_width, frame_height),
            peak_allocated_bytes=snap.max_allocated_bytes,
            peak_reserved_bytes=snap.max_reserved_bytes,
            warnings=list(self.warnings),
        )

    def _track_chunked(
        self,
        frame_paths: list[Path],
        *,
        object_id: int,
        box_xyxy: tuple[int, int, int, int],
    ) -> list[EdgeTAMFrameMask]:
        from visionforge.observability.gpu_metrics import empty_cuda_cache

        step = max(1, _CHUNK_FRAMES - _CHUNK_OVERLAP)
        by_idx: dict[int, EdgeTAMFrameMask] = {}
        seed_box = box_xyxy
        n = len(frame_paths)
        start = 0
        while start < n:
            end = min(n, start + _CHUNK_FRAMES)
            chunk = frame_paths[start:end]
            with tempfile.TemporaryDirectory(prefix="vf_edgetam_chunk_") as tmp:
                tmp_dir = Path(tmp)
                for i, src in enumerate(chunk):
                    _link_or_copy(src, tmp_dir / f"{i:05d}{src.suffix.lower()}")
                chunk_masks = self._track_dir(
                    tmp_dir,
                    object_id=object_id,
                    box_xyxy=seed_box,
                    point_xy=None,
                    frame_offset=start,
                )

            keep_from = 0 if start == 0 else _CHUNK_OVERLAP
            for fm in chunk_masks:
                if (fm.frame_index - start) >= keep_from:
                    by_idx[fm.frame_index] = fm

            last = next((m for m in reversed(chunk_masks) if m.mask.any()), None)
            if last is not None:
                nb = _mask_to_box(last.mask)
                if nb is not None:
                    seed_box = nb

            empty_cuda_cache()
            if end >= n:
                break
            start += step

        return [by_idx[i] for i in sorted(by_idx)]

    def _track_dir(
        self,
        frames_dir: Path,
        *,
        object_id: int,
        box_xyxy: tuple[int, int, int, int] | None,
        point_xy: tuple[int, int] | None,
        frame_offset: int,
    ) -> list[EdgeTAMFrameMask]:
        import torch

        assert self._predictor is not None
        masks: list[EdgeTAMFrameMask] = []

        with torch.inference_mode():
            use_bf16 = torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            if not use_bf16:
                self.warnings.append("bfloat16 unsupported; using float16 autocast")

            with torch.autocast("cuda", dtype=dtype):
                # Offload frames/state to CPU — critical for long clips on 8GB GPUs.
                # Without this, init_state places all resized tensors on CUDA
                # (~14GiB for ~1100 frames at model image_size).
                state = self._predictor.init_state(
                    video_path=str(frames_dir),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                )
                if box_xyxy is not None:
                    box = np.array(box_xyxy, dtype=np.float32)
                    self._predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=object_id,
                        box=box,
                    )
                else:
                    assert point_xy is not None
                    points = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)
                    self._predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=object_id,
                        points=points,
                        labels=labels,
                    )

                for frame_idx, obj_ids, out_masks in self._predictor.propagate_in_video(state):
                    arr = out_masks
                    if hasattr(arr, "detach"):
                        arr = arr.detach().float().cpu().numpy()
                    arr = np.asarray(arr)
                    global_idx = int(frame_idx) + int(frame_offset)
                    if arr.ndim == 4:
                        for i, oid in enumerate(obj_ids):
                            m = arr[i, 0]
                            binary = m > 0.0 if m.dtype != np.bool_ else m.astype(bool)
                            masks.append(
                                EdgeTAMFrameMask(
                                    frame_index=global_idx,
                                    object_id=int(oid),
                                    mask=binary.astype(bool),
                                )
                            )
                    elif arr.ndim == 3:
                        for i, oid in enumerate(obj_ids):
                            m = arr[i]
                            binary = m > 0.0 if m.dtype != np.bool_ else m.astype(bool)
                            masks.append(
                                EdgeTAMFrameMask(
                                    frame_index=global_idx,
                                    object_id=int(oid),
                                    mask=binary.astype(bool),
                                )
                            )
                    else:
                        raise EdgeTAMAdapterError(f"Unexpected mask rank: {arr.ndim}")

                try:
                    self._predictor.reset_state(state)
                except Exception:
                    pass

        return masks

    def close(self) -> None:
        self._predictor = None
        try:
            import gc

            import torch

            from visionforge.observability.gpu_metrics import empty_cuda_cache

            gc.collect()
            empty_cuda_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
