"""DINOv3 ViT-S/16 local-only adapter (no downloads)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class DINOv3AdapterError(Exception):
    pass


@dataclass
class DINOv3EncodeResult:
    embedding: Any
    shape: tuple[int, ...]
    dtype: str
    finite: bool
    checkpoint_path: str
    package_dir: str
    input_resolution: tuple[int, int]
    load_time_sec: float
    inference_time_sec: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None


class DINOv3Adapter:
    """Load facebook DINOv3 ViT-S/16 strictly from a local package directory."""

    def __init__(self, package_dir: str | Path, device: str | None = None) -> None:
        self.package_dir = Path(package_dir)
        self.device = device
        self._model = None
        self._processor = None
        self.checkpoint_path: Path | None = None
        self.load_time_sec = 0.0

    def load(self) -> None:
        import time

        import torch
        from transformers import AutoImageProcessor, AutoModel

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise DINOv3AdapterError("CUDA requested but unavailable")

        weight = self.package_dir / "model.safetensors"
        if not weight.is_file():
            raise DINOv3AdapterError(f"Missing local weight: {weight}")
        self.checkpoint_path = weight

        t0 = time.perf_counter()
        # local_files_only prevents any Hub download attempt.
        self._processor = AutoImageProcessor.from_pretrained(
            str(self.package_dir),
            local_files_only=True,
        )
        self._model = AutoModel.from_pretrained(
            str(self.package_dir),
            local_files_only=True,
            dtype=torch.float32,
        )
        self._model.to(self.device)
        self._model.eval()
        self.load_time_sec = time.perf_counter() - t0

    def encode_rgb_image(self, image: np.ndarray) -> DINOv3EncodeResult:
        if self._model is None or self._processor is None:
            raise DINOv3AdapterError("Model not loaded")
        if image.ndim != 3 or image.shape[2] != 3:
            raise DINOv3AdapterError("Expected HxWx3 RGB uint8/float image")

        import time

        import torch
        from PIL import Image

        from visionforge.observability.gpu_metrics import reset_peak_stats, snapshot_gpu

        pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
        inputs = self._processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        h = int(inputs["pixel_values"].shape[-2])
        w = int(inputs["pixel_values"].shape[-1])

        if str(self.device).startswith("cuda"):
            reset_peak_stats()

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self._model(**inputs)
            # Prefer pooled embedding when available; else CLS token.
            if getattr(outputs, "pooler_output", None) is not None:
                emb = outputs.pooler_output
            else:
                emb = outputs.last_hidden_state[:, 0]
            emb_cpu = emb.detach().float().cpu().numpy()
        infer_s = time.perf_counter() - t0

        snap = snapshot_gpu() if str(self.device).startswith("cuda") else None
        finite = bool(np.isfinite(emb_cpu).all())
        return DINOv3EncodeResult(
            embedding=emb_cpu,
            shape=tuple(emb_cpu.shape),
            dtype=str(emb_cpu.dtype),
            finite=finite,
            checkpoint_path=str(self.checkpoint_path),
            package_dir=str(self.package_dir),
            input_resolution=(w, h),
            load_time_sec=round(self.load_time_sec, 4),
            inference_time_sec=round(infer_s, 4),
            peak_allocated_bytes=snap.max_allocated_bytes if snap else None,
            peak_reserved_bytes=snap.max_reserved_bytes if snap else None,
        )

    def close(self) -> None:
        self._model = None
        self._processor = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
