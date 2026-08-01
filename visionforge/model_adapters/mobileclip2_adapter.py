"""MobileCLIP2-S0 local checkpoint adapter via official open_clip API."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


class MobileCLIP2AdapterError(Exception):
    pass


@dataclass
class MobileCLIP2Result:
    checkpoint_path: str
    image_embedding_shape: tuple[int, ...]
    text_embedding_shape: tuple[int, ...]
    embedding_dim: int
    labels: list[str]
    similarity_scores: list[float]
    finite: bool
    load_time_sec: float
    inference_time_sec: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None


class MobileCLIP2Adapter:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self.load_time_sec = 0.0

    def load(self) -> None:
        import time

        import open_clip
        import torch

        # Host RAM is tight on this 16GB machine; limit CPU thread pools.
        torch.set_num_threads(1)
        try:
            import torch.backends.cudnn as cudnn

            cudnn.benchmark = False
        except Exception:
            pass

        if not self.checkpoint_path.is_file():
            raise MobileCLIP2AdapterError(f"Missing checkpoint: {self.checkpoint_path}")
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise MobileCLIP2AdapterError("CUDA requested but unavailable")

        if str(self.device).startswith("cuda"):
            import gc

            gc.collect()
            torch.cuda.empty_cache()

        t0 = time.perf_counter()
        # S0 requires mean/std (0,0,0)/(1,1,1) when loading a local .pt path.
        model, _, preprocess = open_clip.create_model_and_transforms(
            "MobileCLIP2-S0",
            pretrained=str(self.checkpoint_path),
            image_mean=(0.0, 0.0, 0.0),
            image_std=(1.0, 1.0, 1.0),
        )
        reparam = None
        try:
            from open_clip.utils import reparameterize_model as reparam  # type: ignore
        except Exception:
            try:
                from mobileclip.modules.common.mobileone import (  # type: ignore
                    reparameterize_model as reparam,
                )
            except Exception:
                reparam = None

        model.eval()
        if reparam is not None:
            model = reparam(model)
        model = model.to(self.device)
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("MobileCLIP2-S0")
        self.load_time_sec = time.perf_counter() - t0

    def encode_and_compare(
        self,
        image: np.ndarray,
        labels: Sequence[str],
    ) -> MobileCLIP2Result:
        if self._model is None or self._preprocess is None or self._tokenizer is None:
            raise MobileCLIP2AdapterError("Model not loaded")
        if len(labels) < 2:
            raise MobileCLIP2AdapterError("Provide at least two text labels")

        import time

        import torch
        from PIL import Image

        from visionforge.observability.gpu_metrics import reset_peak_stats, snapshot_gpu

        pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
        image_tensor = self._preprocess(pil).unsqueeze(0).to(self.device)
        text = self._tokenizer(list(labels)).to(self.device)

        if str(self.device).startswith("cuda"):
            reset_peak_stats()

        t0 = time.perf_counter()
        with torch.inference_mode():
            image_features = self._model.encode_image(image_tensor)
            text_features = self._model.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).float().cpu().numpy().reshape(-1)
            img_np = image_features.float().cpu().numpy()
            txt_np = text_features.float().cpu().numpy()
        infer_s = time.perf_counter() - t0

        snap = snapshot_gpu() if str(self.device).startswith("cuda") else None
        finite = bool(
            np.isfinite(img_np).all()
            and np.isfinite(txt_np).all()
            and np.isfinite(sims).all()
        )
        return MobileCLIP2Result(
            checkpoint_path=str(self.checkpoint_path),
            image_embedding_shape=tuple(img_np.shape),
            text_embedding_shape=tuple(txt_np.shape),
            embedding_dim=int(img_np.shape[-1]),
            labels=list(labels),
            similarity_scores=[float(x) for x in sims.tolist()],
            finite=finite,
            load_time_sec=round(self.load_time_sec, 4),
            inference_time_sec=round(infer_s, 4),
            peak_allocated_bytes=snap.max_allocated_bytes if snap else None,
            peak_reserved_bytes=snap.max_reserved_bytes if snap else None,
        )

    def close(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
