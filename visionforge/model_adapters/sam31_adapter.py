"""Thin VisionForge adapter around official SAM 3.1 Object Multiplex APIs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

PromptKind = Literal["point", "box"]


@dataclass(frozen=True)
class PointPrompt:
    x: float
    y: float
    label: int = 1  # 1 = positive, 0 = negative
    absolute: bool = True


@dataclass(frozen=True)
class BoxPrompt:
    x0: float
    y0: float
    x1: float
    y1: float
    absolute: bool = True


@dataclass
class FrameMask:
    frame_index: int
    object_id: int
    mask: Any  # numpy bool/uint8 array HxW
    score: float | None = None


@dataclass
class TrackResult:
    masks: list[FrameMask] = field(default_factory=list)
    checkpoint_path: str | None = None
    checkpoint_name: str | None = None
    session_id: str | None = None
    warnings: list[str] = field(default_factory=list)


class SAM31AdapterError(Exception):
    """Raised for VisionForge-level adapter validation / runtime errors."""


class SAM31Adapter:
    """Hides upstream SAM 3.1 multiplex predictor details behind a small API."""

    APPROVED_REPO = "facebook/sam3.1"
    APPROVED_CHECKPOINT_NAME = "sam3.1_multiplex.pt"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        compile_model: bool = False,
        use_fa3: bool = False,
        use_rope_real: bool = False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device
        self.compile_model = compile_model
        # FA3 / rope_real often unavailable or fragile on Windows consumer GPUs.
        self.use_fa3 = use_fa3
        self.use_rope_real = use_rope_real
        self._predictor: Any | None = None
        self._session_id: str | None = None
        self._frame_size: tuple[int, int] | None = None  # (W, H)
        self._resolved_checkpoint: Path | None = None
        self.warnings: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def resolved_checkpoint(self) -> Path | None:
        return self._resolved_checkpoint

    def _require_cuda(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise SAM31AdapterError("PyTorch is not installed") from exc
        if not torch.cuda.is_available():
            raise SAM31AdapterError("CUDA is unavailable; SAM 3.1 smoke test requires a CUDA GPU")

    def _check_hf_auth(self) -> None:
        try:
            from huggingface_hub import HfApi
            from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
        except ImportError as exc:
            raise SAM31AdapterError("huggingface_hub is not installed") from exc

        token_present = False
        try:
            from huggingface_hub import get_token

            token_present = bool(get_token())
        except Exception:
            token_present = False

        if not token_present:
            raise SAM31AdapterError(
                "Hugging Face authentication unavailable. Run `hf auth login` locally "
                "(do not paste tokens into chat or source files)."
            )

        api = HfApi()
        try:
            api.model_info(self.APPROVED_REPO)
        except GatedRepoError as exc:
            raise SAM31AdapterError(
                f"No access to gated repo {self.APPROVED_REPO}. "
                "Request access on Hugging Face, then re-authenticate with `hf auth login`."
            ) from exc
        except HfHubHTTPError as exc:
            raise SAM31AdapterError(
                f"Unable to verify access to {self.APPROVED_REPO}: HTTP error"
            ) from exc

    def validate_prompt(
        self,
        *,
        points: Sequence[PointPrompt] | None = None,
        box: BoxPrompt | None = None,
    ) -> PromptKind:
        has_points = bool(points)
        has_box = box is not None
        if has_points == has_box:
            raise SAM31AdapterError("Provide exactly one of: points prompt OR box prompt")
        if points is not None:
            if len(points) < 1:
                raise SAM31AdapterError("points prompt must contain at least one point")
            for p in points:
                if p.label not in (0, 1):
                    raise SAM31AdapterError("point label must be 0 or 1")
                if p.absolute and self._frame_size is not None:
                    w, h = self._frame_size
                    if not (0 <= p.x < w and 0 <= p.y < h):
                        raise SAM31AdapterError(f"point ({p.x}, {p.y}) outside frame {w}x{h}")
            return "point"
        assert box is not None
        if box.x1 <= box.x0 or box.y1 <= box.y0:
            raise SAM31AdapterError("box must satisfy x1>x0 and y1>y0")
        return "box"

    def validate_masks(self, masks: Sequence[FrameMask], *, expected_hw: tuple[int, int]) -> None:
        if not masks:
            raise SAM31AdapterError("expected non-empty mask list")
        eh, ew = expected_hw
        for m in masks:
            if m.frame_index < 0:
                raise SAM31AdapterError("frame_index must be >= 0")
            arr = m.mask
            if arr is None:
                raise SAM31AdapterError("mask array is None")
            shape = getattr(arr, "shape", None)
            if shape is None or len(shape) != 2:
                raise SAM31AdapterError(f"mask must be HxW, got shape={shape}")
            if tuple(shape) != (eh, ew):
                raise SAM31AdapterError(f"mask shape {tuple(shape)} != expected {(eh, ew)}")
            try:
                import numpy as np

                if not np.asarray(arr).any():
                    raise SAM31AdapterError(
                        f"mask empty for frame={m.frame_index} object={m.object_id}"
                    )
            except SAM31AdapterError:
                raise
            except Exception as exc:
                raise SAM31AdapterError(f"unable to validate mask contents: {exc}") from exc

    def load(self) -> None:
        """Load the real SAM 3.1 Object Multiplex predictor."""
        self._require_cuda()
        if self.checkpoint_path is None:
            self._check_hf_auth()

        try:
            from sam3.model_builder import build_sam3_multiplex_video_predictor
        except ImportError as exc:
            detail = str(exc)
            if "triton" in detail.lower():
                raise SAM31AdapterError(
                    "Official sam3 import failed because 'triton' is unavailable. "
                    "Triton has no Windows wheels (pip: No matching distribution). "
                    "WSL2 Ubuntu is required for native SAM 3.1 execution on this machine. "
                    f"Underlying import error: {detail}"
                ) from exc
            raise SAM31AdapterError(
                "Official sam3 package is not installed or failed to import. "
                "Install from https://github.com/facebookresearch/sam3 "
                f"(external dependency). Underlying error: {detail}"
            ) from exc

        ckpt_arg = str(self.checkpoint_path) if self.checkpoint_path else None
        warn: list[str] = []
        if not self.use_fa3:
            warn.append("use_fa3=False (safe default for consumer Windows GPU)")
        if not self.use_rope_real:
            warn.append("use_rope_real=False (safe default)")
        if self.compile_model:
            warn.append("compile=True requested")
        else:
            warn.append("compile=False (first proof-of-life; no experimental compile)")

        self._predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=ckpt_arg,
            compile=self.compile_model,
            use_fa3=self.use_fa3,
            use_rope_real=self.use_rope_real,
            async_loading_frames=False,
        )
        self.warnings.extend(warn)

        # Resolve checkpoint identity without exposing credentials.
        resolved = self._discover_checkpoint_path()
        self._resolved_checkpoint = resolved
        if resolved is not None and resolved.name != self.APPROVED_CHECKPOINT_NAME:
            raise SAM31AdapterError(
                f"Checkpoint must be {self.APPROVED_CHECKPOINT_NAME}, got {resolved.name}"
            )

    def _discover_checkpoint_path(self) -> Path | None:
        if self.checkpoint_path is not None:
            return Path(self.checkpoint_path)
        # Prefer HF cache location used by sam3.download_ckpt_from_hf
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(
                self.APPROVED_REPO,
                self.APPROVED_CHECKPOINT_NAME,
            )
            incomplete = str(cached).endswith("___incomplete___")
            if cached and cached != "___not_found___" and not incomplete:
                return Path(str(cached))
        except Exception:
            pass
        return None

    def start_video_session(self, resource_path: str | Path) -> str:
        if self._predictor is None:
            raise SAM31AdapterError("Model not loaded; call load() first")
        path = Path(resource_path)
        if not path.exists():
            raise SAM31AdapterError(f"video resource not found: {path}")

        # Infer frame size from first JPEG if directory.
        if path.is_dir():
            frames = sorted(path.glob("*.jpg")) + sorted(path.glob("*.jpeg"))
            if not frames:
                raise SAM31AdapterError(f"no JPEG frames in {path}")
            from PIL import Image

            with Image.open(frames[0]) as im:
                self._frame_size = im.size  # (W, H)
        else:
            self._frame_size = None

        response = self._predictor.handle_request(
            {"type": "start_session", "resource_path": str(path)}
        )
        session_id = response.get("session_id")
        if not session_id:
            raise SAM31AdapterError(f"start_session failed: {response}")
        self._session_id = session_id
        return session_id

    def add_prompt(
        self,
        *,
        frame_index: int = 0,
        object_id: int = 1,
        points: Sequence[PointPrompt] | None = None,
        box: BoxPrompt | None = None,
    ) -> dict[str, Any]:
        if self._predictor is None or self._session_id is None:
            raise SAM31AdapterError("start_video_session() required before add_prompt()")
        if frame_index < 0:
            raise SAM31AdapterError("frame_index must be >= 0")

        kind = self.validate_prompt(points=points, box=box)
        import torch

        request: dict[str, Any] = {
            "type": "add_prompt",
            "session_id": self._session_id,
            "frame_index": frame_index,
            "obj_id": object_id,
        }

        w_h = self._frame_size
        if kind == "point":
            assert points is not None and w_h is not None
            w, h = w_h
            rel = []
            labels = []
            for p in points:
                x = p.x / w if p.absolute else p.x
                y = p.y / h if p.absolute else p.y
                rel.append([x, y])
                labels.append(p.label)
            request["points"] = torch.tensor(rel, dtype=torch.float32)
            request["point_labels"] = torch.tensor(labels, dtype=torch.int32)
        else:
            assert box is not None and w_h is not None
            w, h = w_h
            if box.absolute:
                # Upstream examples use relative xywh for boxes in some paths;
                # multiplex add_prompt commonly accepts boxes as relative xyxy-like tensors.
                # Convert absolute XYXY -> relative XYXY.
                rel_box = [
                    box.x0 / w,
                    box.y0 / h,
                    box.x1 / w,
                    box.y1 / h,
                ]
            else:
                rel_box = [box.x0, box.y0, box.x1, box.y1]
            request["bounding_boxes"] = torch.tensor([rel_box], dtype=torch.float32)
            request["bounding_box_labels"] = torch.tensor([1], dtype=torch.int32)

        response = self._predictor.handle_request(request)
        return response

    def track(self) -> TrackResult:
        if self._predictor is None or self._session_id is None:
            raise SAM31AdapterError("session not started")

        import numpy as np

        masks: list[FrameMask] = []
        for response in self._predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": self._session_id}
        ):
            frame_index = int(response["frame_index"])
            outputs = response.get("outputs") or {}
            # Upstream may return dicts with out_binary_masks / out_obj_ids.
            binary = outputs.get("out_binary_masks")
            obj_ids = outputs.get("out_obj_ids")
            scores = outputs.get("out_probs") or outputs.get("out_scores")

            if binary is None:
                # Alternate key names seen across SAM3 demos.
                binary = outputs.get("binary_masks") or outputs.get("masks")
            if obj_ids is None:
                obj_ids = outputs.get("obj_ids") or outputs.get("object_ids")

            if binary is None:
                continue

            binary_arr = np.asarray(binary)
            if binary_arr.ndim == 2:
                binary_arr = binary_arr[None, ...]
            if obj_ids is None:
                obj_ids = list(range(1, len(binary_arr) + 1))
            obj_ids = list(obj_ids)

            for i, oid in enumerate(obj_ids):
                mask_i = binary_arr[i]
                if hasattr(mask_i, "detach"):
                    mask_i = mask_i.detach().cpu().numpy()
                mask_i = np.asarray(mask_i).astype(bool)
                score = None
                if scores is not None:
                    try:
                        score = float(np.asarray(scores).reshape(-1)[i])
                    except Exception:
                        score = None
                masks.append(
                    FrameMask(
                        frame_index=frame_index,
                        object_id=int(oid),
                        mask=mask_i,
                        score=score,
                    )
                )

        ckpt = self._resolved_checkpoint
        return TrackResult(
            masks=masks,
            checkpoint_path=str(ckpt) if ckpt else None,
            checkpoint_name=ckpt.name if ckpt else self.APPROVED_CHECKPOINT_NAME,
            session_id=self._session_id,
            warnings=list(self.warnings),
        )

    def close(self) -> None:
        """Release session + GPU resources."""
        try:
            if self._predictor is not None and self._session_id is not None:
                self._predictor.handle_request(
                    {"type": "close_session", "session_id": self._session_id}
                )
        except Exception:
            pass
        self._session_id = None
        self._predictor = None
        try:
            from visionforge.observability.gpu_metrics import empty_cuda_cache

            empty_cuda_cache()
        except Exception:
            pass

        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
