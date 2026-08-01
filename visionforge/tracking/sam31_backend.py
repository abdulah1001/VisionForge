"""SAM 3.1 tracker backend — WSL2 CUDA runtime via VisionForge-SAM31."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from visionforge.model_registry import LocalModelRegistry, ModelId
from visionforge.tracking import (
    CapabilityStatus,
    TrackedFrameMask,
    TrackerBackendError,
    TrackerCapability,
    TrackerId,
    TrackerRunResult,
)

# Empirically, ~12 frames fit on ~8 GB for SAM 3.1 multiplex in this environment.
# Chunked tracking preserves full-range jobs without silent EdgeTAM fallback.
_DEFAULT_CHUNK = 10


class SAM31TrackerBackend:
    """Runs official sam3 multiplex tracking inside WSL2 when available.

    Never silently falls back to EdgeTAM.
    """

    def __init__(self, *, chunk_size: int = _DEFAULT_CHUNK) -> None:
        self._checkpoint: Path | None = None
        self._probe_detail: str | None = None
        self._loaded = False
        self._chunk_size = max(4, int(chunk_size))

    @property
    def tracker_id(self) -> TrackerId:
        return TrackerId.SAM31

    def capability(self) -> TrackerCapability:
        try:
            pkg = LocalModelRegistry().validate(ModelId.SAM31)
            ckpt = pkg.primary_checkpoint
        except Exception as exc:
            return TrackerCapability(
                tracker_id=TrackerId.SAM31,
                status=CapabilityStatus.UNAVAILABLE,
                detail=f"SAM 3.1 checkpoint validation failed: {exc}",
            )

        try:
            from visionforge.wsl import distro_installed, probe_sam31_runtime
        except Exception as exc:
            return TrackerCapability(
                tracker_id=TrackerId.SAM31,
                status=CapabilityStatus.BLOCKED_NATIVE_WINDOWS,
                detail=(
                    "SAM 3.1 Multiplex checkpoint is present locally at "
                    f"{ckpt}, but WSL bridge import failed ({exc}). "
                    "Official sam3 requires Triton (no native-Windows wheel). "
                    "No silent fallback to EdgeTAM."
                ),
            )

        if not distro_installed():
            return TrackerCapability(
                tracker_id=TrackerId.SAM31,
                status=CapabilityStatus.BLOCKED_WSL_MISSING,
                detail=(
                    f"Checkpoint present at {ckpt}, but WSL distro "
                    "VisionForge-SAM31 is not installed. "
                    "No silent fallback to EdgeTAM."
                ),
            )

        probe = probe_sam31_runtime(timeout_sec=90.0)
        self._probe_detail = probe.detail
        if probe.ok:
            return TrackerCapability(
                tracker_id=TrackerId.SAM31,
                status=CapabilityStatus.AVAILABLE_WSL2,
                detail=(
                    f"SAM 3.1 Multiplex via WSL2 ({ckpt}). {probe.detail}"
                ),
            )

        return TrackerCapability(
            tracker_id=TrackerId.SAM31,
            status=CapabilityStatus.BLOCKED_NATIVE_WINDOWS,
            detail=(
                f"Checkpoint present at {ckpt}, but WSL SAM 3.1 runtime is not "
                f"ready: {probe.detail}. No silent fallback to EdgeTAM."
            ),
        )

    def load(self) -> None:
        cap = self.capability()
        if cap.status != CapabilityStatus.AVAILABLE_WSL2:
            raise TrackerBackendError(f"{cap.status.value}: {cap.detail}")
        pkg = LocalModelRegistry().validate(ModelId.SAM31)
        self._checkpoint = pkg.primary_checkpoint
        self._loaded = True

    def _track_chunk(
        self,
        frames_dir: Path,
        *,
        box_xyxy: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
        object_id: int,
        out_dir: Path,
    ) -> tuple[list[TrackedFrameMask], dict]:
        from visionforge.wsl import (
            WSLBridgeError,
            run_wsl_json,
            windows_to_wsl_path,
            wsl_to_windows_path,
        )

        x0, y0, x1, y1 = box_xyxy
        try:
            payload = run_wsl_json(
                [
                    "--mode",
                    "track",
                    "--checkpoint",
                    windows_to_wsl_path(self._checkpoint),
                    "--frames-dir",
                    windows_to_wsl_path(frames_dir),
                    "--out-dir",
                    windows_to_wsl_path(out_dir),
                    "--box",
                    str(x0),
                    str(y0),
                    str(x1),
                    str(y1),
                    "--object-id",
                    str(int(object_id)),
                ],
                timeout_sec=1800.0,
            )
        except WSLBridgeError as exc:
            raise TrackerBackendError(
                f"SAM31 WSL track failed (no EdgeTAM fallback): {exc}"
            ) from exc

        frames: list[TrackedFrameMask] = []
        for item in payload.get("frames") or []:
            mask_path = item.get("mask_path")
            if mask_path:
                win_mask = wsl_to_windows_path(str(mask_path))
                mask = np.load(win_mask).astype(bool)
            else:
                mask = np.zeros((frame_height, frame_width), dtype=bool)
            valid = bool(item.get("valid", False)) and bool(mask.any())
            frames.append(
                TrackedFrameMask(
                    frame_index=int(item["frame_index"]),
                    object_id=int(item.get("object_id", object_id)),
                    mask=mask,
                    valid=valid,
                    error=item.get("error") if not valid else None,
                )
            )
        return frames, payload

    def track(
        self,
        frames_dir: Path,
        *,
        box_xyxy: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
        object_id: int = 1,
    ) -> TrackerRunResult:
        if not self._loaded or self._checkpoint is None:
            raise TrackerBackendError("SAM31 WSL backend not loaded")

        frames_dir = Path(frames_dir)
        if not frames_dir.is_dir():
            raise TrackerBackendError(f"frames_dir not found: {frames_dir}")

        x0, y0, x1, y1 = box_xyxy
        if x1 <= x0 or y1 <= y0:
            raise TrackerBackendError("box must satisfy x1>x0 and y1>y0")
        if frame_width <= 0 or frame_height <= 0:
            raise TrackerBackendError("frame dimensions must be positive")

        # Ordered frame files
        paths = sorted(
            [
                p
                for p in frames_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
        if not paths:
            raise TrackerBackendError(f"no frames in {frames_dir}")

        out_root = frames_dir.parent / "_sam31_wsl_out"
        out_root.mkdir(parents=True, exist_ok=True)

        all_frames: list[TrackedFrameMask] = []
        warnings: list[str] = []
        load_t = 0.0
        infer_t = 0.0
        peak_a = 0
        peak_r = 0
        used_cuda = False
        current_box = box_xyxy
        chunks = 0

        for start in range(0, len(paths), self._chunk_size):
            end = min(len(paths), start + self._chunk_size)
            # Overlap previous frame into next chunks for re-initialization continuity.
            overlap = 1 if start > 0 else 0
            src_start = max(0, start - overlap)
            chunk_paths = paths[src_start:end]
            chunks += 1
            chunk_dir = Path(tempfile.mkdtemp(prefix="sam31_chunk_", dir=str(out_root)))
            chunk_out = out_root / f"chunk_{start:05d}"
            chunk_out.mkdir(parents=True, exist_ok=True)
            try:
                for i, src in enumerate(chunk_paths):
                    dest = chunk_dir / f"{i:05d}.jpg"
                    if src.suffix.lower() in {".jpg", ".jpeg"}:
                        shutil.copy2(src, dest)
                    else:
                        Image.open(src).convert("RGB").save(dest, quality=95)
                # When overlapping, re-init box on the overlapped first frame
                init_box = current_box
                chunk_frames, payload = self._track_chunk(
                    chunk_dir,
                    box_xyxy=init_box,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    object_id=object_id,
                    out_dir=chunk_out,
                )
                # Remap local indices to global; drop overlapped first frame outputs
                for fr in chunk_frames:
                    global_idx = src_start + int(fr.frame_index)
                    if overlap and global_idx < start:
                        continue
                    fr.frame_index = global_idx
                    all_frames.append(fr)
                load_t += float(payload.get("load_time_sec") or 0.0)
                infer_t += float(payload.get("inference_time_sec") or 0.0)
                if payload.get("peak_allocated_bytes"):
                    peak_a = max(peak_a, int(payload["peak_allocated_bytes"]))
                if payload.get("peak_reserved_bytes"):
                    peak_r = max(peak_r, int(payload["peak_reserved_bytes"]))
                used_cuda = used_cuda or bool(payload.get("used_real_cuda", True))
                warnings.extend(payload.get("warnings") or [])

                # Seed next chunk from last valid mask bbox
                last_valid = next(
                    (f for f in reversed(chunk_frames) if f.valid and f.mask.any()),
                    None,
                )
                if last_valid is not None:
                    ys, xs = np.where(last_valid.mask)
                    current_box = (
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() + 1),
                        float(ys.max() + 1),
                    )
            finally:
                shutil.rmtree(chunk_dir, ignore_errors=True)

        if chunks > 1:
            warnings.append(
                f"sam31_chunked_tracking:chunks={chunks},chunk_size={self._chunk_size}"
            )
        if self._probe_detail:
            warnings.append(self._probe_detail)

        # Ensure contiguous frame list covering all indices
        by_idx = {f.frame_index: f for f in all_frames}
        ordered = [
            by_idx.get(
                i,
                TrackedFrameMask(
                    frame_index=i,
                    object_id=object_id,
                    mask=np.zeros((frame_height, frame_width), dtype=bool),
                    valid=False,
                    error="missing_chunk_output",
                ),
            )
            for i in range(len(paths))
        ]

        return TrackerRunResult(
            tracker_id=TrackerId.SAM31,
            checkpoint_path=str(self._checkpoint),
            frames=ordered,
            load_time_sec=load_t,
            inference_time_sec=infer_t,
            peak_allocated_bytes=peak_a or None,
            peak_reserved_bytes=peak_r or None,
            warnings=warnings,
            used_real_cuda=used_cuda,
        )

    def close(self) -> None:
        self._loaded = False
        self._checkpoint = None
