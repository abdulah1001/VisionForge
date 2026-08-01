"""EdgeTAM implementation of VideoTrackerBackend."""
from __future__ import annotations

from pathlib import Path

from visionforge.model_registry import LocalModelRegistry, ModelId
from visionforge.tracking import (
    CapabilityStatus,
    TrackedFrameMask,
    TrackerBackendError,
    TrackerCapability,
    TrackerId,
    TrackerRunResult,
)


class EdgeTAMTrackerBackend:
    def __init__(self) -> None:
        self._adapter = None
        self._checkpoint: Path | None = None

    @property
    def tracker_id(self) -> TrackerId:
        return TrackerId.EDGETAM

    def capability(self) -> TrackerCapability:
        return TrackerCapability(
            tracker_id=TrackerId.EDGETAM,
            status=CapabilityStatus.AVAILABLE,
            detail="Official EdgeTAM CUDA tracker via local edgetam.pt",
        )

    def load(self) -> None:
        from visionforge.model_adapters.edgetam_adapter import EdgeTAMAdapter

        pkg = LocalModelRegistry().validate(ModelId.EDGETAM)
        self._checkpoint = pkg.primary_checkpoint
        self._adapter = EdgeTAMAdapter(pkg.primary_checkpoint, device="cuda")
        self._adapter.load()

    def track(
        self,
        frames_dir: Path,
        *,
        box_xyxy: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
        object_id: int = 1,
    ) -> TrackerRunResult:
        if self._adapter is None or self._checkpoint is None:
            raise TrackerBackendError("EdgeTAM backend not loaded")

        box = (
            int(round(box_xyxy[0])),
            int(round(box_xyxy[1])),
            int(round(box_xyxy[2])),
            int(round(box_xyxy[3])),
        )
        raw = self._adapter.track_video_dir(
            frames_dir,
            object_id=object_id,
            box_xyxy=box,
            frame_width=frame_width,
            frame_height=frame_height,
            require_nonempty=False,
        )

        frames: list[TrackedFrameMask] = []
        for fm in raw.masks:
            mask = fm.mask.astype(bool)
            valid = bool(mask.any())
            frames.append(
                TrackedFrameMask(
                    frame_index=int(fm.frame_index),
                    object_id=int(fm.object_id),
                    mask=mask,
                    valid=valid,
                    error=None if valid else "empty_mask",
                )
            )

        return TrackerRunResult(
            tracker_id=TrackerId.EDGETAM,
            checkpoint_path=str(self._checkpoint),
            frames=frames,
            load_time_sec=raw.load_time_sec,
            inference_time_sec=raw.inference_time_sec,
            peak_allocated_bytes=raw.peak_allocated_bytes,
            peak_reserved_bytes=raw.peak_reserved_bytes,
            warnings=list(raw.warnings),
            used_real_cuda=True,
        )

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.close()
        self._adapter = None
