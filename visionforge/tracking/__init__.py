"""Video tracker backend protocol and capability status."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


class TrackerId(str, Enum):
    EDGETAM = "edgetam"
    SAM31 = "sam31"


class CapabilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    AVAILABLE_WSL2 = "AVAILABLE_WSL2"
    BLOCKED_NATIVE_WINDOWS = "BLOCKED_NATIVE_WINDOWS"
    BLOCKED_WSL_MISSING = "BLOCKED_WSL_MISSING"
    UNAVAILABLE = "UNAVAILABLE"


_AVAILABLE_STATUSES = frozenset(
    {
        CapabilityStatus.AVAILABLE,
        CapabilityStatus.AVAILABLE_WSL2,
    }
)


@dataclass(frozen=True)
class TrackerCapability:
    tracker_id: TrackerId
    status: CapabilityStatus
    detail: str


@dataclass
class TrackedFrameMask:
    frame_index: int
    object_id: int
    mask: np.ndarray  # HxW bool
    valid: bool
    error: str | None = None


@dataclass
class TrackerRunResult:
    tracker_id: TrackerId
    checkpoint_path: str
    frames: list[TrackedFrameMask] = field(default_factory=list)
    load_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    warnings: list[str] = field(default_factory=list)
    used_real_cuda: bool = False


class TrackerBackendError(Exception):
    pass


@runtime_checkable
class VideoTrackerBackend(Protocol):
    @property
    def tracker_id(self) -> TrackerId: ...

    def capability(self) -> TrackerCapability: ...

    def load(self) -> None: ...

    def track(
        self,
        frames_dir: Path,
        *,
        box_xyxy: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
        object_id: int = 1,
    ) -> TrackerRunResult: ...

    def close(self) -> None: ...


def select_tracker_backend(name: str) -> VideoTrackerBackend:
    """Instantiate the requested backend. Never silently swaps backends."""
    key = name.strip().lower()
    if key in ("edgetam", TrackerId.EDGETAM.value):
        from visionforge.tracking.edgetam_backend import EdgeTAMTrackerBackend

        return EdgeTAMTrackerBackend()
    if key in ("sam31", "sam3.1", TrackerId.SAM31.value):
        from visionforge.tracking.sam31_backend import SAM31TrackerBackend

        return SAM31TrackerBackend()
    raise TrackerBackendError(
        f"Unknown tracker backend {name!r}. Supported: edgetam, sam31"
    )


def require_available(backend: VideoTrackerBackend) -> None:
    cap = backend.capability()
    if cap.status not in _AVAILABLE_STATUSES:
        raise TrackerBackendError(
            f"Tracker {cap.tracker_id.value} is not available: "
            f"{cap.status.value} — {cap.detail}"
        )
