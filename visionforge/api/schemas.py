"""Pydantic request/response models for the VisionForge API."""
from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    REVIEW_REQUIRED = "review_required"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {
        JobStatus.SUCCEEDED,
        JobStatus.REVIEW_REQUIRED,
        JobStatus.PARTIAL,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
)


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracker: Literal["edgetam", "sam31"]
    box: list[float] = Field(..., min_length=4, max_length=4)
    labels: list[str] = Field(default_factory=list)
    max_frames: int | None = None
    processing_width: int | None = None
    processing_height: int | None = None
    analysis_mode: Literal["full", "sampled"] = "full"
    selection_mode: Literal["manual", "automatic"] = "manual"
    mask_confirmed: bool = False
    fps_override: float | None = None
    start_frame: int | None = None
    parent_job_id: str | None = None
    revision_id: str | None = None
    operation: Literal["track_analyze", "remove_object"] = "track_analyze"
    selected_label: str | None = None
    anchor_time_sec: float | None = None
    quality_mode: Literal["standard", "high"] = "standard"

    @field_validator("box")
    @classmethod
    def _box_finite(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("box must have exactly 4 values")
        for x in v:
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise ValueError("box values must be numbers")
            if not math.isfinite(float(x)):
                raise ValueError("box values must be finite")
        x0, y0, x1, y1 = (float(x) for x in v)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("box must satisfy x1>x0 and y1>y0")
        return [x0, y0, x1, y1]

    @field_validator("labels")
    @classmethod
    def _labels_ok(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for raw in v:
            if not isinstance(raw, str):
                raise ValueError("labels must be strings")
            s = raw.strip()
            if not s:
                raise ValueError("labels must not be empty strings")
            if any(ord(ch) < 32 for ch in s):
                raise ValueError("labels must not contain control characters")
            out.append(s)
        return out

    @field_validator("start_frame")
    @classmethod
    def _start_ok(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if int(v) < 0:
            raise ValueError("start_frame must be >= 0")
        return int(v)

    @model_validator(mode="after")
    def _res_pair(self) -> JobSpec:
        w, h = self.processing_width, self.processing_height
        if (w is None) ^ (h is None):
            raise ValueError("processing_width and processing_height must both be set or both null")
        if w is not None and (w < 16 or h is None or h < 16):
            raise ValueError("processing dimensions must be >= 16")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        if self.analysis_mode == "sampled" and self.max_frames is None:
            raise ValueError("sampled analysis_mode requires max_frames")
        if self.analysis_mode == "full" and self.max_frames is not None:
            # Allow safety cap but mark as capped full
            pass
        if not self.mask_confirmed:
            raise ValueError("mask_confirmed must be true before starting analysis")
        return self


class CorrectionSpec(BaseModel):
    """Manual reinitialization / revision request (does not overwrite parent)."""

    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(..., ge=0)
    box: list[float] = Field(..., min_length=4, max_length=4)
    tracker: Literal["edgetam", "sam31"]
    mask_confirmed: bool = False
    labels: list[str] = Field(default_factory=list)
    analysis_mode: Literal["full", "sampled"] = "full"
    max_frames: int | None = None

    @field_validator("box")
    @classmethod
    def _box_ok(cls, v: list[float]) -> list[float]:
        x0, y0, x1, y1 = (float(x) for x in v)
        if not all(math.isfinite(x) for x in (x0, y0, x1, y1)):
            raise ValueError("box must be finite")
        if x1 <= x0 or y1 <= y0:
            raise ValueError("box must satisfy x1>x0 and y1>y0")
        return [x0, y0, x1, y1]

    @model_validator(mode="after")
    def _confirmed(self) -> CorrectionSpec:
        if not self.mask_confirmed:
            raise ValueError("mask_confirmed must be true for corrections")
        return self


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus
    tracker: str
    status_url: str
    parent_job_id: str | None = None
    revision_id: str | None = None


class JobStatusView(BaseModel):
    job_id: str
    tracker: str
    status: JobStatus
    stage: str | None = None
    stage_progress: dict[str, Any] | None = None
    overall_percent: float | None = None
    frames_completed: int | None = None
    frames_total: int | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    queue_position: int | None = None
    pipeline_run_id: str | None = None
    warning_code: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    result_url: str | None = None
    artifacts_url: str | None = None
    download_url: str | None = None
    runtime: str | None = None


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str, *, default: str = "upload.bin") -> str:
    base = (name or "").replace("\\", "/").split("/")[-1].strip()
    base = _SAFE_NAME_RE.sub("_", base)
    base = base.strip("._")
    if not base or base in {".", ".."}:
        return default
    return base[:180]
