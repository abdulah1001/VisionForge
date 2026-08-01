"""Reusable structured progress events for VisionForge pipelines."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


class ProgressSink(Protocol):
    def emit(
        self,
        stage: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        overall_percent: float | None = None,
        message: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None: ...


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class NullProgress:
    def emit(
        self,
        stage: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        overall_percent: float | None = None,
        message: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        return


class JsonlProgressFile:
    """Append-only JSONL progress sink (CLI/API child process)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(
        self,
        stage: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        overall_percent: float | None = None,
        message: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "event": "progress",
            "stage": stage,
            "timestamp": _utc_now_iso(),
        }
        if completed is not None:
            event["completed"] = int(completed)
        if total is not None:
            event["total"] = int(total)
        if overall_percent is not None:
            event["overall_percent"] = round(float(overall_percent), 2)
        if message:
            event["message"] = message
        if extra:
            event["extra"] = extra
        line = json.dumps(event, ensure_ascii=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()


def open_progress(path: Path | None) -> ProgressSink:
    if path is None:
        return NullProgress()
    return JsonlProgressFile(Path(path))


# Approximate overall weights for honest progress (sum ≈ 100).
STAGE_WEIGHTS: dict[str, float] = {
    "validating": 3,
    "preparing_input": 7,
    "tracking": 36,
    "creating_artifacts": 7,
    "validating_masks": 4,
    "encoding_video": 9,
    "dinov3": 12,
    "mobileclip2": 8,
    "recovering_identity": 6,
    "finalizing": 4,
    "completed": 0,
}


def overall_percent_for(stage: str, stage_fraction: float = 1.0) -> float:
    """Map stage + intra-stage fraction [0,1] to overall percent."""
    keys = [
        "validating",
        "preparing_input",
        "tracking",
        "creating_artifacts",
        "validating_masks",
        "encoding_video",
        "dinov3",
        "mobileclip2",
        "recovering_identity",
        "finalizing",
        "completed",
    ]
    done = 0.0
    for k in keys:
        if k == stage:
            done += STAGE_WEIGHTS.get(k, 0) * max(0.0, min(1.0, stage_fraction))
            break
        done += STAGE_WEIGHTS.get(k, 0)
    if stage == "completed":
        return 100.0
    return min(99.0, round(done, 2))
