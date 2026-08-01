"""Health and readiness endpoints."""
from __future__ import annotations

import shutil

from fastapi import APIRouter, Request

from visionforge.api.util import disk_writable

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    # Intentionally does not load models or touch CUDA/WSL.
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request) -> dict:
    settings = request.app.state.settings
    queue = request.app.state.queue
    jobs_root = settings.jobs_root
    writable = disk_writable(jobs_root)
    free = 0
    try:
        free = shutil.disk_usage(jobs_root).free
    except Exception:
        free = 0
    accepting = (
        writable
        and free >= settings.min_free_disk_bytes
        and queue.is_accepting()
    )
    return {
        "status": "ready" if accepting else "not_ready",
        "worker": {
            "alive": True,
            "active_job_id": queue.active_job_id(),
            "gpu_concurrency": settings.gpu_concurrency,
        },
        "queue": {
            "queued": queue.queued_count(),
            "max_queued": settings.max_queued_jobs,
            "accepting": queue.is_accepting(),
        },
        "artifacts": {
            "jobs_root_writable": writable,
            "free_bytes": free,
            "min_free_bytes": settings.min_free_disk_bytes,
        },
        "accepting_jobs": accepting,
    }
