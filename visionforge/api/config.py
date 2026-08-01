"""API service configuration (local-only defaults)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return Path(raw)


@dataclass
class ApiSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    allow_non_loopback: bool = False
    jobs_root: Path = field(
        default_factory=lambda: Path("D:/project/artifacts/api_jobs")
    )
    pipeline_output_root: Path = field(
        default_factory=lambda: Path("D:/project/artifacts/e2e_runs")
    )
    python_executable: Path = field(
        default_factory=lambda: Path("D:/project/.venvs/smoke/Scripts/python.exe")
    )
    project_root: Path = field(default_factory=lambda: Path("D:/project"))
    max_upload_bytes: int = 512 * 1024 * 1024
    max_zip_entries: int = 500
    max_zip_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024
    max_image_frames: int = 300
    max_labels: int = 32
    max_label_length: int = 128
    max_max_frames: int = 300
    max_processing_side: int = 1920
    min_free_disk_bytes: int = 2 * 1024 * 1024 * 1024
    max_queued_jobs: int = 4
    gpu_concurrency: int = 1
    job_timeout_sec: float = 3600.0
    cancel_grace_sec: float = 15.0
    capability_cache_sec: float = 120.0
    uvicorn_workers: int = 1

    @classmethod
    def from_env(cls) -> ApiSettings:
        s = cls()
        s.host = os.environ.get("VISIONFORGE_API_HOST", s.host)
        s.port = _env_int("VISIONFORGE_API_PORT", s.port)
        s.allow_non_loopback = os.environ.get(
            "VISIONFORGE_API_ALLOW_NON_LOOPBACK", ""
        ).strip().lower() in {"1", "true", "yes"}
        s.jobs_root = _env_path("VISIONFORGE_API_JOBS_ROOT", s.jobs_root)
        s.pipeline_output_root = _env_path(
            "VISIONFORGE_API_PIPELINE_ROOT", s.pipeline_output_root
        )
        s.python_executable = _env_path(
            "VISIONFORGE_API_PYTHON", s.python_executable
        )
        s.project_root = _env_path("VISIONFORGE_PROJECT_ROOT", s.project_root)
        s.max_upload_bytes = _env_int("VISIONFORGE_API_MAX_UPLOAD_BYTES", s.max_upload_bytes)
        s.max_zip_entries = _env_int("VISIONFORGE_API_MAX_ZIP_ENTRIES", s.max_zip_entries)
        s.max_zip_uncompressed_bytes = _env_int(
            "VISIONFORGE_API_MAX_ZIP_UNCOMPRESSED", s.max_zip_uncompressed_bytes
        )
        s.max_image_frames = _env_int("VISIONFORGE_API_MAX_FRAMES_UPLOAD", s.max_image_frames)
        s.max_labels = _env_int("VISIONFORGE_API_MAX_LABELS", s.max_labels)
        s.max_label_length = _env_int("VISIONFORGE_API_MAX_LABEL_LEN", s.max_label_length)
        s.max_max_frames = _env_int("VISIONFORGE_API_MAX_MAX_FRAMES", s.max_max_frames)
        s.max_processing_side = _env_int(
            "VISIONFORGE_API_MAX_PROCESS_SIDE", s.max_processing_side
        )
        s.min_free_disk_bytes = _env_int(
            "VISIONFORGE_API_MIN_FREE_DISK", s.min_free_disk_bytes
        )
        s.max_queued_jobs = _env_int("VISIONFORGE_API_MAX_QUEUED", s.max_queued_jobs)
        s.gpu_concurrency = _env_int("VISIONFORGE_API_GPU_CONCURRENCY", s.gpu_concurrency)
        s.job_timeout_sec = _env_float("VISIONFORGE_API_JOB_TIMEOUT_SEC", s.job_timeout_sec)
        s.cancel_grace_sec = _env_float(
            "VISIONFORGE_API_CANCEL_GRACE_SEC", s.cancel_grace_sec
        )
        s.capability_cache_sec = _env_float(
            "VISIONFORGE_API_CAPABILITY_CACHE_SEC", s.capability_cache_sec
        )
        s.uvicorn_workers = _env_int("VISIONFORGE_API_UVICORN_WORKERS", s.uvicorn_workers)
        if s.gpu_concurrency != 1:
            raise ValueError("VISIONFORGE_API_GPU_CONCURRENCY must be 1 on this machine")
        if s.uvicorn_workers != 1:
            raise ValueError(
                "VISIONFORGE_API_UVICORN_WORKERS must be 1 "
                "(multiple workers would create independent GPU queues)"
            )
        return s
