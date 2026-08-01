"""Atomic on-disk job metadata store."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from visionforge.api.schemas import TERMINAL_STATUSES, JobStatus


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, ensure_ascii=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def state_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "state.json"

    def create_job(
        self,
        *,
        tracker: str,
        spec: dict[str, Any],
        original_filename: str,
        input_kind: str,
        parent_job_id: str | None = None,
        revision_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            job_id = str(uuid.uuid4())
            jdir = self.job_dir(job_id)
            if jdir.exists():
                raise RuntimeError("UUID collision")
            (jdir / "input").mkdir(parents=True)
            (jdir / "logs").mkdir(parents=True)
            (jdir / "pipeline").mkdir(parents=True)
            state = {
                "job_id": job_id,
                "tracker": tracker,
                "status": JobStatus.QUEUED.value,
                "stage": "queued",
                "stage_progress": None,
                "overall_percent": 0.0,
                "frames_completed": None,
                "frames_total": None,
                "created_at": utc_now(),
                "started_at": None,
                "finished_at": None,
                "pipeline_run_id": None,
                "pipeline_run_dir": None,
                "pid": None,
                "error_code": None,
                "error_message": None,
                "warning_code": None,
                "runtime": "wsl2" if tracker == "sam31" else "native_windows",
                "wsl_distro": "VisionForge-SAM31" if tracker == "sam31" else None,
                "input_kind": input_kind,
                "original_filename": original_filename,
                "mock_or_fallback_used": False,
                "cancel_requested": False,
                "parent_job_id": parent_job_id,
                "revision_id": revision_id,
            }
            atomic_write_json(self.state_path(job_id), state)
            atomic_write_json(
                jdir / "request.json",
                {
                    "job_id": job_id,
                    "tracker": tracker,
                    "spec": spec,
                    "original_filename": original_filename,
                    "input_kind": input_kind,
                    "created_at": state["created_at"],
                    "parent_job_id": parent_job_id,
                    "revision_id": revision_id,
                },
            )
            return state

    def load(self, job_id: str) -> dict[str, Any] | None:
        path = self.state_path(job_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            state = self.load(job_id)
            if state is None:
                raise KeyError(job_id)
            state.update(fields)
            atomic_write_json(self.state_path(job_id), state)
            return state

    def transition(
        self,
        job_id: str,
        new_status: JobStatus,
        *,
        allowed_from: set[JobStatus] | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        with self._lock:
            state = self.load(job_id)
            if state is None:
                raise KeyError(job_id)
            current = JobStatus(state["status"])
            if allowed_from is not None and current not in allowed_from:
                raise ValueError(f"illegal transition {current.value}->{new_status.value}")
            if current in TERMINAL_STATUSES and new_status not in TERMINAL_STATUSES:
                raise ValueError(f"cannot leave terminal state {current.value}")
            if (
                current in TERMINAL_STATUSES
                and new_status in TERMINAL_STATUSES
                and current != new_status
            ):
                raise ValueError(f"cannot leave terminal state {current.value}")
            state["status"] = new_status.value
            state.update(fields)
            atomic_write_json(self.state_path(job_id), state)
            return state

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return jobs
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            state = self.load(child.name)
            if state:
                jobs.append(state)
        return jobs

    def discover_and_recover(self) -> list[str]:
        """On restart: fail orphaned running jobs; return queued job ids to requeue."""
        requeue: list[str] = []
        for state in self.list_jobs():
            job_id = state["job_id"]
            status = JobStatus(state["status"])
            if status == JobStatus.QUEUED:
                requeue.append(job_id)
            elif status in {JobStatus.RUNNING, JobStatus.CANCELLING}:
                pid = state.get("pid")
                alive = False
                if isinstance(pid, int) and pid > 0:
                    try:
                        import psutil  # optional

                        alive = psutil.pid_exists(pid)
                    except Exception:
                        # Fallback: Windows tasklist / POSIX kill(0)
                        alive = _pid_alive(pid)
                if alive:
                    # Do not start a second copy; mark orphaned after attempt to leave alone
                    # Spec: mark interrupted running as failed after verifying process state.
                    # If still alive from previous server, we cannot safely adopt — mark failed.
                    self.update(
                        job_id,
                        status=JobStatus.FAILED.value,
                        finished_at=utc_now(),
                        error_code="ORPHANED_JOB",
                        error_message=(
                            "Job was running when the API restarted; "
                            "previous process was not safely adopted"
                        ),
                        stage="failed",
                    )
                    # Best-effort: do not kill unknown trees here (may be unrelated).
                else:
                    self.update(
                        job_id,
                        status=JobStatus.FAILED.value,
                        finished_at=utc_now(),
                        error_code="SERVER_RESTARTED",
                        error_message="API restarted while job was running",
                        stage="failed",
                        pid=None,
                    )
            elif status == JobStatus.CANCELLING:
                self.update(
                    job_id,
                    status=JobStatus.CANCELLED.value,
                    finished_at=utc_now(),
                    error_code="SERVER_RESTARTED",
                    error_message="Cancellation interrupted by API restart",
                    stage="cancelled",
                )
        return requeue


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
