"""Single-GPU in-process job queue."""
from __future__ import annotations

import threading
from collections import deque

from visionforge.api.config import ApiSettings
from visionforge.api.errors import ApiError
from visionforge.api.jobs.store import JobStore
from visionforge.api.jobs.worker import run_job_subprocess
from visionforge.api.schemas import JobStatus


class JobQueue:
    def __init__(self, store: JobStore, settings: ApiSettings) -> None:
        self.store = store
        self.settings = settings
        self._q: deque[str] = deque()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._active: str | None = None
        self._cancel_flags: dict[str, threading.Event] = {}
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, name="vf-gpu-worker", daemon=True)
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def enqueue(self, job_id: str) -> int:
        with self._cv:
            if len(self._q) >= self.settings.max_queued_jobs:
                raise ApiError(
                    "QUEUE_FULL",
                    "Job queue is full; retry later",
                    status_code=429,
                    extra={"retry_after_sec": 30},
                )
            if job_id in self._q or job_id == self._active:
                raise ApiError(
                    "DUPLICATE_JOB",
                    "Job is already queued or running",
                    status_code=409,
                    job_id=job_id,
                )
            self._q.append(job_id)
            self._cancel_flags[job_id] = threading.Event()
            pos = len(self._q)  # 1-based among queued only
            self._cv.notify()
            return pos

    def queue_position(self, job_id: str) -> int | None:
        with self._lock:
            try:
                return list(self._q).index(job_id) + 1
            except ValueError:
                return None

    def queued_count(self) -> int:
        with self._lock:
            return len(self._q)

    def active_job_id(self) -> str | None:
        with self._lock:
            return self._active

    def is_accepting(self) -> bool:
        with self._lock:
            return len(self._q) < self.settings.max_queued_jobs

    def request_cancel(self, job_id: str) -> None:
        with self._lock:
            flag = self._cancel_flags.get(job_id)
            if flag is not None:
                flag.set()
            # Remove from queue if present
            if job_id in self._q:
                self._q = deque(j for j in self._q if j != job_id)
                self._cv.notify_all()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while not self._q and not self._stop.is_set():
                    self._cv.wait(timeout=0.5)
                if self._stop.is_set():
                    return
                job_id = self._q.popleft()
                self._active = job_id
                cancel_event = self._cancel_flags.setdefault(job_id, threading.Event())

            # Check still queued status
            state = self.store.load(job_id)
            if state is None:
                with self._lock:
                    self._active = None
                continue
            if state.get("status") == JobStatus.CANCELLED.value:
                with self._lock:
                    self._active = None
                continue
            if cancel_event.is_set():
                self.store.update(
                    job_id,
                    status=JobStatus.CANCELLED.value,
                    stage="cancelled",
                    error_code="CANCELLED",
                    error_message="Job cancelled before start",
                )
                with self._lock:
                    self._active = None
                continue

            try:
                run_job_subprocess(
                    job_id=job_id,
                    store=self.store,
                    settings=self.settings,
                    cancel_event=cancel_event,
                )
            except Exception as exc:
                self.store.update(
                    job_id,
                    status=JobStatus.FAILED.value,
                    stage="failed",
                    error_code="WORKER_ERROR",
                    error_message=f"{type(exc).__name__}: {exc}"[:500],
                )
            finally:
                with self._lock:
                    self._active = None
                    self._cv.notify_all()


def recover_queue(queue: JobQueue, store: JobStore) -> None:
    for job_id in store.discover_and_recover():
        state = store.load(job_id)
        if state and state.get("status") == JobStatus.QUEUED.value:
            try:
                queue.enqueue(job_id)
            except ApiError:
                store.update(
                    job_id,
                    status=JobStatus.FAILED.value,
                    error_code="QUEUE_FULL",
                    error_message="Could not requeue after restart",
                    stage="failed",
                )
