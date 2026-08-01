"""Job submit/status/cancel/result/artifact endpoints."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse

from visionforge.api.errors import ApiError, raise_api
from visionforge.api.jobs.artifacts import (
    build_artifacts_zip,
    guess_media_type,
    list_allowed_artifacts,
    resolve_artifact_by_id,
)
from visionforge.api.jobs.store import utc_now
from visionforge.api.schemas import (
    TERMINAL_STATUSES,
    JobAccepted,
    JobSpec,
    JobStatus,
    JobStatusView,
    sanitize_filename,
)
from visionforge.api.upload import (
    detect_input_kind,
    extract_zip_frames,
    sniff_video_or_reject,
    stream_upload_to_file,
    validate_box_against_first_frame,
)

router = APIRouter(prefix="/v1", tags=["jobs"])


def _settings(request: Request):
    return request.app.state.settings


def _store(request: Request):
    return request.app.state.store


def _queue(request: Request):
    return request.app.state.queue


_RESULT_READY = frozenset(
    {
        JobStatus.SUCCEEDED,
        JobStatus.REVIEW_REQUIRED,
        JobStatus.PARTIAL,
        JobStatus.FAILED,
    }
)


def _status_view(request: Request, state: dict) -> JobStatusView:
    queue = _queue(request)
    job_id = state["job_id"]
    status = JobStatus(state["status"])
    pos = queue.queue_position(job_id) if status == JobStatus.QUEUED else None
    result_url = artifacts_url = download_url = None
    if status in _RESULT_READY:
        result_url = f"/v1/jobs/{job_id}/result"
        artifacts_url = f"/v1/jobs/{job_id}/artifacts"
        # Prefer cleaned MP4 download for removal jobs
        op = None
        try:
            req_path = _store(request).job_dir(job_id) / "request.json"
            if req_path.is_file():
                op = json.loads(req_path.read_text(encoding="utf-8")).get("spec", {}).get("operation")
        except Exception:
            op = None
        if op == "remove_object" or status == JobStatus.SUCCEEDED:
            # Still expose cleaned URL when file exists; ZIP remains for debug via artifacts
            cleaned_guess = list((_store(request).job_dir(job_id) / "pipeline").glob("**/cleaned.mp4")) if (_store(request).job_dir(job_id) / "pipeline").is_dir() else []
            download_url = (
                f"/v1/jobs/{job_id}/cleaned"
                if op == "remove_object" or cleaned_guess
                else f"/v1/jobs/{job_id}/download"
            )
        else:
            download_url = f"/v1/jobs/{job_id}/download"
    return JobStatusView(
        job_id=job_id,
        tracker=state.get("tracker"),
        status=status,
        stage=state.get("stage"),
        stage_progress=state.get("stage_progress"),
        overall_percent=state.get("overall_percent"),
        frames_completed=state.get("frames_completed"),
        frames_total=state.get("frames_total"),
        created_at=state.get("created_at"),
        started_at=state.get("started_at"),
        finished_at=state.get("finished_at"),
        queue_position=pos,
        pipeline_run_id=state.get("pipeline_run_id"),
        warning_code=state.get("warning_code"),
        error_code=state.get("error_code"),
        error_message=state.get("error_message"),
        result_url=result_url,
        artifacts_url=artifacts_url,
        download_url=download_url,
        runtime=state.get("runtime"),
    )


@router.post("/jobs", status_code=202, response_model=JobAccepted)
async def submit_job(
    request: Request,
    input: UploadFile = File(...),
    spec: str = Form(...),
) -> JobAccepted:
    settings = _settings(request)
    store = _store(request)
    queue = _queue(request)

    # Disk gate
    try:
        free = shutil.disk_usage(settings.jobs_root).free
    except Exception:
        free = 0
    if free < settings.min_free_disk_bytes:
        raise_api(
            "INSUFFICIENT_DISK",
            "Not enough free disk space for a new job",
            status_code=503,
        )

    try:
        spec_obj = JobSpec.model_validate_json(spec)
    except Exception as exc:
        raise_api("INVALID_SPEC", f"Invalid job spec: {exc}", status_code=422)

    if len(spec_obj.labels) > settings.max_labels:
        raise_api("TOO_MANY_LABELS", "Too many labels", status_code=400)
    for lab in spec_obj.labels:
        if len(lab) > settings.max_label_length:
            raise_api("LABEL_TOO_LONG", "Label exceeds maximum length", status_code=400)
    if spec_obj.max_frames is not None and spec_obj.max_frames > settings.max_max_frames:
        raise_api("MAX_FRAMES_TOO_LARGE", "max_frames exceeds limit", status_code=400)
    if spec_obj.processing_width and (
        spec_obj.processing_width > settings.max_processing_side
        or (spec_obj.processing_height or 0) > settings.max_processing_side
    ):
        raise_api("RESOLUTION_TOO_LARGE", "Processing resolution exceeds limit", status_code=400)

    original = sanitize_filename(input.filename or "upload.bin")
    state = store.create_job(
        tracker=spec_obj.tracker,
        spec=spec_obj.model_dump(),
        original_filename=original,
        input_kind="pending",
    )
    job_id = state["job_id"]
    jdir = store.job_dir(job_id)
    upload_dir = jdir / "input" / "upload"
    upload_dir.mkdir(parents=True, exist_ok=True)
    raw_path = upload_dir / f"raw_{job_id[:8]}"

    try:
        await stream_upload_to_file(input, raw_path, max_bytes=settings.max_upload_bytes)
        kind = detect_input_kind(raw_path, original)
        store.update(job_id, input_kind=kind)

        if kind == "zip":
            frames_root = jdir / "input" / "frames"
            ordered = extract_zip_frames(raw_path, frames_root, settings)
            validate_box_against_first_frame(spec_obj.box, ordered[0])
            (jdir / "input" / "prepared_path.txt").write_text(
                str(ordered[0].parent), encoding="utf-8"
            )
        else:
            # video
            video_path = upload_dir / f"video{Path(original).suffix.lower() or '.mp4'}"
            if video_path.exists():
                video_path.unlink()
            raw_path.replace(video_path)
            sniff_video_or_reject(video_path)
            (jdir / "input" / "prepared_path.txt").write_text(
                str(video_path), encoding="utf-8"
            )
            # Box vs first frame deferred to pipeline for videos

        pos = queue.enqueue(job_id)
        store.update(job_id, queue_position_at_submit=pos)
    except ApiError as exc:
        store.update(
            job_id,
            status=JobStatus.FAILED.value,
            finished_at=utc_now(),
            error_code=exc.code,
            error_message=exc.message,
            stage="failed",
        )
        raise
    except Exception as exc:
        store.update(
            job_id,
            status=JobStatus.FAILED.value,
            finished_at=utc_now(),
            error_code="SUBMIT_FAILED",
            error_message=f"{type(exc).__name__}",
            stage="failed",
        )
        raise_api("SUBMIT_FAILED", "Failed to accept upload", status_code=500, job_id=job_id)

    return JobAccepted(
        job_id=job_id,
        status=JobStatus.QUEUED,
        tracker=spec_obj.tracker,
        status_url=f"/v1/jobs/{job_id}",
    )


@router.get("/jobs")
def list_jobs(request: Request) -> dict:
    store = _store(request)
    items = [_status_view(request, s).model_dump() for s in store.list_jobs()]
    return {"jobs": items}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> JobStatusView:
    state = _store(request).load(job_id)
    if state is None:
        raise_api("NOT_FOUND", "Job not found", status_code=404, job_id=job_id)
    return _status_view(request, state)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict:
    store = _store(request)
    queue = _queue(request)
    state = store.load(job_id)
    if state is None:
        raise_api("NOT_FOUND", "Job not found", status_code=404, job_id=job_id)

    status = JobStatus(state["status"])
    if status in TERMINAL_STATUSES:
        return {
            "job_id": job_id,
            "status": status.value,
            "message": "Job already finished; cancel is idempotent",
        }

    store.update(job_id, cancel_requested=True)
    queue.request_cancel(job_id)

    if status == JobStatus.QUEUED:
        store.update(
            job_id,
            status=JobStatus.CANCELLED.value,
            finished_at=utc_now(),
            stage="cancelled",
            error_code="CANCELLED",
            error_message="Job cancelled while queued",
        )
        return {"job_id": job_id, "status": JobStatus.CANCELLED.value}

    # Running / cancelling
    store.update(job_id, status=JobStatus.CANCELLING.value, stage="cancelling")
    return {"job_id": job_id, "status": JobStatus.CANCELLING.value}


@router.get("/jobs/{job_id}/result")
def job_result(job_id: str, request: Request) -> dict:
    store = _store(request)
    state = store.load(job_id)
    if state is None:
        raise_api("NOT_FOUND", "Job not found", status_code=404, job_id=job_id)
    status = JobStatus(state["status"])
    if status == JobStatus.CANCELLED:
        raise_api(
            "JOB_CANCELLED",
            "Job was cancelled; no successful result",
            status_code=409,
            job_id=job_id,
        )
    if status not in _RESULT_READY:
        raise_api(
            "JOB_NOT_COMPLETE",
            f"Job is {status.value}; result not ready",
            status_code=409,
            job_id=job_id,
        )
    result_path = store.job_dir(job_id) / "result.json"
    if not result_path.is_file():
        raise_api("RESULT_MISSING", "Result file missing", status_code=500, job_id=job_id)
    return json.loads(result_path.read_text(encoding="utf-8"))


@router.get("/jobs/{job_id}/artifacts")
def job_artifacts(job_id: str, request: Request) -> dict:
    store = _store(request)
    state = store.load(job_id)
    if state is None:
        raise_api("NOT_FOUND", "Job not found", status_code=404, job_id=job_id)
    if JobStatus(state["status"]) not in _RESULT_READY:
        raise_api(
            "JOB_NOT_COMPLETE",
            "Artifacts available only for completed jobs with outputs",
            status_code=409,
            job_id=job_id,
        )
    items = list_allowed_artifacts(store.job_dir(job_id), state)
    # Rewrite preview URLs with real job id
    for item in items:
        if item.get("preview_url"):
            item["preview_url"] = f"/v1/jobs/{job_id}/artifacts/{item['id']}"
    return {"job_id": job_id, "artifacts": items}


@router.get("/jobs/{job_id}/artifacts/{artifact_id}")
def get_artifact(job_id: str, artifact_id: str, request: Request):
    store = _store(request)
    state = store.load(job_id)
    if state is None:
        raise_api("NOT_FOUND", "Job not found", status_code=404, job_id=job_id)
    if JobStatus(state["status"]) not in _RESULT_READY:
        raise_api(
            "JOB_NOT_COMPLETE",
            "Artifacts available only for completed jobs with outputs",
            status_code=409,
            job_id=job_id,
        )
    path = resolve_artifact_by_id(store.job_dir(job_id), state, artifact_id)
    return FileResponse(
        path=path,
        media_type=guess_media_type(path),
        filename=path.name,
    )


@router.get("/jobs/{job_id}/download")
def download_artifacts(job_id: str, request: Request):
    store = _store(request)
    state = store.load(job_id)
    if state is None:
        raise_api("NOT_FOUND", "Job not found", status_code=404, job_id=job_id)
    if JobStatus(state["status"]) not in _RESULT_READY:
        raise_api(
            "JOB_NOT_COMPLETE",
            "Download available only for completed jobs with outputs",
            status_code=409,
            job_id=job_id,
        )
    dest = store.job_dir(job_id) / "artifacts_bundle.zip"
    build_artifacts_zip(store.job_dir(job_id), state, dest)
    return FileResponse(
        path=dest,
        media_type="application/zip",
        filename=f"visionforge-job-{job_id}.zip",
    )


@router.get("/jobs/{job_id}/cleaned")
def download_cleaned(job_id: str, request: Request):
    """Serve cleaned.mp4 for remove_object jobs (falls back to common output paths)."""
    store = _store(request)
    state = store.load(job_id)
    if state is None:
        raise_api("NOT_FOUND", "Job not found", status_code=404, job_id=job_id)
    if JobStatus(state["status"]) not in {
        JobStatus.SUCCEEDED,
        JobStatus.REVIEW_REQUIRED,
        JobStatus.PARTIAL,
    }:
        raise_api(
            "JOB_NOT_COMPLETE",
            "Cleaned video available only for successful jobs",
            status_code=409,
            job_id=job_id,
        )
    jdir = store.job_dir(job_id)
    candidates = [
        jdir / "pipeline" / "cleaned.mp4",
        jdir / "cleaned.mp4",
        jdir / "pipeline" / "outputs" / "cleaned.mp4",
    ]
    # Also search under pipeline run dirs
    pipe = jdir / "pipeline"
    if pipe.is_dir():
        candidates.extend(sorted(pipe.glob("**/cleaned.mp4")))
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise_api(
            "CLEANED_MISSING",
            "Cleaned video not found for this job",
            status_code=404,
            job_id=job_id,
        )
    return FileResponse(
        path=path,
        media_type="video/mp4",
        filename=f"cleaned-{job_id}.mp4",
        content_disposition_type="inline",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )


@router.post("/jobs/{job_id}/corrections", status_code=202, response_model=JobAccepted)
async def create_correction(job_id: str, request: Request) -> JobAccepted:
    """Create a child revision job from a corrected box at a frame index.

    Does not overwrite the parent job's manifests or artifacts.
    """
    import uuid as _uuid

    from visionforge.api.schemas import CorrectionSpec

    store = _store(request)
    queue = _queue(request)
    settings = _settings(request)
    parent = store.load(job_id)
    if parent is None:
        raise_api("NOT_FOUND", "Parent job not found", status_code=404, job_id=job_id)
    if JobStatus(parent["status"]) not in _RESULT_READY:
        raise_api(
            "JOB_NOT_COMPLETE",
            "Corrections require a completed parent with outputs",
            status_code=409,
            job_id=job_id,
        )
    try:
        body = await request.json()
        corr = CorrectionSpec.model_validate(body)
    except Exception as exc:
        raise_api("INVALID_CORRECTION", f"Invalid correction: {exc}", status_code=422)

    revision_id = str(_uuid.uuid4())
    spec = {
        "tracker": corr.tracker,
        "box": corr.box,
        "labels": corr.labels,
        "max_frames": corr.max_frames,
        "processing_width": None,
        "processing_height": None,
        "analysis_mode": corr.analysis_mode,
        "selection_mode": "manual",
        "mask_confirmed": True,
        "start_frame": corr.frame_index,
        "parent_job_id": job_id,
        "revision_id": revision_id,
    }
    child = store.create_job(
        tracker=corr.tracker,
        spec=spec,
        original_filename=parent.get("original_filename") or "parent_input",
        input_kind=parent.get("input_kind") or "video",
        parent_job_id=job_id,
        revision_id=revision_id,
    )
    child_id = child["job_id"]
    parent_dir = store.job_dir(job_id)
    child_dir = store.job_dir(child_id)
    # Copy prepared input without mutating parent
    src_upload = parent_dir / "input" / "upload"
    dst_upload = child_dir / "input" / "upload"
    dst_upload.mkdir(parents=True, exist_ok=True)
    copied = False
    if src_upload.is_dir():
        for p in src_upload.iterdir():
            if p.is_file():
                shutil.copy2(p, dst_upload / p.name)
                copied = True
    src_frames = parent_dir / "input" / "frames"
    if src_frames.is_dir():
        dst_frames = child_dir / "input" / "frames"
        if dst_frames.exists():
            shutil.rmtree(dst_frames, ignore_errors=True)
        shutil.copytree(src_frames, dst_frames)
        copied = True
        ordered = dst_frames / "ordered"
        if ordered.is_dir():
            (child_dir / "input" / "prepared_path.txt").write_text(
                str(ordered), encoding="utf-8"
            )
    if not copied:
        prepared = parent_dir / "input" / "prepared_path.txt"
        if prepared.is_file():
            shutil.copy2(prepared, child_dir / "input" / "prepared_path.txt")
            copied = True
    if not copied:
        store.update(
            child_id,
            status=JobStatus.FAILED.value,
            finished_at=utc_now(),
            error_code="CORRECTION_INPUT_MISSING",
            error_message="Could not copy parent input",
            stage="failed",
        )
        raise_api(
            "CORRECTION_INPUT_MISSING",
            "Parent input artifacts unavailable for revision",
            status_code=500,
            job_id=child_id,
        )

    # Ensure video prepared_path for video uploads
    videos = list(dst_upload.glob("video.*")) if dst_upload.is_dir() else []
    if videos and not (child_dir / "input" / "prepared_path.txt").is_file():
        (child_dir / "input" / "prepared_path.txt").write_text(
            str(videos[0]), encoding="utf-8"
        )

    queue.enqueue(child_id)
    _ = settings  # reserved for future disk gates already applied at parent
    return JobAccepted(
        job_id=child_id,
        status=JobStatus.QUEUED,
        tracker=corr.tracker,
        status_url=f"/v1/jobs/{child_id}",
        parent_job_id=job_id,
        revision_id=revision_id,
    )
