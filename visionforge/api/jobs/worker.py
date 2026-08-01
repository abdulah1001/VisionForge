"""Build CLI argv and run one pipeline job as a controlled subprocess."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from visionforge.api.config import ApiSettings
from visionforge.api.jobs.progress_tail import tail_progress
from visionforge.api.jobs.store import JobStore, utc_now
from visionforge.api.schemas import JobStatus

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000


def build_pipeline_command(
    *,
    settings: ApiSettings,
    input_path: Path,
    tracker: str,
    box: list[float],
    labels: list[str],
    max_frames: int | None,
    processing_width: int | None,
    processing_height: int | None,
    output_root: Path,
    progress_file: Path,
    run_id: str | None = None,
    start_frame: int | None = None,
    parent_job_id: str | None = None,
    revision_id: str | None = None,
    operation: str = "track_analyze",
    selected_label: str | None = None,
    quality_mode: str = "standard",
) -> list[str]:
    if operation == "remove_object":
        max_side = 1280
        if processing_width and processing_height:
            max_side = max(int(processing_width), int(processing_height))
        cmd: list[str] = [
            str(settings.python_executable),
            "-m",
            "visionforge.cli.remove_object",
            "--input",
            str(input_path),
            "--box",
            str(box[0]),
            str(box[1]),
            str(box[2]),
            str(box[3]),
            "--tracker",
            tracker,
            "--output-root",
            str(output_root),
            "--progress-file",
            str(progress_file),
            "--quality",
            quality_mode if quality_mode in {"standard", "high"} else "standard",
            "--max-side",
            str(max_side),
        ]
        if selected_label:
            cmd.extend(["--label", str(selected_label)])
        if max_frames is not None:
            cmd.extend(["--max-frames", str(int(max_frames))])
        if run_id:
            cmd.extend(["--run-id", str(run_id)])
        return cmd

    cmd = [
        str(settings.python_executable),
        "-m",
        "visionforge.cli.e2e_pipeline",
        "--input",
        str(input_path),
        "--box",
        str(box[0]),
        str(box[1]),
        str(box[2]),
        str(box[3]),
        "--labels",
        ",".join(labels),
        "--tracker",
        tracker,
        "--output-root",
        str(output_root),
        "--progress-file",
        str(progress_file),
    ]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(int(max_frames))])
    if start_frame is not None and int(start_frame) > 0:
        cmd.extend(["--start-frame", str(int(start_frame))])
    if processing_width is not None and processing_height is not None:
        cmd.extend(["--width", str(int(processing_width))])
        cmd.extend(["--height", str(int(processing_height))])
    if run_id:
        cmd.extend(["--run-id", str(run_id)])
    if parent_job_id:
        cmd.extend(["--parent-job-id", str(parent_job_id)])
    if revision_id:
        cmd.extend(["--revision-id", str(revision_id)])
    return cmd


def pipeline_env(settings: ApiSettings) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # Prefer project on path
    pp = str(settings.project_root)
    env["PYTHONPATH"] = pp + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _terminate_process_tree(proc: subprocess.Popen, *, grace_sec: float) -> int | None:
    if proc.poll() is not None:
        return proc.returncode
    try:
        if os.name == "nt":
            # Graceful CTRL_BREAK to process group, then taskkill tree.
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                proc.wait(timeout=grace_sec)
                return proc.returncode
            except subprocess.TimeoutExpired:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                    shell=False,
                )
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=grace_sec)
                return proc.returncode
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    return proc.poll()


def run_job_subprocess(
    *,
    job_id: str,
    store: JobStore,
    settings: ApiSettings,
    cancel_event,
) -> None:
    state = store.load(job_id)
    if state is None:
        return
    req = json.loads((store.job_dir(job_id) / "request.json").read_text(encoding="utf-8"))
    spec = req["spec"]
    jdir = store.job_dir(job_id)
    input_dir = jdir / "input"
    # Resolve prepared input: frames dir or video file
    ordered = input_dir / "frames" / "ordered"
    upload_dir = input_dir / "upload"
    video_candidates = (
        list(upload_dir.glob("video.*")) if upload_dir.is_dir() else []
    )
    if ordered.is_dir() and any(ordered.iterdir()):
        input_path = ordered
    elif video_candidates:
        input_path = video_candidates[0]
    else:
        # fallback: any prepared path recorded
        prepared = input_dir / "prepared_path.txt"
        if prepared.is_file():
            input_path = Path(prepared.read_text(encoding="utf-8").strip())
        else:
            store.update(
                job_id,
                status=JobStatus.FAILED.value,
                finished_at=utc_now(),
                error_code="INPUT_MISSING",
                error_message="Prepared job input not found",
                stage="failed",
            )
            return

    progress_file = jdir / "logs" / "progress.jsonl"
    stdout_path = jdir / "logs" / "stdout.log"
    stderr_path = jdir / "logs" / "stderr.log"
    pipeline_root = jdir / "pipeline"

    cmd = build_pipeline_command(
        settings=settings,
        input_path=input_path,
        tracker=str(spec["tracker"]),
        box=list(spec["box"]),
        labels=list(spec.get("labels") or []),
        max_frames=spec.get("max_frames"),
        processing_width=spec.get("processing_width"),
        processing_height=spec.get("processing_height"),
        output_root=pipeline_root,
        progress_file=progress_file,
        start_frame=spec.get("start_frame"),
        parent_job_id=req.get("parent_job_id") or spec.get("parent_job_id"),
        revision_id=req.get("revision_id") or spec.get("revision_id"),
        operation=str(spec.get("operation") or "track_analyze"),
        selected_label=spec.get("selected_label"),
        quality_mode=str(spec.get("quality_mode") or "standard"),
    )

    store.update(
        job_id,
        status=JobStatus.RUNNING.value,
        started_at=utc_now(),
        stage="validating",
        overall_percent=0.0,
    )

    creationflags = 0
    if os.name == "nt":
        creationflags = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW

    with stdout_path.open("wb") as out_fh, stderr_path.open("wb") as err_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(settings.project_root),
            env=pipeline_env(settings),
            stdout=out_fh,
            stderr=err_fh,
            stdin=subprocess.DEVNULL,
            shell=False,
            creationflags=creationflags,
        )
        store.update(job_id, pid=proc.pid)

        deadline = time.monotonic() + float(settings.job_timeout_sec)
        last_progress: dict[str, Any] | None = None
        while True:
            if cancel_event.is_set() or (store.load(job_id) or {}).get("cancel_requested"):
                store.update(
                    job_id,
                    status=JobStatus.CANCELLING.value,
                    stage="cancelling",
                )
                rc = _terminate_process_tree(proc, grace_sec=settings.cancel_grace_sec)
                store.update(
                    job_id,
                    status=JobStatus.CANCELLED.value,
                    finished_at=utc_now(),
                    stage="cancelled",
                    error_code="CANCELLED",
                    error_message="Job cancelled by client",
                    pid=None,
                    exit_code=rc,
                )
                return

            # Progress tail
            for event in tail_progress(progress_file, last_progress):
                last_progress = event
                fields: dict[str, Any] = {
                    "stage": event.get("stage"),
                    "overall_percent": event.get("overall_percent"),
                    "stage_progress": {
                        "completed": event.get("completed"),
                        "total": event.get("total"),
                    },
                }
                if event.get("completed") is not None:
                    fields["frames_completed"] = event.get("completed")
                if event.get("total") is not None:
                    fields["frames_total"] = event.get("total")
                try:
                    store.update(job_id, **fields)
                except Exception:
                    pass

            rc = proc.poll()
            if rc is not None:
                break
            if time.monotonic() > deadline:
                _terminate_process_tree(proc, grace_sec=settings.cancel_grace_sec)
                store.update(
                    job_id,
                    status=JobStatus.FAILED.value,
                    finished_at=utc_now(),
                    stage="failed",
                    error_code="JOB_TIMEOUT",
                    error_message="Job exceeded configured timeout",
                    pid=None,
                )
                return
            time.sleep(0.5)

    # Child finished
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    manifest_path = None
    run_dir = None
    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if line.startswith("{") and '"status"' in line:
            # May be pretty-printed; try full file JSON parse instead
            break
    try:
        payload = json.loads(stdout_text)
    except json.JSONDecodeError:
        # Pretty JSON from CLI
        try:
            start = stdout_text.find("{")
            end = stdout_text.rfind("}")
            payload = json.loads(stdout_text[start : end + 1]) if start >= 0 else {}
        except Exception:
            payload = {}

    if payload.get("status") == "ok":
        manifest_path = Path(str(payload.get("manifest_path", "")))
        run_dir = Path(str(payload.get("run_dir", "")))
        if not manifest_path.is_file():
            store.update(
                job_id,
                status=JobStatus.FAILED.value,
                finished_at=utc_now(),
                error_code="MANIFEST_MISSING",
                error_message="Pipeline reported ok but manifest is missing",
                stage="failed",
                pid=None,
            )
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("mock_or_fallback_used"):
            store.update(
                job_id,
                status=JobStatus.FAILED.value,
                finished_at=utc_now(),
                error_code="FALLBACK_DETECTED",
                error_message="Pipeline used mock or tracker fallback",
                stage="failed",
                pid=None,
            )
            return
        # Write sanitized result
        result = sanitize_result(manifest, job_id=job_id)
        from visionforge.api.jobs.store import atomic_write_json

        atomic_write_json(jdir / "result.json", result)
        final_status = result.get("status") or "succeeded"
        if final_status not in {
            JobStatus.SUCCEEDED.value,
            JobStatus.REVIEW_REQUIRED.value,
            JobStatus.PARTIAL.value,
            JobStatus.FAILED.value,
        }:
            final_status = JobStatus.SUCCEEDED.value
        store.update(
            job_id,
            status=final_status,
            finished_at=utc_now(),
            stage="completed",
            overall_percent=100.0,
            pipeline_run_id=manifest.get("run_id"),
            pipeline_run_dir=str(run_dir),
            frames_completed=manifest.get("frames", {}).get("successful"),
            frames_total=manifest.get("frames", {}).get("processed"),
            warning_code=(
                "QUALITY_REVIEW"
                if final_status == JobStatus.REVIEW_REQUIRED.value
                else (
                    "PARTIAL_TRACKING"
                    if final_status == JobStatus.PARTIAL.value
                    else None
                )
            ),
            pid=None,
            mock_or_fallback_used=False,
        )
        return

    err = payload.get("error") or "Pipeline failed"
    code = str(payload.get("error_code") or "PIPELINE_FAILED")
    detail = str(payload.get("detail") or "")
    blob = f"{err}\n{detail}".lower()
    if code not in {"GPU_OOM"} and (
        "out of memory" in blob or "outofmemory" in blob or "cuda out of memory" in blob
    ):
        code = "GPU_OOM"
        err = (
            "This video is too large for the available GPU. "
            "Optimize it to 640p and try again."
        )
    store.update(
        job_id,
        status=JobStatus.FAILED.value,
        finished_at=utc_now(),
        stage="failed",
        error_code=code,
        error_message=str(err)[:500],
        pid=None,
    )


def sanitize_result(manifest: dict[str, Any], *, job_id: str) -> dict[str, Any]:
    if manifest.get("operation") == "remove_object":
        audio = manifest.get("audio") or {}
        proc = manifest.get("processing") or {}
        frames = manifest.get("frames") or {}
        return {
            "job_id": job_id,
            "pipeline_run_id": manifest.get("run_id"),
            "status": manifest.get("status") or "succeeded",
            "operation": "remove_object",
            "selected_label": manifest.get("selected_label"),
            "frames": {
                "processed": frames.get("processed"),
                "successful": frames.get("successful"),
            },
            "processing": {
                "width": proc.get("width"),
                "height": proc.get("height"),
                "fps": proc.get("fps"),
                "quality_mode": proc.get("quality_mode"),
            },
            "cleaned_video": {"available": True, "path": "cleaned.mp4"},
            "audio": {
                "preserved": bool(audio.get("audio_preserved")),
                "present_in_source": audio.get("audio_present_in_source"),
            },
            "user_message": manifest.get("user_message") or "Object removed successfully.",
            "mock_or_fallback_used": False,
        }
    frames = manifest.get("frames") or {}
    outputs = manifest.get("outputs") or {}
    run_dir = Path(str(outputs.get("run_dir") or ""))
    mask_n = len(list((run_dir / "masks").glob("mask_*.png"))) if run_dir.is_dir() else 0
    overlay_n = (
        len(list((run_dir / "overlays").glob("overlay_*.jpg"))) if run_dir.is_dir() else 0
    )
    crop_n = len(list((run_dir / "crops").glob("crop_*.jpg"))) if run_dir.is_dir() else 0
    quality = manifest.get("quality") or {}
    inventory = manifest.get("artifact_inventory") or {}
    annotated = manifest.get("annotated_video") or {}
    tracker = manifest.get("selected_tracker_backend")
    runtime = "wsl2" if tracker == "sam31" else "native_windows"
    mclip = manifest.get("mobileclip2_summary") or {}
    stages = manifest.get("stages") or {}
    recommended = (
        manifest.get("recommended_status")
        or quality.get("recommended_status")
        or "succeeded"
    )
    return {
        "job_id": job_id,
        "pipeline_run_id": manifest.get("run_id"),
        "status": recommended,
        "selected_tracker": tracker,
        "tracker_runtime": runtime,
        "wsl_distro": "VisionForge-SAM31" if tracker == "sam31" else None,
        "frames": {
            "processed": frames.get("processed"),
            "successful": frames.get("successful"),
            "failed_count": len(frames.get("failed") or []),
            "valid_masks": quality.get("valid_masks"),
            "invalid_masks": quality.get("invalid_masks"),
            "empty_masks": quality.get("empty_masks"),
        },
        "artifact_counts": {
            "mask_files": inventory.get("mask_files", mask_n),
            "valid_masks": inventory.get("valid_masks", quality.get("valid_masks")),
            "invalid_masks": inventory.get("invalid_masks", quality.get("invalid_masks")),
            "empty_masks": inventory.get("empty_masks", quality.get("empty_masks")),
            "overlays": inventory.get("overlay_files", overlay_n),
            "crops": inventory.get("crop_files", crop_n),
            # legacy keys for older UI
            "masks": inventory.get("valid_masks", mask_n),
        },
        "quality": {
            "recommended_status": recommended,
            "suspected_drift_count": quality.get("suspected_drift_count"),
            "longest_failure_sequence": quality.get("longest_failure_sequence"),
            "frames_requiring_review": quality.get("frames_requiring_review"),
            "valid_ratio": quality.get("valid_ratio"),
            "note": quality.get("note"),
            "warnings": (quality.get("warnings") or [])[:50],
        },
        "annotated_video": {
            "available": bool(annotated),
            "width": annotated.get("width"),
            "height": annotated.get("height"),
            "fps": annotated.get("fps"),
            "frames": annotated.get("frames"),
            "duration_sec": annotated.get("duration_sec"),
            "codec": annotated.get("codec"),
            "pixel_format": annotated.get("pixel_format"),
            "faststart": annotated.get("faststart"),
            "audio": annotated.get("audio"),
            "audio_note": (
                (annotated.get("audio") or {}).get("note")
                or (annotated.get("audio") or {}).get("warning")
                or annotated.get("audio_note")
            ),
            "ffprobe": annotated.get("ffprobe"),
            "artifact_name": "annotated.mp4",
        },
        "dinov3_feature_shape": (stages.get("dinov3") or {}).get("feature_shape"),
        "identity_summary": {
            "vs_first_summary": (manifest.get("identity_summary") or {}).get(
                "vs_first_summary"
            ),
            "consecutive_summary": (manifest.get("identity_summary") or {}).get(
                "consecutive_summary"
            ),
            "note": (
                "DINOv3 similarity measures visual appearance consistency with the "
                "confirmed reference object. It does not guarantee identity."
            ),
        },
        "mobileclip2": {
            "image_feature_shape": mclip.get("image_feature_shape"),
            "text_feature_shape": mclip.get("text_feature_shape"),
            "mean_scores": mclip.get("mean_scores"),
            "highest_scoring_aggregate_label": mclip.get(
                "highest_scoring_aggregate_label"
            ),
            "valid_crops_used": mclip.get("valid_crops_used"),
            "skipped": mclip.get("skipped", False),
            "note": (
                "These scores express similarity between the tracked visual result "
                "and the labels supplied for this job. They are not guaranteed "
                "detections or classifications."
            ),
        },
        "stages_timing": {
            name: {
                "load_time_sec": st.get("load_time_sec"),
                "inference_time_sec": st.get("inference_time_sec"),
                "peak_cuda_allocated_mb": st.get("peak_cuda_allocated_mb"),
                "peak_cuda_reserved_mb": st.get("peak_cuda_reserved_mb"),
            }
            for name, st in stages.items()
            if isinstance(st, dict)
        },
        "total_duration_sec": (manifest.get("timing") or {}).get("total_pipeline_sec"),
        "warnings": list(manifest.get("warnings") or [])[:50],
        "real_cuda_inference": bool(manifest.get("real_cuda_inference")),
        "offline_local_only": bool(manifest.get("offline_local_only")),
        "mock_or_fallback_used": bool(manifest.get("mock_or_fallback_used")),
        "download_url": f"/v1/jobs/{job_id}/download",
    }


# Avoid unused import lint for sys in some paths
_ = sys
