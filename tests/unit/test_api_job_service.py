"""Unit tests for API job specs, uploads, store, progress, artifacts, commands."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from visionforge.api.config import ApiSettings
from visionforge.api.errors import ApiError
from visionforge.api.jobs.artifacts import (
    build_artifacts_zip,
    list_allowed_artifacts,
    safe_job_file,
)
from visionforge.api.jobs.progress_tail import parse_progress_line, read_all_progress
from visionforge.api.jobs.queue import JobQueue
from visionforge.api.jobs.store import JobStore, atomic_write_json
from visionforge.api.jobs.worker import build_pipeline_command, pipeline_env
from visionforge.api.schemas import JobSpec, JobStatus, sanitize_filename
from visionforge.api.upload import extract_zip_frames, validate_box_against_first_frame
from visionforge.pipeline.progress import JsonlProgressFile, overall_percent_for


def test_job_spec_rejects_unknown_tracker() -> None:
    with pytest.raises(Exception):
        JobSpec.model_validate(
            {
                "tracker": "magic",
                "box": [1, 2, 3, 4],
                "labels": ["a"],
            }
        )


def test_job_spec_box_and_labels() -> None:
    with pytest.raises(Exception):
        JobSpec.model_validate(
            {"tracker": "edgetam", "box": [1, 1, 1, 2], "labels": ["a"]}
        )
    with pytest.raises(Exception):
        JobSpec.model_validate(
            {"tracker": "edgetam", "box": [0, 0, 10, 10], "labels": ["a\x00b"]}
        )
    with pytest.raises(Exception):
        JobSpec.model_validate(
            {"tracker": "edgetam", "box": [float("nan"), 0, 1, 1], "labels": ["a"]}
        )
    ok = JobSpec.model_validate(
        {
            "tracker": "sam31",
            "box": [20, 60, 60, 100],
            "labels": ["a red circle", "a green rectangle"],
            "max_frames": 8,
            "analysis_mode": "sampled",
            "mask_confirmed": True,
        }
    )
    assert ok.tracker == "sam31"


def test_job_spec_requires_mask_confirmed() -> None:
    with pytest.raises(Exception):
        JobSpec.model_validate(
            {
                "tracker": "edgetam",
                "box": [0, 0, 10, 10],
                "labels": ["a"],
                "mask_confirmed": False,
            }
        )


def test_job_spec_rejects_extra_fields() -> None:
    with pytest.raises(Exception):
        JobSpec.model_validate(
            {
                "tracker": "edgetam",
                "box": [0, 0, 1, 1],
                "labels": ["a"],
                "mask_confirmed": True,
                "evil": True,
            }
        )


def test_sanitize_filename() -> None:
    assert ".." not in sanitize_filename("../x.png")
    assert sanitize_filename("") == "upload.bin"


def test_uuid_job_ids_and_atomic_state(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    a = store.create_job(
        tracker="edgetam",
        spec={"tracker": "edgetam", "box": [0, 0, 1, 1], "labels": ["a"]},
        original_filename="f.zip",
        input_kind="zip",
    )
    b = store.create_job(
        tracker="sam31",
        spec={"tracker": "sam31", "box": [0, 0, 1, 1], "labels": ["a"]},
        original_filename="f.zip",
        input_kind="zip",
    )
    assert a["job_id"] != b["job_id"]
    assert len(a["job_id"]) == 36
    store.update(a["job_id"], status=JobStatus.RUNNING.value)
    loaded = store.load(a["job_id"])
    assert loaded is not None
    assert loaded["status"] == JobStatus.RUNNING.value


def test_illegal_terminal_transition(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    st = store.create_job(
        tracker="edgetam",
        spec={"tracker": "edgetam", "box": [0, 0, 1, 1], "labels": ["a"]},
        original_filename="f.zip",
        input_kind="zip",
    )
    store.update(st["job_id"], status=JobStatus.SUCCEEDED.value)
    with pytest.raises(ValueError, match="terminal"):
        store.transition(st["job_id"], JobStatus.RUNNING)


def test_queue_full(tmp_path: Path) -> None:
    settings = ApiSettings(jobs_root=tmp_path / "jobs", max_queued_jobs=1)
    store = JobStore(settings.jobs_root)
    q = JobQueue(store, settings)
    # Don't start worker thread for this test — enqueue only
    q._started = True
    j1 = store.create_job(
        tracker="edgetam",
        spec={"tracker": "edgetam", "box": [0, 0, 1, 1], "labels": ["a"]},
        original_filename="a.zip",
        input_kind="zip",
    )
    j2 = store.create_job(
        tracker="edgetam",
        spec={"tracker": "edgetam", "box": [0, 0, 1, 1], "labels": ["a"]},
        original_filename="b.zip",
        input_kind="zip",
    )
    assert q.enqueue(j1["job_id"]) == 1
    with pytest.raises(ApiError) as ei:
        q.enqueue(j2["job_id"])
    assert ei.value.code == "QUEUE_FULL"
    assert ei.value.status_code == 429


def test_queued_cancel_removes_from_queue(tmp_path: Path) -> None:
    settings = ApiSettings(jobs_root=tmp_path / "jobs", max_queued_jobs=4)
    store = JobStore(settings.jobs_root)
    q = JobQueue(store, settings)
    q._started = True
    j1 = store.create_job(
        tracker="edgetam",
        spec={"tracker": "edgetam", "box": [0, 0, 1, 1], "labels": ["a"]},
        original_filename="a.zip",
        input_kind="zip",
    )
    q.enqueue(j1["job_id"])
    assert q.queue_position(j1["job_id"]) == 1
    q.request_cancel(j1["job_id"])
    assert q.queue_position(j1["job_id"]) is None


def test_progress_parse_and_malformed(tmp_path: Path) -> None:
    p = tmp_path / "progress.jsonl"
    sink = JsonlProgressFile(p)
    sink.emit("tracking", completed=2, total=8, overall_percent=20)
    assert parse_progress_line("not-json") is None
    assert parse_progress_line('{"event":"progress"}') is None
    events = read_all_progress(p)
    assert len(events) == 1
    assert events[0]["stage"] == "tracking"
    assert overall_percent_for("completed") == 100.0


def test_command_no_shell_and_offline_env(tmp_path: Path) -> None:
    settings = ApiSettings(
        python_executable=Path("D:/project/.venvs/smoke/Scripts/python.exe"),
        project_root=Path("D:/project"),
    )
    cmd = build_pipeline_command(
        settings=settings,
        input_path=tmp_path / "frames",
        tracker="edgetam",
        box=[20, 60, 60, 100],
        labels=["a", "b"],
        max_frames=8,
        processing_width=None,
        processing_height=None,
        output_root=tmp_path / "out",
        progress_file=tmp_path / "p.jsonl",
    )
    assert cmd[0].endswith("python.exe") or "python" in cmd[0]
    assert "-m" in cmd
    assert "visionforge.cli.e2e_pipeline" in cmd
    assert all(isinstance(x, str) for x in cmd)
    env = pipeline_env(settings)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def _make_zip(path: Path, names: list[str], *, bad_name: str | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for i, name in enumerate(names):
            img = Image.new("RGB", (32, 32), color=(i * 10, 20, 30))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            zf.writestr(name, buf.getvalue())
        if bad_name:
            zf.writestr(bad_name, b"x")
    return path


def test_zip_path_traversal_rejected(tmp_path: Path) -> None:
    settings = ApiSettings()
    zpath = _make_zip(tmp_path / "t.zip", ["00000.jpg"])
    # rewrite with traversal
    with zipfile.ZipFile(zpath, "w") as zf:
        img = Image.new("RGB", (16, 16), color=(1, 2, 3))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        zf.writestr("../evil.jpg", buf.getvalue())
    with pytest.raises(ApiError) as ei:
        extract_zip_frames(zpath, tmp_path / "out", settings)
    assert ei.value.code == "ZIP_PATH_TRAVERSAL"


def test_zip_symlink_rejected(tmp_path: Path) -> None:
    settings = ApiSettings()
    zpath = tmp_path / "sym.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        info = zipfile.ZipInfo("link.jpg")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        zf.writestr(info, b"")
    with pytest.raises(ApiError) as ei:
        extract_zip_frames(zpath, tmp_path / "out", settings)
    assert ei.value.code == "ZIP_SYMLINK_REJECTED"


def test_zip_bomb_and_entry_limits(tmp_path: Path) -> None:
    settings = ApiSettings(max_zip_entries=2, max_zip_uncompressed_bytes=1000)
    zpath = tmp_path / "many.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for i in range(3):
            zf.writestr(f"{i}.jpg", b"abc")
    with pytest.raises(ApiError) as ei:
        extract_zip_frames(zpath, tmp_path / "out", settings)
    assert ei.value.code == "ZIP_TOO_MANY_ENTRIES"


def test_zip_happy_and_box_validation(tmp_path: Path) -> None:
    settings = ApiSettings()
    zpath = _make_zip(tmp_path / "ok.zip", ["b.jpg", "a.jpg"])
    frames = extract_zip_frames(zpath, tmp_path / "frames", settings)
    assert len(frames) == 2
    assert frames[0].name.startswith("00000")
    validate_box_against_first_frame([1, 1, 10, 10], frames[0])
    with pytest.raises(ApiError) as ei:
        validate_box_against_first_frame([1000, 1000, 1100, 1100], frames[0])
    assert ei.value.code == "BOX_OUTSIDE_FRAME"


def test_artifact_allowlist_and_traversal(tmp_path: Path) -> None:
    from visionforge.api.jobs.artifacts import resolve_artifact_by_id

    job = tmp_path / "job"
    pipe = job / "pipeline" / "run"
    (pipe / "masks").mkdir(parents=True)
    (pipe / "manifest.json").write_text("{}", encoding="utf-8")
    (pipe / "masks" / "mask_00000.png").write_bytes(b"x")
    secret = job / "secret.pt"
    secret.write_bytes(b"weights")
    state = {
        "pipeline_run_dir": str(pipe),
        "status": "succeeded",
    }
    items = list_allowed_artifacts(job, state)
    names = {i["name"] for i in items}
    assert "manifest.json" in names
    assert "secret.pt" not in names
    with pytest.raises(ApiError):
        safe_job_file(job, "../secret.pt")
    zpath = job / "bundle.zip"
    build_artifacts_zip(job, state, zpath)
    assert zpath.is_file()

    mask_item = next(i for i in items if i["name"] == "mask_00000.png")
    resolved = resolve_artifact_by_id(job, state, mask_item["id"])
    assert resolved.name == "mask_00000.png"
    with pytest.raises(ApiError) as ei:
        resolve_artifact_by_id(job, state, "../etc/passwd")
    assert ei.value.code == "INVALID_ARTIFACT_ID"
    with pytest.raises(ApiError) as ei2:
        resolve_artifact_by_id(job, state, "a" * 24)
    assert ei2.value.code == "NOT_FOUND"


def test_server_restart_recovery(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    running = store.create_job(
        tracker="edgetam",
        spec={"tracker": "edgetam", "box": [0, 0, 1, 1], "labels": ["a"]},
        original_filename="a.zip",
        input_kind="zip",
    )
    queued = store.create_job(
        tracker="edgetam",
        spec={"tracker": "edgetam", "box": [0, 0, 1, 1], "labels": ["a"]},
        original_filename="b.zip",
        input_kind="zip",
    )
    store.update(running["job_id"], status=JobStatus.RUNNING.value, pid=99999999)
    requeue = store.discover_and_recover()
    assert queued["job_id"] in requeue
    failed = store.load(running["job_id"])
    assert failed is not None
    assert failed["status"] == JobStatus.FAILED.value
    assert failed["error_code"] in {"SERVER_RESTARTED", "ORPHANED_JOB"}


def test_health_does_not_import_torch(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    settings = ApiSettings(jobs_root=tmp_path / "jobs", min_free_disk_bytes=1)
    from visionforge.api.app import create_app

    app = create_app(settings)
    with patch.dict("sys.modules", {"torch": MagicMock()}):
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"


def test_sanitize_result_no_fallback_flag() -> None:
    from visionforge.api.jobs.worker import sanitize_result

    manifest = {
        "run_id": "r1",
        "selected_tracker_backend": "edgetam",
        "frames": {"processed": 8, "successful": 8, "failed": []},
        "outputs": {"run_dir": str(Path("D:/project/artifacts/e2e_runs"))},
        "stages": {"dinov3": {"feature_shape": [8, 384]}},
        "mobileclip2_summary": {
            "image_feature_shape": [8, 512],
            "text_feature_shape": [3, 512],
            "mean_scores": {"a": 0.1},
            "highest_scoring_aggregate_label": "a",
        },
        "identity_summary": {},
        "timing": {"total_pipeline_sec": 1.0},
        "warnings": [],
        "real_cuda_inference": True,
        "offline_local_only": True,
        "mock_or_fallback_used": False,
    }
    out = sanitize_result(manifest, job_id="jid")
    assert out["mock_or_fallback_used"] is False
    assert out["tracker_runtime"] == "native_windows"


def test_atomic_write_json(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_write_json(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8"))["a"] == 1
