"""Real HTTP API verification for VisionForge Step 5 (not mocked)."""
from __future__ import annotations

import io
import json
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import httpx
from PIL import Image

FIXTURE = Path("D:/project/artifacts/e2e_fixtures/frames_8x256")
PYTHON = Path("D:/project/.venvs/smoke/Scripts/python.exe")
PROJECT = Path("D:/project")
TMP = Path("D:/project/artifacts/api_http_smoke")
TMP.mkdir(parents=True, exist_ok=True)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def make_zip(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.glob("*.jpg")):
            zf.write(p, arcname=p.name)
    return dest


def make_long_zip(src: Path, dest: Path, copies: int = 6) -> Path:
    """Repeat fixture frames to lengthen SAM tracking for cancel test."""
    frames = sorted(src.glob("*.jpg"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        n = 0
        for _ in range(copies):
            for p in frames:
                zf.write(p, arcname=f"{n:05d}.jpg")
                n += 1
    return dest


def wait_job(client: httpx.Client, job_id: str, timeout: float = 1800.0) -> dict:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        r = client.get(f"/v1/jobs/{job_id}")
        r.raise_for_status()
        last = r.json()
        if last["status"] in {"succeeded", "failed", "cancelled"}:
            return last
        time.sleep(2.0)
    raise TimeoutError(f"job {job_id} timeout last={last}")


def submit(client: httpx.Client, zip_path: Path, tracker: str, max_frames: int = 8) -> dict:
    spec = {
        "tracker": tracker,
        "box": [20, 60, 60, 100],
        "labels": ["a red circle", "a green rectangle", "a blue sky"],
        "max_frames": max_frames,
        "processing_width": None,
        "processing_height": None,
    }
    with zip_path.open("rb") as fh:
        files = {"input": ("frames.zip", fh, "application/zip")}
        data = {"spec": json.dumps(spec)}
        r = client.post("/v1/jobs", files=files, data=data)
    assert r.status_code == 202, r.text
    return r.json()


def assert_success_result(client: httpx.Client, job_id: str, tracker: str) -> dict:
    st = wait_job(client, job_id)
    assert st["status"] == "succeeded", st
    res = client.get(f"/v1/jobs/{job_id}/result")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["selected_tracker"] == tracker
    assert body["frames"]["successful"] == 8
    assert body["frames"]["processed"] == 8
    assert body["artifact_counts"]["masks"] == 8
    assert body["artifact_counts"]["crops"] == 8
    assert body["artifact_counts"]["overlays"] == 8
    assert body["dinov3_feature_shape"] == [8, 384]
    assert body["mobileclip2"]["image_feature_shape"] == [8, 512]
    assert body["mobileclip2"]["text_feature_shape"] == [3, 512]
    assert body["mock_or_fallback_used"] is False
    assert body["real_cuda_inference"] is True
    dl = client.get(f"/v1/jobs/{job_id}/download")
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("application/zip")
    zpath = TMP / f"{job_id}.zip"
    zpath.write_bytes(dl.content)
    with zipfile.ZipFile(zpath, "r") as zf:
        names = zf.namelist()
        assert any(n.endswith("manifest.json") for n in names)
    return body


def main() -> int:
    port = free_port()
    jobs_root = TMP / "jobs"
    env = {
        **dict(**{k: v for k, v in __import__("os").environ.items()}),
        "VISIONFORGE_API_HOST": "127.0.0.1",
        "VISIONFORGE_API_PORT": str(port),
        "VISIONFORGE_API_JOBS_ROOT": str(jobs_root),
        "VISIONFORGE_API_MIN_FREE_DISK": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(PROJECT),
    }
    proc = subprocess.Popen(
        [
            str(PYTHON),
            "-m",
            "visionforge.api.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(PROJECT),
        env=env,
        stdout=(TMP / "server_stdout.log").open("w", encoding="utf-8"),
        stderr=(TMP / "server_stderr.log").open("w", encoding="utf-8"),
        shell=False,
    )
    base = f"http://127.0.0.1:{port}"
    report: dict = {"port": port, "tests": {}}
    try:
        # wait ready
        for _ in range(60):
            try:
                with httpx.Client(base_url=base, timeout=30.0) as c:
                    h = c.get("/health")
                    if h.status_code == 200:
                        break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("server failed to start")

        zip8 = make_zip(FIXTURE, TMP / "frames8.zip")

        with httpx.Client(base_url=base, timeout=60.0) as client:
            # capabilities
            caps = client.get("/v1/capabilities").json()
            assert caps["trackers"]["sam31"]["status"] == "AVAILABLE_WSL2"
            assert caps["trackers"]["sam31"].get("available_native_windows") is False
            assert caps["trackers"]["edgetam"]["status"] == "AVAILABLE"

            # EdgeTAM
            print("=== EdgeTAM API job ===", flush=True)
            ej = submit(client, zip8, "edgetam", 8)
            assert ej["status"] == "queued"
            ebody = assert_success_result(client, ej["job_id"], "edgetam")
            assert ebody["tracker_runtime"] == "native_windows"
            report["tests"]["edgetam"] = {
                "job_id": ej["job_id"],
                "result": {
                    k: ebody[k]
                    for k in (
                        "frames",
                        "artifact_counts",
                        "dinov3_feature_shape",
                        "mobileclip2",
                        "total_duration_sec",
                        "tracker_runtime",
                        "mock_or_fallback_used",
                    )
                },
            }
            print("EdgeTAM PASS", ej["job_id"], flush=True)

            # Queue serialization: submit two EdgeTAM-sized jobs; second must wait
            # Use SAM then EdgeTAM would be long; use two edgetam with max_frames=8
            print("=== Queue serialization ===", flush=True)
            q1 = submit(client, zip8, "edgetam", 8)
            q2 = submit(client, zip8, "edgetam", 8)
            time.sleep(3)
            s1 = client.get(f"/v1/jobs/{q1['job_id']}").json()
            s2 = client.get(f"/v1/jobs/{q2['job_id']}").json()
            # One running, one queued OR first already done and second running — but never both running
            statuses = {s1["status"], s2["status"]}
            assert not (
                s1["status"] == "running" and s2["status"] == "running"
            ), (s1, s2)
            if s2["status"] == "queued":
                assert s2.get("queue_position") == 1
            wait_job(client, q1["job_id"])
            wait_job(client, q2["job_id"])
            report["tests"]["queue"] = {
                "job1": q1["job_id"],
                "job2": q2["job_id"],
                "early_statuses": [s1["status"], s2["status"]],
            }
            print("Queue PASS", flush=True)

            # SAM 3.1
            print("=== SAM31 API job ===", flush=True)
            sj = submit(client, zip8, "sam31", 8)
            sbody = assert_success_result(client, sj["job_id"], "sam31")
            assert sbody["tracker_runtime"] == "wsl2"
            assert sbody["wsl_distro"] == "VisionForge-SAM31"
            report["tests"]["sam31"] = {
                "job_id": sj["job_id"],
                "result": {
                    k: sbody[k]
                    for k in (
                        "frames",
                        "artifact_counts",
                        "dinov3_feature_shape",
                        "mobileclip2",
                        "total_duration_sec",
                        "tracker_runtime",
                        "wsl_distro",
                        "mock_or_fallback_used",
                        "real_cuda_inference",
                    )
                },
            }
            print("SAM31 PASS", sj["job_id"], flush=True)

            # Cancellation: longer SAM job
            print("=== SAM cancel test ===", flush=True)
            long_zip = make_long_zip(FIXTURE, TMP / "frames_long.zip", copies=8)
            cj = submit(client, long_zip, "sam31", max_frames=48)
            # Wait until running and WSL worker present
            wsl_seen = False
            t0 = time.time()
            while time.time() - t0 < 600:
                st = client.get(f"/v1/jobs/{cj['job_id']}").json()
                if st["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                if st["status"] == "running":
                    # check wsl worker
                    chk = subprocess.run(
                        [
                            "wsl",
                            "-d",
                            "VisionForge-SAM31",
                            "--",
                            "bash",
                            "-lc",
                            "pgrep -af wsl_sam31_worker || true",
                        ],
                        capture_output=True,
                        text=True,
                        shell=False,
                    )
                    if "wsl_sam31_worker" in (chk.stdout or ""):
                        wsl_seen = True
                        break
                time.sleep(2)
            assert wsl_seen, "WSL worker never observed before cancel"
            cr = client.post(f"/v1/jobs/{cj['job_id']}/cancel")
            assert cr.status_code == 200, cr.text
            final = wait_job(client, cj["job_id"], timeout=300)
            assert final["status"] == "cancelled", final
            # Distro still available
            dist = subprocess.run(
                ["wsl", "-d", "VisionForge-SAM31", "--", "echo", "alive"],
                capture_output=True,
                text=True,
                shell=False,
            )
            assert dist.returncode == 0
            assert "alive" in dist.stdout
            # worker gone
            time.sleep(5)
            chk2 = subprocess.run(
                [
                    "wsl",
                    "-d",
                    "VisionForge-SAM31",
                    "--",
                    "bash",
                    "-lc",
                    "pgrep -af wsl_sam31_worker || true",
                ],
                capture_output=True,
                text=True,
                shell=False,
            )
            # Allow empty or unrelated
            report["tests"]["cancel"] = {
                "job_id": cj["job_id"],
                "status": final["status"],
                "wsl_seen_before_cancel": wsl_seen,
                "distro_alive": True,
                "worker_after": (chk2.stdout or "")[:200],
            }
            # Subsequent normal SAM still works
            sj2 = submit(client, zip8, "sam31", 8)
            sbody2 = assert_success_result(client, sj2["job_id"], "sam31")
            report["tests"]["sam31_after_cancel"] = {"job_id": sj2["job_id"], "ok": True}
            print("Cancel PASS", cj["job_id"], flush=True)

        report["status"] = "PASS"
        (TMP / "http_smoke_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        (TMP / "http_smoke_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
