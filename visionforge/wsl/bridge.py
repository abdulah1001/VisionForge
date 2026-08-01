"""Windows ↔ WSL2 helpers for SAM 3.1 execution."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

DEFAULT_DISTRO = "VisionForge-SAM31"
DEFAULT_WSL_PYTHON = "/opt/visionforge/venv/bin/python"
DEFAULT_WORKER = "/mnt/d/project/scripts/wsl_sam31_worker.py"
DEFAULT_CHECKPOINT_WSL = "/mnt/d/project/models/sam31/sam3.1_multiplex.pt"

_PROBE_CACHE: tuple[float, "WSLProbeResult"] | None = None
_PROBE_CACHE_TTL_SEC = 120.0


class WSLBridgeError(Exception):
    pass


def windows_to_wsl_path(path: str | Path) -> str:
    """Convert a Windows path to a WSL /mnt/<drive>/... path."""
    p = Path(path).resolve()
    s = str(p)
    # Handle PureWindowsPath drive
    win = PureWindowsPath(s)
    if win.drive:
        drive = win.drive.rstrip(":").lower()
        rest = "/".join(win.parts[1:]).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    # Already POSIX-looking
    return s.replace("\\", "/")


def wsl_to_windows_path(path: str) -> Path:
    """Convert /mnt/d/... back to D:\\... when possible."""
    norm = path.replace("\\", "/")
    if norm.startswith("/mnt/") and len(norm) > 6:
        drive = norm[5].upper()
        rest = norm[6:].lstrip("/")
        return Path(f"{drive}:/{rest}")
    return Path(path)


@dataclass(frozen=True)
class WSLProbeResult:
    ok: bool
    detail: str
    payload: dict[str, Any] | None = None


def _wsl_exe() -> str:
    exe = shutil.which("wsl") or shutil.which("wsl.exe")
    if not exe:
        raise WSLBridgeError("wsl.exe not found on PATH")
    return exe


def distro_name() -> str:
    return os.environ.get("VISIONFORGE_WSL_DISTRO", DEFAULT_DISTRO)


def wsl_python() -> str:
    return os.environ.get("VISIONFORGE_WSL_PYTHON", DEFAULT_WSL_PYTHON)


def worker_script() -> str:
    return os.environ.get("VISIONFORGE_WSL_WORKER", DEFAULT_WORKER)


def list_distros() -> list[str]:
    """Return installed WSL distro names (best-effort)."""
    exe = _wsl_exe()
    proc = subprocess.run(
        [exe, "-l", "-q"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    # wsl often emits UTF-16LE on Windows
    raw = proc.stdout
    text = ""
    for enc in ("utf-16-le", "utf-8", "utf-16"):
        try:
            text = raw.decode(enc)
            if text.strip():
                break
        except Exception:
            continue
    if not text.strip() and proc.stderr:
        try:
            text = proc.stderr.decode("utf-8", errors="replace")
        except Exception:
            text = ""
    names: list[str] = []
    for line in text.splitlines():
        name = line.strip().strip("\x00").strip()
        if name:
            names.append(name)
    return names


def distro_installed(name: str | None = None) -> bool:
    target = (name or distro_name()).lower()
    try:
        return any(n.lower() == target for n in list_distros())
    except Exception:
        return False


def run_wsl_json(
    args: list[str],
    *,
    timeout_sec: float = 600.0,
) -> dict[str, Any]:
    """Run worker inside WSL and parse the last JSON object from stdout."""
    exe = _wsl_exe()
    cmd = [
        exe,
        "-d",
        distro_name(),
        "--",
        wsl_python(),
        worker_script(),
        *args,
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    payload: dict[str, Any] | None = None
    # Worker prints one JSON object; take the last non-empty line that parses.
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if payload is None:
        raise WSLBridgeError(
            f"WSL worker returned no JSON (exit={proc.returncode}). "
            f"stderr={stderr[-1500:]!r} stdout={stdout[-1500:]!r}"
        )
    if not payload.get("ok", False):
        raise WSLBridgeError(
            f"WSL worker failed: {payload.get('error', payload)} "
            f"(exit={proc.returncode})"
        )
    return payload


def probe_sam31_runtime(*, timeout_sec: float = 120.0) -> WSLProbeResult:
    """Check whether VisionForge-SAM31 can import torch/triton/sam3."""
    import time

    global _PROBE_CACHE
    now = time.monotonic()
    if _PROBE_CACHE is not None:
        cached_at, cached = _PROBE_CACHE
        if now - cached_at < _PROBE_CACHE_TTL_SEC:
            return cached

    if not distro_installed():
        result = WSLProbeResult(
            ok=False,
            detail=(
                f"WSL distro {distro_name()!r} is not installed. "
                "Import Ubuntu to D:\\WSL\\VisionForge-SAM31 first."
            ),
        )
        _PROBE_CACHE = (now, result)
        return result
    try:
        payload = run_wsl_json(["--mode", "probe"], timeout_sec=timeout_sec)
    except Exception as exc:
        result = WSLProbeResult(ok=False, detail=str(exc))
        _PROBE_CACHE = (now, result)
        return result
    if not payload.get("cuda_available"):
        result = WSLProbeResult(
            ok=False,
            detail=f"WSL probe ok but CUDA unavailable: {payload}",
            payload=payload,
        )
        _PROBE_CACHE = (now, result)
        return result
    if not payload.get("checkpoint_exists"):
        result = WSLProbeResult(
            ok=False,
            detail="WSL runtime ok but SAM 3.1 checkpoint not visible inside WSL",
            payload=payload,
        )
        _PROBE_CACHE = (now, result)
        return result
    result = WSLProbeResult(
        ok=True,
        detail=(
            f"WSL2 {distro_name()} ready: torch={payload.get('torch')} "
            f"triton={payload.get('triton')} gpu={payload.get('gpu_name')}"
        ),
        payload=payload,
    )
    _PROBE_CACHE = (now, result)
    return result
