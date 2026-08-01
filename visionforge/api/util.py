"""Small API helpers."""
from __future__ import annotations

import shutil
from pathlib import Path


def disk_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        usage = shutil.disk_usage(path)
        return usage.free > 0
    except Exception:
        return False
