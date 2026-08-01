"""Force large caches onto D: (C: has insufficient free space).

Call apply_d_drive_caches() before pip / huggingface / torch downloads.
"""
from __future__ import annotations

import os
from pathlib import Path

D_CACHES = Path("D:/caches")


def apply_d_drive_caches() -> dict[str, str]:
    roots = {
        "PIP_CACHE_DIR": D_CACHES / "pip",
        "HF_HOME": D_CACHES / "huggingface",
        "HUGGINGFACE_HUB_CACHE": D_CACHES / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": D_CACHES / "huggingface" / "transformers",
        "TORCH_HOME": D_CACHES / "torch",
        "XDG_CACHE_HOME": D_CACHES / "xdg",
        "TEMP": D_CACHES / "tmp",
        "TMP": D_CACHES / "tmp",
        "TMPDIR": D_CACHES / "tmp",
        "PIP_TMPDIR": D_CACHES / "tmp",
    }
    applied: dict[str, str] = {}
    for key, path in roots.items():
        path.mkdir(parents=True, exist_ok=True)
        value = str(path)
        os.environ[key] = value
        applied[key] = value
    return applied
