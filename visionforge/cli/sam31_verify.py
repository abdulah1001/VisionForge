"""Validate local SAM 3.1 checkpoint only; optional harmless import check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from visionforge._cache_paths import apply_d_drive_caches
from visionforge.model_registry import LocalModelRegistry, ModelId, ModelRegistryError

apply_d_drive_caches()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("D:/project/artifacts/sam31_verify"),
    )
    args = p.parse_args(argv)

    out: dict = {
        "SAM31_CHECKPOINT": "INVALID",
        "SAM31_RUNTIME": "BLOCKED_NATIVE_WINDOWS",
        "runtime_blocker": None,
        "checkpoint": None,
    }
    try:
        pkg = LocalModelRegistry().validate(ModelId.SAM31)
        out["SAM31_CHECKPOINT"] = "VERIFIED"
        out["checkpoint"] = {
            "path": str(pkg.primary_checkpoint),
            "name": pkg.primary_checkpoint.name,
            "size_bytes": pkg.primary_checkpoint.stat().st_size,
            "package_dir": str(pkg.package_dir),
        }
    except ModelRegistryError as exc:
        out["checkpoint_error"] = str(exc)

    # Harmless import check only — do not install unofficial Triton workarounds.
    try:
        import sam3  # noqa: F401

        out["SAM31_RUNTIME"] = "AVAILABLE"
        out["import_sam3"] = "ok"
    except Exception as exc:
        out["SAM31_RUNTIME"] = "BLOCKED_NATIVE_WINDOWS"
        out["runtime_blocker"] = f"{type(exc).__name__}: {exc}"
        out["import_sam3"] = "failed"

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = args.artifacts_dir / "metrics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if out["SAM31_CHECKPOINT"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
