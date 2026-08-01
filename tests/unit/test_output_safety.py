"""Unit tests for output directory safety and manifest serialization."""
from __future__ import annotations

from pathlib import Path

import pytest

from visionforge.pipeline.io import PipelineIOError, create_run_directory
from visionforge.pipeline.manifest import write_json


def test_create_run_directory_unique(tmp_path: Path) -> None:
    d1, id1 = create_run_directory(tmp_path, run_id="run-a")
    assert d1.is_dir()
    assert id1 == "run-a"
    with pytest.raises(PipelineIOError, match="already exists"):
        create_run_directory(tmp_path, run_id="run-a")
    d2, _ = create_run_directory(tmp_path, run_id="run-a", allow_existing=True)
    assert d2 == d1


def test_manifest_write(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_json(path, {"ok": True, "n": 1})
    text = path.read_text(encoding="utf-8")
    assert '"ok": true' in text
