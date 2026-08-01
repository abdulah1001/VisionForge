"""Unit tests for Windows ↔ WSL path helpers."""
from __future__ import annotations

from pathlib import Path

from visionforge.wsl import windows_to_wsl_path, wsl_to_windows_path


def test_windows_to_wsl_path_drive_d() -> None:
    out = windows_to_wsl_path(r"D:\project\models\sam31\sam3.1_multiplex.pt")
    assert out.startswith("/mnt/d/")
    assert out.endswith("project/models/sam31/sam3.1_multiplex.pt")


def test_wsl_to_windows_roundtrip_prefix() -> None:
    win = wsl_to_windows_path("/mnt/d/project/artifacts/mask_00000.npy")
    assert str(win).replace("\\", "/").lower().startswith("d:/project/artifacts/")
    assert Path(win).name == "mask_00000.npy"
