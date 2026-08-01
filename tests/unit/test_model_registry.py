"""Unit tests for local model registry (no network, no real multi-GB loads)."""
from __future__ import annotations

from pathlib import Path

import pytest

from visionforge.model_registry import (
    LocalModelRegistry,
    ModelId,
    ModelRegistryError,
    project_root_from,
)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _make_fake_tree(root: Path) -> None:
    (root / "pyproject.toml").write_text('[project]\nname="visionforge"\n', encoding="utf-8")
    (root / "visionforge").mkdir(parents=True, exist_ok=True)
    (root / "visionforge" / "__init__.py").write_text("", encoding="utf-8")

    _write(root / "models" / "sam31" / "sam3.1_multiplex.pt", b"PK" + b"\0" * 1_000_001)
    # Bypass min size for SAM in tests by using a custom registry below for small files.
    _write(root / "models" / "edgetam" / "edgetam.pt", b"PK" + b"\0" * 10_000_001)
    _write(
        root / "models" / "dinov3-vits16" / "model.safetensors",
        b'{"__metadata__":{}}' + b"\0" * 10_000_001,
    )
    _write(root / "models" / "dinov3-vits16" / "config.json", b'{"model_type":"dinov3_vit"}')
    _write(
        root / "models" / "dinov3-vits16" / "preprocessor_config.json",
        b'{"image_processor_type":"x"}',
    )
    _write(root / "models" / "mobileclip2-s0" / "mobileclip2_s0.pt", b"PK" + b"\0" * 10_000_001)
    _write(root / "models" / "mobileclip2-s0" / "config.json", b'{"embed_dim":512}')


def test_project_root_resolution(tmp_path: Path) -> None:
    _make_fake_tree(tmp_path)
    assert project_root_from(tmp_path / "visionforge") == tmp_path.resolve()


def test_path_resolution_relative_to_project(tmp_path: Path) -> None:
    _make_fake_tree(tmp_path)
    # Shrink SAM min size for unit speed via monkeypatched specs copy
    from visionforge import model_registry as mr

    specs = dict(mr.DEFAULT_SPECS)
    sam = specs[ModelId.SAM31]
    specs[ModelId.SAM31] = mr.ModelPackageSpec(
        model_id=sam.model_id,
        env_dir_var=sam.env_dir_var,
        default_relative_dir=sam.default_relative_dir,
        primary_weight_name=sam.primary_weight_name,
        required_files=(
            mr.ModelFileSpec("sam3.1_multiplex.pt", (".pt",), min_bytes=1_000_000, role="weight"),
        ),
    )
    reg = LocalModelRegistry(project_root=tmp_path, specs=specs)
    pkg = reg.validate(ModelId.EDGETAM)
    assert pkg.primary_checkpoint == (tmp_path / "models" / "edgetam" / "edgetam.pt").resolve()
    assert pkg.package_dir.is_relative_to(tmp_path.resolve())


def test_env_override(tmp_path: Path) -> None:
    _make_fake_tree(tmp_path)
    alt = tmp_path / "alt_edgetam"
    _write(alt / "edgetam.pt", b"PK" + b"\0" * 10_000_001)
    reg = LocalModelRegistry(
        project_root=tmp_path,
        environ={"VISIONFORGE_EDGETAM_DIR": str(alt)},
    )
    pkg = reg.validate(ModelId.EDGETAM)
    assert pkg.package_dir == alt.resolve()


def test_missing_file(tmp_path: Path) -> None:
    _make_fake_tree(tmp_path)
    (tmp_path / "models" / "edgetam" / "edgetam.pt").unlink()
    reg = LocalModelRegistry(project_root=tmp_path)
    with pytest.raises(ModelRegistryError, match="Missing required model file"):
        reg.validate(ModelId.EDGETAM)


def test_invalid_extension(tmp_path: Path) -> None:
    _make_fake_tree(tmp_path)
    bad = tmp_path / "models" / "edgetam" / "edgetam.pt"
    bad.unlink()
    _write(tmp_path / "models" / "edgetam" / "edgetam.bin", b"PK" + b"\0" * 10_000_001)
    # Still looking for .pt path
    reg = LocalModelRegistry(project_root=tmp_path)
    with pytest.raises(ModelRegistryError, match="Missing required model file"):
        reg.validate(ModelId.EDGETAM)


def test_incomplete_download(tmp_path: Path) -> None:
    _make_fake_tree(tmp_path)
    _write(tmp_path / "models" / "edgetam" / "edgetam.pt.crdownload", b"partial")
    reg = LocalModelRegistry(project_root=tmp_path)
    with pytest.raises(ModelRegistryError, match="Incomplete download"):
        reg.validate(ModelId.EDGETAM)


def test_html_error_page_rejected(tmp_path: Path) -> None:
    _make_fake_tree(tmp_path)
    path = tmp_path / "models" / "edgetam" / "edgetam.pt"
    path.write_bytes(b"<!DOCTYPE html><html>error</html>" + b" " * 10_000_001)
    reg = LocalModelRegistry(project_root=tmp_path)
    with pytest.raises(ModelRegistryError, match="HTML"):
        reg.validate(ModelId.EDGETAM)


def test_models_dir_excluded_by_gitignore() -> None:
    root = project_root_from()
    gi = (root / ".gitignore").read_text(encoding="utf-8")
    assert "models/" in gi
    assert ".venvs/" in gi
    assert "*.pt" in gi
    assert "*.safetensors" in gi
