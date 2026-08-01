"""Local-only model path registry. Never downloads weights or authenticates."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class ModelId(str, Enum):
    SAM31 = "sam31"
    EDGETAM = "edgetam"
    DINOV3_VITS16 = "dinov3_vits16"
    MOBILECLIP2_S0 = "mobileclip2_s0"


class ModelRegistryError(Exception):
    """Raised when a local model package is missing, incomplete, or invalid."""


_INCOMPLETE_SUFFIXES = (".crdownload", ".tmp", ".part", ".download")
_HTML_SNIFF_PREFIXES = (b"<!DOCTYPE", b"<html", b"<HTML", b"<head", b"<HEAD")


@dataclass(frozen=True)
class ModelFileSpec:
    """A required file relative to a model package directory."""

    relative_path: str
    allowed_suffixes: tuple[str, ...]
    min_bytes: int = 1
    role: str = "weight"


@dataclass(frozen=True)
class ModelPackageSpec:
    model_id: ModelId
    env_dir_var: str
    default_relative_dir: str
    required_files: tuple[ModelFileSpec, ...]
    primary_weight_name: str


@dataclass(frozen=True)
class ValidatedModelFile:
    path: Path
    role: str
    size_bytes: int
    suffix: str


@dataclass(frozen=True)
class ValidatedModelPackage:
    model_id: ModelId
    package_dir: Path
    files: tuple[ValidatedModelFile, ...]
    primary_checkpoint: Path
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def file_map(self) -> dict[str, Path]:
        return {f.path.name: f.path for f in self.files}


DEFAULT_SPECS: Mapping[ModelId, ModelPackageSpec] = {
    ModelId.SAM31: ModelPackageSpec(
        model_id=ModelId.SAM31,
        env_dir_var="VISIONFORGE_SAM31_DIR",
        default_relative_dir="models/sam31",
        primary_weight_name="sam3.1_multiplex.pt",
        required_files=(
            ModelFileSpec(
                "sam3.1_multiplex.pt",
                (".pt",),
                min_bytes=1_000_000_000,
                role="weight",
            ),
        ),
    ),
    ModelId.EDGETAM: ModelPackageSpec(
        model_id=ModelId.EDGETAM,
        env_dir_var="VISIONFORGE_EDGETAM_DIR",
        default_relative_dir="models/edgetam",
        primary_weight_name="edgetam.pt",
        required_files=(
            ModelFileSpec("edgetam.pt", (".pt",), min_bytes=10_000_000, role="weight"),
        ),
    ),
    ModelId.DINOV3_VITS16: ModelPackageSpec(
        model_id=ModelId.DINOV3_VITS16,
        env_dir_var="VISIONFORGE_DINOV3_DIR",
        default_relative_dir="models/dinov3-vits16",
        primary_weight_name="model.safetensors",
        required_files=(
            ModelFileSpec(
                "model.safetensors",
                (".safetensors",),
                min_bytes=10_000_000,
                role="weight",
            ),
            ModelFileSpec("config.json", (".json",), min_bytes=10, role="config"),
            ModelFileSpec(
                "preprocessor_config.json",
                (".json",),
                min_bytes=10,
                role="preprocessor",
            ),
        ),
    ),
    ModelId.MOBILECLIP2_S0: ModelPackageSpec(
        model_id=ModelId.MOBILECLIP2_S0,
        env_dir_var="VISIONFORGE_MOBILECLIP2_DIR",
        default_relative_dir="models/mobileclip2-s0",
        primary_weight_name="mobileclip2_s0.pt",
        required_files=(
            ModelFileSpec(
                "mobileclip2_s0.pt",
                (".pt",),
                min_bytes=10_000_000,
                role="weight",
            ),
            ModelFileSpec("config.json", (".json",), min_bytes=10, role="config"),
        ),
    ),
}


def project_root_from(start: Path | None = None) -> Path:
    """Resolve VisionForge project root containing pyproject.toml."""
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "visionforge").is_dir():
            return candidate
    # Fallback: package parent parents[1] when imported from installed editable.
    pkg = Path(__file__).resolve().parents[1]
    if (pkg / "pyproject.toml").is_file():
        return pkg
    raise ModelRegistryError(f"Unable to locate VisionForge project root from {cur}")


def _looks_like_html(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            head = fh.read(256).lstrip()
    except OSError as exc:
        raise ModelRegistryError(f"Unable to read file for validation: {path}") from exc
    return any(head.startswith(prefix) for prefix in _HTML_SNIFF_PREFIXES)


def _validate_one_file(package_dir: Path, spec: ModelFileSpec) -> ValidatedModelFile:
    path = (package_dir / spec.relative_path).resolve()
    name_lower = path.name.lower()
    if any(name_lower.endswith(sfx) for sfx in _INCOMPLETE_SUFFIXES):
        raise ModelRegistryError(f"Incomplete download artifact rejected: {path}")
    if not path.is_file():
        raise ModelRegistryError(f"Missing required model file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ModelRegistryError(f"Zero-byte model file: {path}")
    if size < spec.min_bytes:
        raise ModelRegistryError(
            f"File too small for role={spec.role}: {path} size={size} min={spec.min_bytes}"
        )
    suffix = path.suffix.lower()
    if suffix not in spec.allowed_suffixes:
        raise ModelRegistryError(
            f"Invalid extension for {path}: got {suffix}, allowed {spec.allowed_suffixes}"
        )
    if spec.role == "weight" and _looks_like_html(path):
        raise ModelRegistryError(f"Weight file looks like an HTML error page: {path}")
    return ValidatedModelFile(path=path, role=spec.role, size_bytes=size, suffix=suffix)


class LocalModelRegistry:
    """Resolve and validate local model packages relative to the project root."""

    def __init__(
        self,
        project_root: Path | None = None,
        specs: Mapping[ModelId, ModelPackageSpec] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.project_root = (project_root or project_root_from()).resolve()
        self.specs = dict(specs or DEFAULT_SPECS)
        self.environ = dict(environ) if environ is not None else dict(os.environ)

    def resolve_package_dir(self, model_id: ModelId | str) -> Path:
        mid = ModelId(model_id)
        spec = self.specs[mid]
        override = self.environ.get(spec.env_dir_var)
        if override:
            path = Path(override).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            return path.resolve()
        return (self.project_root / spec.default_relative_dir).resolve()

    def validate(self, model_id: ModelId | str) -> ValidatedModelPackage:
        mid = ModelId(model_id)
        spec = self.specs[mid]
        package_dir = self.resolve_package_dir(mid)
        if not package_dir.is_dir():
            raise ModelRegistryError(f"Model package directory missing: {package_dir}")

        warnings: list[str] = []
        # Reject incomplete siblings in the package directory.
        for child in package_dir.iterdir():
            if child.is_file() and any(
                child.name.lower().endswith(sfx) for sfx in _INCOMPLETE_SUFFIXES
            ):
                raise ModelRegistryError(f"Incomplete download present in package: {child}")

        validated: list[ValidatedModelFile] = []
        for file_spec in spec.required_files:
            validated.append(_validate_one_file(package_dir, file_spec))

        primary = package_dir / spec.primary_weight_name
        if not primary.is_file():
            # Allow alternate primary if listed required weight exists under another name.
            weight_files = [f for f in validated if f.role == "weight"]
            if not weight_files:
                raise ModelRegistryError(f"No weight file validated for {mid.value}")
            primary = weight_files[0].path
            warnings.append(
                "Primary weight name differs from expected "
                f"{spec.primary_weight_name}: {primary.name}"
            )
        else:
            primary = primary.resolve()

        return ValidatedModelPackage(
            model_id=mid,
            package_dir=package_dir,
            files=tuple(validated),
            primary_checkpoint=primary,
            warnings=tuple(warnings),
        )

    def validate_all(
        self, model_ids: Iterable[ModelId | str] | None = None
    ) -> dict[ModelId, ValidatedModelPackage]:
        ids: Sequence[ModelId | str] = (
            list(model_ids) if model_ids is not None else list(self.specs.keys())
        )
        return {ModelId(mid): self.validate(mid) for mid in ids}
