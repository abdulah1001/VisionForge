"""Artifact listing and safe ZIP download."""
from __future__ import annotations

import hashlib
import mimetypes
import zipfile
from pathlib import Path

from visionforge.api.errors import ApiError

ALLOWED_SUFFIXES = {
    ".json",
    ".npy",
    ".png",
    ".jpg",
    ".jpeg",
    ".log",
    ".txt",
    ".jsonl",
    ".mp4",
}

ALLOWED_NAMES = {
    "manifest.json",
    "run_metadata.json",
    "identity_similarities.json",
    "mobileclip2_similarities.json",
    "quality_report.json",
    "recovery_decisions.json",
    "annotated.mp4",
    "cleaned.mp4",
    "cleaned_video_only.mp4",
    "dinov3_embeddings.npy",
    "mobileclip2_image_embeddings.npy",
    "mobileclip2_text_embeddings.npy",
    "result.json",
    "state.json",
    "request.json",
    "stdout.log",
    "stderr.log",
    "progress.jsonl",
}


def make_artifact_id(group: str, rel: str) -> str:
    digest = hashlib.sha256(f"{group}:{rel}".encode("utf-8")).hexdigest()
    return digest[:24]


def resolve_pipeline_dir(state: dict) -> Path | None:
    raw = state.get("pipeline_run_dir")
    if not raw:
        return None
    p = Path(str(raw))
    if not p.is_dir():
        return None
    return p.resolve()


def iter_allowed_files(job_dir: Path, state: dict) -> list[tuple[str, Path, str]]:
    """Return list of (group, absolute_path, rel_posix)."""
    out: list[tuple[str, Path, str]] = []
    pipeline = resolve_pipeline_dir(state)
    roots: list[tuple[str, Path]] = [("job", job_dir.resolve())]
    if pipeline is not None:
        roots.append(("pipeline", pipeline))

    for label, root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if not _is_allowed_file(path, root):
                continue
            rel = path.relative_to(root).as_posix()
            out.append((label, path.resolve(), rel))
    return out


def list_allowed_artifacts(job_dir: Path, state: dict) -> list[dict]:
    items: list[dict] = []
    for label, path, rel in iter_allowed_files(job_dir, state):
        aid = make_artifact_id(label, rel)
        previewable = path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".json",
            ".txt",
            ".log",
            ".jsonl",
            ".mp4",
        }
        items.append(
            {
                "id": aid,
                "group": label,
                "path": rel,
                "size_bytes": path.stat().st_size,
                "name": path.name,
                "preview_url": (
                    f"/v1/jobs/{job_dir.name}/artifacts/{aid}" if previewable else None
                ),
            }
        )
    return items


def resolve_artifact_by_id(job_dir: Path, state: dict, artifact_id: str) -> Path:
    if not artifact_id or not artifact_id.isalnum() or len(artifact_id) > 64:
        raise ApiError("INVALID_ARTIFACT_ID", "Illegal artifact id", status_code=400)
    for label, path, rel in iter_allowed_files(job_dir, state):
        if make_artifact_id(label, rel) == artifact_id:
            if path.is_symlink():
                raise ApiError("FORBIDDEN_ARTIFACT", "Symlink rejected", status_code=403)
            return path
    raise ApiError("NOT_FOUND", "Artifact not found", status_code=404)


def guess_media_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _is_allowed_file(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
        root_res = root.resolve()
        resolved.relative_to(root_res)
    except Exception:
        return False
    if resolved.is_symlink():
        return False
    name = resolved.name.lower()
    suffix = resolved.suffix.lower()
    if name in {n.lower() for n in ALLOWED_NAMES}:
        return True
    if suffix not in ALLOWED_SUFFIXES:
        return False
    parent = resolved.parent.name.lower()
    if parent in {"masks", "overlays", "crops", "logs", "frames_proc", "ordered"}:
        return True
    return name.endswith(".json") or name.endswith(".npy") or name.endswith(".log")


def build_artifacts_zip(job_dir: Path, state: dict, dest_zip: Path) -> Path:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    pipeline = resolve_pipeline_dir(state)
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ("result.json", "state.json", "request.json"):
            p = job_dir / name
            if p.is_file() and _is_allowed_file(p, job_dir):
                zf.write(p, arcname=f"job/{name}")
        logs = job_dir / "logs"
        if logs.is_dir():
            for p in logs.iterdir():
                if p.is_file() and _is_allowed_file(p, job_dir):
                    zf.write(p, arcname=f"job/logs/{p.name}")
        if pipeline is not None:
            for p in pipeline.rglob("*"):
                if not p.is_file():
                    continue
                if not _is_allowed_file(p, pipeline):
                    continue
                rel = p.relative_to(pipeline).as_posix()
                if ".." in rel.split("/"):
                    continue
                zf.write(p, arcname=f"pipeline/{rel}")
    return dest_zip


def safe_job_file(job_dir: Path, rel: str) -> Path:
    if not rel or rel.startswith("/") or "\\" in rel or ".." in rel.split("/"):
        raise ApiError("PATH_TRAVERSAL", "Illegal artifact path", status_code=400)
    target = (job_dir / rel).resolve()
    root = job_dir.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ApiError("PATH_TRAVERSAL", "Illegal artifact path", status_code=400) from exc
    if target.is_symlink() or not target.is_file():
        raise ApiError("NOT_FOUND", "Artifact not found", status_code=404)
    if not _is_allowed_file(target, root):
        raise ApiError("FORBIDDEN_ARTIFACT", "Artifact is not downloadable", status_code=403)
    return target
