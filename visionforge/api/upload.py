"""Secure streamed upload and ZIP extraction."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

from PIL import Image

from visionforge.api.config import ApiSettings
from visionforge.api.errors import ApiError
from visionforge.api.schemas import sanitize_filename

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_ARCHIVE_EXTS = {".zip"}


async def stream_upload_to_file(
    upload,
    dest: Path,
    *,
    max_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with dest.open("wb") as fh:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ApiError(
                    "UPLOAD_TOO_LARGE",
                    f"Upload exceeds maximum of {max_bytes} bytes",
                    status_code=413,
                )
            fh.write(chunk)
    if total <= 0:
        raise ApiError("EMPTY_UPLOAD", "Uploaded file is empty", status_code=400)
    return total


def detect_input_kind(path: Path, original_name: str) -> str:
    ext = Path(sanitize_filename(original_name)).suffix.lower()
    if not ext:
        ext = path.suffix.lower()
    # sniff ZIP magic
    with path.open("rb") as fh:
        magic = fh.read(12)
    if magic.startswith(b"PK\x03\x04") or magic.startswith(b"PK\x05\x06"):
        return "zip"
    if ext in _ARCHIVE_EXTS:
        return "zip"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _IMAGE_EXTS:
        raise ApiError(
            "UNSUPPORTED_UPLOAD",
            "Single images are not accepted; upload a ZIP of ordered frames or a video",
            status_code=400,
        )
    if ext in _VIDEO_EXTS or magic[4:8] in {b"ftyp"}:
        return "video"
    raise ApiError(
        "UNSUPPORTED_UPLOAD",
        "Unsupported upload type; provide a video or ZIP of images",
        status_code=400,
    )


def _is_unsafe_zip_name(name: str) -> bool:
    norm = name.replace("\\", "/")
    if not norm or norm.endswith("/"):
        return False  # directory entry handled separately
    if norm.startswith("/") or re_abs_drive(norm):
        return True
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return True
    return False


def re_abs_drive(name: str) -> bool:
    return len(name) >= 2 and name[1] == ":" and name[0].isalpha()


def extract_zip_frames(
    zip_path: Path,
    dest_dir: Path,
    settings: ApiSettings,
) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    uncompressed = 0
    try:
        zf = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile as exc:
        raise ApiError("CORRUPT_ZIP", "Uploaded ZIP is corrupt", status_code=400) from exc

    with zf:
        infos = zf.infolist()
        if len(infos) > settings.max_zip_entries:
            raise ApiError(
                "ZIP_TOO_MANY_ENTRIES",
                f"ZIP exceeds maximum of {settings.max_zip_entries} entries",
                status_code=400,
            )
        file_infos = [i for i in infos if not i.is_dir()]
        if not file_infos:
            raise ApiError("EMPTY_ZIP", "ZIP contains no files", status_code=400)

        for info in file_infos:
            name = info.filename
            if _is_unsafe_zip_name(name):
                raise ApiError(
                    "ZIP_PATH_TRAVERSAL",
                    "ZIP contains unsafe path entries",
                    status_code=400,
                )
            # Symlink / special attributes
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ApiError(
                    "ZIP_SYMLINK_REJECTED",
                    "ZIP symlinks are not allowed",
                    status_code=400,
                )
            if info.compress_type not in (
                zipfile.ZIP_STORED,
                zipfile.ZIP_DEFLATED,
                zipfile.ZIP_BZIP2,
                zipfile.ZIP_LZMA,
            ):
                raise ApiError(
                    "ZIP_UNSUPPORTED_COMPRESSION",
                    "ZIP uses unsupported compression",
                    status_code=400,
                )
            uncompressed += int(info.file_size)
            if uncompressed > settings.max_zip_uncompressed_bytes:
                raise ApiError(
                    "ZIP_BOMB",
                    "ZIP uncompressed size exceeds limit",
                    status_code=400,
                )
            base = sanitize_filename(Path(name).name)
            ext = Path(base).suffix.lower()
            if ext in _ARCHIVE_EXTS:
                raise ApiError(
                    "NESTED_ARCHIVE",
                    "Nested archives are not allowed",
                    status_code=400,
                )
            if ext not in _IMAGE_EXTS:
                raise ApiError(
                    "UNSUPPORTED_ZIP_ENTRY",
                    f"ZIP entry is not a supported image: {base}",
                    status_code=400,
                )
            # ratio bomb check for single entry
            if info.compress_size > 0 and info.file_size / max(1, info.compress_size) > 100:
                if info.file_size > 50 * 1024 * 1024:
                    raise ApiError(
                        "ZIP_BOMB",
                        "Suspicious ZIP compression ratio",
                        status_code=400,
                    )

            out = dest_dir / base
            if out.exists():
                raise ApiError(
                    "DUPLICATE_FILENAME",
                    f"Duplicate unsafe filename in ZIP: {base}",
                    status_code=400,
                )
            with zf.open(info, "r") as src, out.open("wb") as dst:
                remaining = int(info.file_size)
                while remaining > 0:
                    chunk = src.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    dst.write(chunk)
                    remaining -= len(chunk)
            # verify image
            try:
                with Image.open(out) as im:
                    im.load()
            except Exception as exc:
                raise ApiError(
                    "UNREADABLE_IMAGE",
                    f"Unreadable image in ZIP: {base}",
                    status_code=400,
                ) from exc
            frames.append(out)

    if len(frames) > settings.max_image_frames:
        raise ApiError(
            "TOO_MANY_FRAMES",
            f"ZIP exceeds maximum of {settings.max_image_frames} frames",
            status_code=400,
        )
    if not frames:
        raise ApiError("EMPTY_ZIP", "ZIP contained no usable frames", status_code=400)

    # Stable ordered names
    frames_sorted = sorted(frames, key=lambda p: p.name.lower())
    ordered_dir = dest_dir / "ordered"
    ordered_dir.mkdir(parents=True, exist_ok=True)
    ordered: list[Path] = []
    for i, src in enumerate(frames_sorted):
        ext = src.suffix.lower() or ".jpg"
        if ext == ".jpeg":
            ext = ".jpg"
        dest = ordered_dir / f"{i:05d}{ext}"
        dest.write_bytes(src.read_bytes())
        ordered.append(dest)
    return ordered


def validate_box_against_first_frame(
    box: list[float],
    first_frame: Path,
) -> tuple[int, int]:
    with Image.open(first_frame) as im:
        w, h = im.size
    x0, y0, x1, y1 = box
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        raise ApiError(
            "BOX_OUTSIDE_FRAME",
            "Bounding box is completely outside the first frame",
            status_code=400,
        )
    return int(w), int(h)


def sniff_video_or_reject(path: Path) -> None:
    # Light validation: non-empty and recognizable container magic when possible.
    size = path.stat().st_size
    if size < 32:
        raise ApiError("CORRUPT_VIDEO", "Video file is too small", status_code=400)
    with path.open("rb") as fh:
        head = fh.read(32)
    # MP4/MOV often have ftyp at offset 4; AVI has RIFF....AVI
    if head[4:8] == b"ftyp" or head[:4] == b"RIFF" or head[:4] == b"\x1aE\xdf\xa3":
        return
    # Allow extension-based acceptance for other containers; pipeline will fail clearly.
    if path.suffix.lower() in _VIDEO_EXTS:
        return
    raise ApiError("CORRUPT_VIDEO", "Unrecognized video container", status_code=400)


# silence unused import warning for io if any
_ = io
