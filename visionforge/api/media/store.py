"""Local media probe session store (opaque IDs, no path exposure)."""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from visionforge.api.errors import ApiError
from visionforge.api.schemas import sanitize_filename


class MediaStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, preview_id: str) -> Path:
        if not preview_id or not all(c.isalnum() or c == "-" for c in preview_id):
            raise ApiError("INVALID_PREVIEW_ID", "Illegal preview id", status_code=400)
        d = (self.root / preview_id).resolve()
        if not str(d).startswith(str(self.root.resolve())):
            raise ApiError("INVALID_PREVIEW_ID", "Illegal preview id", status_code=400)
        return d

    def create_from_upload(self, data: bytes, filename: str, *, max_bytes: int) -> dict:
        if len(data) > max_bytes:
            raise ApiError("UPLOAD_TOO_LARGE", "File exceeds size limit", status_code=413)
        if len(data) < 16:
            raise ApiError("UPLOAD_EMPTY", "Empty upload", status_code=400)
        preview_id = str(uuid.uuid4())
        d = self._dir(preview_id)
        d.mkdir(parents=True, exist_ok=False)
        safe = sanitize_filename(filename)
        raw_path = d / f"source{Path(safe).suffix.lower() or '.bin'}"
        raw_path.write_bytes(data)
        meta = probe_media_file(raw_path, original_filename=safe)
        frame_path = d / "first_frame.jpg"
        extract_first_frame(raw_path, frame_path, meta)
        meta.update(
            {
                "preview_id": preview_id,
                "created_at": time.time(),
                "first_frame_url": f"/v1/media/{preview_id}/frame",
            }
        )
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def load_meta(self, preview_id: str) -> dict:
        d = self._dir(preview_id)
        meta_path = d / "meta.json"
        if not meta_path.is_file():
            raise ApiError("PREVIEW_NOT_FOUND", "Unknown preview", status_code=404)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def first_frame_path(self, preview_id: str) -> Path:
        d = self._dir(preview_id)
        p = d / "first_frame.jpg"
        if not p.is_file():
            raise ApiError("FRAME_MISSING", "First frame not available", status_code=404)
        return p

    def source_path(self, preview_id: str) -> Path:
        d = self._dir(preview_id)
        matches = list(d.glob("source.*"))
        if not matches:
            raise ApiError("SOURCE_MISSING", "Source media missing", status_code=404)
        return matches[0]

    def cleanup(self, preview_id: str) -> None:
        d = self._dir(preview_id)
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def probe_media_file(path: Path, *, original_filename: str) -> dict:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        with Image.open(path) as im:
            w, h = im.size
        return {
            "filename": original_filename,
            "kind": "image",
            "mime_hint": f"image/{suffix.lstrip('.')}",
            "width": int(w),
            "height": int(h),
            "duration_sec": None,
            "fps": None,
            "estimated_frames": 1,
            "orientation_normalized": True,
        }

    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise ApiError("UNSUPPORTED_MEDIA", f"Unsupported type: {suffix}", status_code=400)

    try:
        import cv2
    except ImportError as exc:
        raise ApiError(
            "OPENCV_MISSING",
            "OpenCV required for video probe",
            status_code=500,
        ) from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ApiError("VIDEO_UNREADABLE", "Unable to open video", status_code=400)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Read one frame to confirm decode + actual dims
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise ApiError("VIDEO_NO_FRAMES", "No decodable frames", status_code=400)
        h, w = bgr.shape[:2]
        width, height = int(w), int(h)
        duration = (frame_count / fps) if fps > 1e-3 and frame_count > 0 else None
        # Orientation: OpenCV typically delivers upright pixels after decode;
        # record rotation metadata flag as normalized for this pipeline.
        return {
            "filename": original_filename,
            "kind": "video",
            "mime_hint": "video/mp4" if suffix == ".mp4" else f"video/{suffix.lstrip('.')}",
            "width": width,
            "height": height,
            "duration_sec": round(duration, 3) if duration is not None else None,
            "fps": round(fps, 3) if fps > 1e-3 else None,
            "estimated_frames": frame_count if frame_count > 0 else None,
            "orientation_normalized": True,
        }
    finally:
        cap.release()


def extract_first_frame(source: Path, dest: Path, meta: dict) -> None:
    if meta.get("kind") == "image":
        Image.open(source).convert("RGB").save(dest, quality=95)
        return
    import cv2

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ApiError("VIDEO_UNREADABLE", "Unable to open video", status_code=400)
    try:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise ApiError("VIDEO_NO_FRAMES", "No decodable frames", status_code=400)
        rgb = bgr[:, :, ::-1]
        Image.fromarray(rgb).save(dest, quality=95)
        # Align meta dims to actual decoded first frame
        meta["width"] = int(rgb.shape[1])
        meta["height"] = int(rgb.shape[0])
    finally:
        cap.release()


def extract_frame_at_time(source: Path, dest: Path, time_sec: float) -> None:
    """Decode a frame near time_sec (seconds) into dest JPEG."""
    import cv2

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ApiError("VIDEO_UNREADABLE", "Unable to open video", status_code=400)
    try:
        t = max(0.0, float(time_sec))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = cap.read()
        if not ok or bgr is None:
            raise ApiError("VIDEO_NO_FRAMES", "No decodable frames", status_code=400)
        rgb = bgr[:, :, ::-1]
        Image.fromarray(rgb).save(dest, quality=95)
    finally:
        cap.release()


def render_box_mask_preview(
    frame_path: Path,
    box: list[float],
    *,
    out_mask: Path,
    out_overlay: Path,
) -> dict:
    """Generate a confirmation mask by running a tight box fill + edge refine.

    This is NOT a substitute for tracker confirmation. For product confirmation
    the caller should prefer tracker_mask_preview when GPU is available.
    """
    rgb = np.asarray(Image.open(frame_path).convert("RGB"))
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ApiError("INVALID_BOX", "Box is empty after clamping", status_code=400)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    # Optional OpenCV grabCut refine for better preview (still selection aid)
    try:
        import cv2

        gc_mask = np.zeros((h, w), np.uint8)
        rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(rgb[:, :, ::-1], gc_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype(np.uint8)
        if mask.sum() < 32:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            method = "box_fill"
        else:
            method = "opencv_grabcut"
    except Exception:
        method = "box_fill"

    Image.fromarray(mask, mode="L").save(out_mask)
    overlay = rgb.astype(np.float32)
    m = mask.astype(bool)
    overlay[m, 0] = overlay[m, 0] * 0.55 + 101 * 0.45
    overlay[m, 1] = overlay[m, 1] * 0.55 + 221 * 0.45
    overlay[m, 2] = overlay[m, 2] * 0.55 + 244 * 0.45
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_overlay, quality=90)
    area = int((mask > 0).sum())
    box_area = max(1, (x2 - x1) * (y2 - y1))
    return {
        "method": method,
        "note": (
            "Preview segmentation for confirmation. Full tracking uses the selected tracker."
            if method != "box_fill"
            else "Box-fill preview only; confirm carefully before full tracking."
        ),
        "mask_area_px": area,
        "mask_to_box_ratio": round(area / box_area, 4),
        "empty": area < 1,
        "width": w,
        "height": h,
        "box": [x1, y1, x2, y2],
    }
