"""Media probe / first-frame / candidates / tracker mask-preview routes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from visionforge.api.errors import ApiError
from visionforge.api.media.mask_preview import tracker_mask_preview
from visionforge.api.media.store import MediaStore, extract_frame_at_time, render_box_mask_preview
from visionforge.detection import get_default_detector, get_text_prompt_detector

router = APIRouter(prefix="/v1/media", tags=["media"])


def _store(request: Request) -> MediaStore:
    settings = request.app.state.settings
    root = Path(settings.jobs_root).parent / "media_previews"
    if not hasattr(request.app.state, "media_store"):
        request.app.state.media_store = MediaStore(root)
    return request.app.state.media_store


def _detector_payload() -> dict:
    det = get_default_detector()
    cap = det.capabilities()
    return {
        "status": cap.status,
        "name": cap.name,
        "detail": cap.detail,
        "supports_text_prompt": cap.supports_text_prompt,
        "supports_class_agnostic": cap.supports_class_agnostic,
    }


@router.post("/probe")
async def probe_media(
    request: Request,
    input: UploadFile = File(...),
) -> dict:
    settings = request.app.state.settings
    data = await input.read()
    store = _store(request)
    meta = store.create_from_upload(
        data,
        input.filename or "upload.bin",
        max_bytes=settings.max_upload_bytes,
    )
    return {
        "preview_id": meta["preview_id"],
        "filename": meta["filename"],
        "kind": meta["kind"],
        "mime_hint": meta.get("mime_hint"),
        "width": meta["width"],
        "height": meta["height"],
        "duration_sec": meta.get("duration_sec"),
        "fps": meta.get("fps"),
        "estimated_frames": meta.get("estimated_frames"),
        "orientation_normalized": meta.get("orientation_normalized", True),
        "first_frame_url": meta["first_frame_url"],
        "limits": {
            "max_upload_bytes": settings.max_upload_bytes,
            "max_frames": settings.max_max_frames,
            "max_processing_side": settings.max_processing_side,
        },
        "detector": _detector_payload(),
    }


@router.get("/{preview_id}/frame")
def get_first_frame(preview_id: str, request: Request):
    store = _store(request)
    path = store.first_frame_path(preview_id)
    return FileResponse(path, media_type="image/jpeg", filename="first_frame.jpg")


@router.post("/{preview_id}/candidates")
async def candidates(
    preview_id: str,
    request: Request,
    text_prompt: str | None = Form(default=None),
    time_sec: str | None = Form(default=None),
) -> dict:
    store = _store(request)
    meta = store.load_meta(preview_id)

    frame_path = store.first_frame_path(preview_id)
    t_raw = (time_sec or "").strip()
    if t_raw and meta.get("kind") == "video":
        try:
            t_val = float(t_raw)
        except ValueError:
            t_val = None
        if t_val is not None and t_val > 0.02:
            seek_path = store._dir(preview_id) / "seek_frame.jpg"
            extract_frame_at_time(store.source_path(preview_id), seek_path, t_val)
            frame_path = seek_path

    frame = np.asarray(Image.open(frame_path).convert("RGB"))
    prompt = (text_prompt or "").strip() or None
    det = get_text_prompt_detector() if prompt else get_default_detector()
    cap = det.capabilities()
    if cap.status != "AVAILABLE":
        return {
            "preview_id": preview_id,
            "status": cap.status,
            "candidates": [],
            "detector": _detector_payload(),
            "detail": cap.detail,
        }
    try:
        found = det.detect(frame, text_prompt=prompt)
    except Exception as exc:
        raise ApiError(
            "DETECTION_FAILED",
            f"Detector inference failed: {exc}"[:280],
            status_code=500,
        ) from exc
    finally:
        # Free VRAM so subsequent tracker jobs (esp. WSL SAM) are not starved.
        close = getattr(det, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    return {
        "preview_id": preview_id,
        "status": "AVAILABLE" if found else "NO_RESULTS",
        "frame_width": meta["width"],
        "frame_height": meta["height"],
        "text_prompt": prompt,
        "detector": {
            **_detector_payload(),
            "runtime": "native_windows",
            "mode": "text_prompt" if prompt else "vocabulary",
        },
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "box_xyxy": [float(x) for x in c.box_xyxy],
                "score": c.score,
                "label": c.label,
                "has_mask": c.has_mask,
            }
            for c in found
        ],
        "detail": None if found else "No candidates above threshold for this frame.",
    }


@router.post("/{preview_id}/mask-preview")
async def mask_preview(
    preview_id: str,
    request: Request,
    spec: str = Form(...),
) -> dict:
    store = _store(request)
    meta = store.load_meta(preview_id)
    try:
        payload = json.loads(spec)
    except json.JSONDecodeError as exc:
        raise ApiError("INVALID_SPEC", "spec must be JSON", status_code=400) from exc

    box = payload.get("box")
    if not isinstance(box, list) or len(box) != 4:
        raise ApiError("INVALID_BOX", "box must be [x1,y1,x2,y2]", status_code=400)
    try:
        box_f = [float(x) for x in box]
    except (TypeError, ValueError) as exc:
        raise ApiError("INVALID_BOX", "box values must be numbers", status_code=400) from exc
    if box_f[2] <= box_f[0] or box_f[3] <= box_f[1]:
        raise ApiError("INVALID_BOX", "box must satisfy x1<x2 and y1<y2", status_code=400)

    tracker = str(payload.get("tracker", "edgetam")).strip().lower()
    method = str(payload.get("method", "tracker")).strip().lower()
    d = store._dir(preview_id)
    out_mask = d / "preview_mask.png"
    out_overlay = d / "preview_overlay.jpg"

    if method == "grabcut_diagnostic":
        # Explicit optional diagnostic only — never satisfies tracker confirmation.
        diag = render_box_mask_preview(
            store.first_frame_path(preview_id),
            box_f,
            out_mask=out_mask,
            out_overlay=out_overlay,
        )
        diag["method"] = "grabcut_diagnostic"
        diag["valid_for_confirmation"] = False
        return {
            "preview_id": preview_id,
            "frame_width": meta["width"],
            "frame_height": meta["height"],
            "diagnostics": diag,
            "mask_url": f"/v1/media/{preview_id}/preview-mask",
            "overlay_url": f"/v1/media/{preview_id}/preview-overlay",
            "confirmed_required": True,
            "valid_for_confirmation": False,
            "message": (
                "GrabCut diagnostic preview only. Confirm using a real tracker mask."
            ),
        }

    diag = tracker_mask_preview(
        frame_path=store.first_frame_path(preview_id),
        box_xyxy=box_f,
        tracker=tracker,
        out_mask=out_mask,
        out_overlay=out_overlay,
    )
    if diag.get("empty") or not diag.get("valid"):
        raise ApiError(
            "INVALID_INITIAL_MASK",
            "Tracker initial mask failed validation; redraw or choose another object.",
            status_code=400,
            extra={"diagnostics": diag},
        )
    return {
        "preview_id": preview_id,
        "frame_width": meta["width"],
        "frame_height": meta["height"],
        "diagnostics": {**diag, "valid_for_confirmation": True},
        "mask_url": f"/v1/media/{preview_id}/preview-mask",
        "overlay_url": f"/v1/media/{preview_id}/preview-overlay",
        "confirmed_required": True,
        "valid_for_confirmation": True,
        "message": "This is the object I want to track. Confirm to continue.",
    }


@router.get("/{preview_id}/preview-mask")
def preview_mask(preview_id: str, request: Request):
    store = _store(request)
    path = store._dir(preview_id) / "preview_mask.png"
    if not path.is_file():
        raise ApiError("PREVIEW_MISSING", "Run mask-preview first", status_code=404)
    return FileResponse(path, media_type="image/png")


@router.get("/{preview_id}/preview-overlay")
def preview_overlay(preview_id: str, request: Request):
    store = _store(request)
    path = store._dir(preview_id) / "preview_overlay.jpg"
    if not path.is_file():
        raise ApiError("PREVIEW_MISSING", "Run mask-preview first", status_code=404)
    return FileResponse(path, media_type="image/jpeg")
