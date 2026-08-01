"""First-frame tracker mask preview using the selected real tracker."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from visionforge.api.errors import ApiError
from visionforge.pipeline.annotate import overlay_mask_rgba
from visionforge.pipeline.quality import diagnose_mask
from visionforge.tracking import require_available, select_tracker_backend


def tracker_mask_preview(
    *,
    frame_path: Path,
    box_xyxy: list[float],
    tracker: str,
    out_mask: Path,
    out_overlay: Path,
) -> dict:
    """Run EdgeTAM or SAM 3.1 on a one-frame sequence for confirmation.

    Never falls back to GrabCut or another tracker.
    """
    tracker = str(tracker).strip().lower()
    if tracker not in {"edgetam", "sam31"}:
        raise ApiError("INVALID_TRACKER", "tracker must be edgetam or sam31", status_code=400)

    rgb = np.asarray(Image.open(frame_path).convert("RGB"))
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    if not (x2 > x1 and y2 > y1):
        raise ApiError("INVALID_BOX", "Box must have positive area", status_code=400)

    backend = select_tracker_backend(tracker)
    require_available(backend)

    tmp = Path(tempfile.mkdtemp(prefix="vf_mask_preview_"))
    try:
        frame_dir = tmp / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(frame_dir / "00000.jpg", quality=95)
        backend.load()
        try:
            result = backend.track(
                frame_dir,
                box_xyxy=(x1, y1, x2, y2),
                frame_width=w,
                frame_height=h,
                object_id=1,
            )
        finally:
            backend.close()

        if not result.frames:
            raise ApiError("MASK_PREVIEW_FAILED", "Tracker returned no frames", status_code=500)
        fr = result.frames[0]
        mask = np.asarray(fr.mask).astype(bool)
        if mask.shape != (h, w):
            raise ApiError(
                "MASK_SHAPE_MISMATCH",
                f"Mask shape {mask.shape} != frame {(h, w)}",
                status_code=500,
            )

        diag = diagnose_mask(
            mask,
            frame_index=0,
            frame_w=w,
            frame_h=h,
            box_xyxy=(x1, y1, x2, y2),
        )
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_mask)
        overlay = overlay_mask_rgba(rgb, mask, opacity=0.45)
        Image.fromarray(overlay).save(out_overlay, quality=90)

        runtime = "native_windows" if tracker == "edgetam" else "wsl2"
        wsl_distro = "VisionForge-SAM31" if tracker == "sam31" else None
        if diag.empty or not diag.valid:
            return {
                "method": f"tracker:{tracker}",
                "tracker": tracker,
                "runtime": runtime,
                "wsl_distro": wsl_distro,
                "used_real_cuda": bool(result.used_real_cuda),
                "empty": diag.empty,
                "valid": False,
                "mask_area_px": diag.area_px,
                "mask_to_box_ratio": diag.mask_to_box_ratio,
                "reasons": diag.reasons,
                "width": w,
                "height": h,
                "box": [x1, y1, x2, y2],
                "note": "Tracker-produced first-frame mask failed validation.",
            }

        return {
            "method": f"tracker:{tracker}",
            "tracker": tracker,
            "runtime": runtime,
            "wsl_distro": wsl_distro,
            "used_real_cuda": bool(result.used_real_cuda),
            "empty": False,
            "valid": True,
            "mask_area_px": diag.area_px,
            "mask_to_box_ratio": diag.mask_to_box_ratio,
            "reasons": diag.reasons,
            "width": w,
            "height": h,
            "box": [x1, y1, x2, y2],
            "checkpoint_path_name": Path(result.checkpoint_path).name
            if result.checkpoint_path
            else None,
            "note": (
                "Real tracker first-frame mask for confirmation. "
                "GrabCut is not used."
            ),
        }
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(
            "MASK_PREVIEW_FAILED",
            f"Tracker mask preview failed: {exc}"[:300],
            status_code=500,
        ) from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
