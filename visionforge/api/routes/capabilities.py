"""Capability reporting without full inference."""
from __future__ import annotations

import time

from fastapi import APIRouter, Request

from visionforge.model_registry import LocalModelRegistry, ModelId
from visionforge.tracking import CapabilityStatus, select_tracker_backend
from visionforge.wsl import distro_name

router = APIRouter(prefix="/v1", tags=["capabilities"])

_CACHE: dict[str, tuple[float, dict]] = {}


@router.get("/capabilities")
def capabilities(request: Request) -> dict:
    settings = request.app.state.settings
    now = time.monotonic()
    cached = _CACHE.get("caps")
    if cached and now - cached[0] < settings.capability_cache_sec:
        return cached[1]

    reg = LocalModelRegistry()
    models = {}
    for mid in (ModelId.EDGETAM, ModelId.DINOV3_VITS16, ModelId.MOBILECLIP2_S0, ModelId.SAM31):
        try:
            pkg = reg.validate(mid)
            models[mid.value] = {
                "status": "AVAILABLE",
                "checkpoint": pkg.primary_checkpoint.name,
            }
        except Exception as exc:
            models[mid.value] = {
                "status": "UNAVAILABLE",
                "detail": str(exc)[:200],
            }

    edge = select_tracker_backend("edgetam").capability()
    sam = select_tracker_backend("sam31").capability()

    cuda = {"available": False, "name": None}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        cuda = {"available": False, "detail": str(exc)[:200]}

    from visionforge.detection import get_default_detector

    det = get_default_detector().capabilities()
    payload = {
        "trackers": {
            "edgetam": {
                "status": edge.status.value,
                "runtime": "native_windows",
                "detail": edge.detail,
            },
            "sam31": {
                "status": sam.status.value,
                "runtime": "wsl2",
                "wsl_distro": distro_name(),
                "detail": sam.detail,
                # Explicit: never claim native Windows availability for SAM 3.1
                "available_native_windows": False,
            },
        },
        "detector": {
            "status": det.status,
            "name": det.name,
            "detail": det.detail,
            "supports_text_prompt": det.supports_text_prompt,
            "supports_class_agnostic": det.supports_class_agnostic,
        },
        "models": models,
        "cuda": cuda,
        "wsl2": {
            "distro": distro_name(),
            "sam31_status": sam.status.value,
            "required_for_sam31": True,
        },
        "notes": [
            "Capability checks do not run full model inference.",
            "SAM 3.1 reports AVAILABLE_WSL2 when ready; never AVAILABLE_NATIVE_WINDOWS.",
            "Automatic candidates use OWL-ViT when weights are installed.",
        ],
    }
    # Sanity: if somehow AVAILABLE without WSL, coerce
    if (
        sam.status == CapabilityStatus.AVAILABLE
        and sam.status != CapabilityStatus.AVAILABLE_WSL2
    ):
        payload["trackers"]["sam31"]["status"] = CapabilityStatus.AVAILABLE_WSL2.value

    _CACHE["caps"] = (now, payload)
    return payload
