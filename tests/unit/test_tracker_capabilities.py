"""Unit tests for tracker capability selection and no silent fallback."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from visionforge.tracking import (
    CapabilityStatus,
    TrackerBackendError,
    TrackerId,
    require_available,
    select_tracker_backend,
)


def test_edgetam_available() -> None:
    backend = select_tracker_backend("edgetam")
    assert backend.tracker_id == TrackerId.EDGETAM
    cap = backend.capability()
    assert cap.status == CapabilityStatus.AVAILABLE


def test_sam31_blocked_when_wsl_missing() -> None:
    backend = select_tracker_backend("sam31")
    assert backend.tracker_id == TrackerId.SAM31
    with patch("visionforge.wsl.bridge.distro_installed", return_value=False):
        # capability imports distro_installed from visionforge.wsl package
        with patch("visionforge.wsl.distro_installed", return_value=False):
            cap = backend.capability()
    assert cap.status == CapabilityStatus.BLOCKED_WSL_MISSING
    with pytest.raises(TrackerBackendError, match="BLOCKED_WSL_MISSING"):
        with patch("visionforge.wsl.distro_installed", return_value=False):
            # re-evaluate inside require via capability
            with patch.object(
                backend,
                "capability",
                return_value=type(cap)(
                    tracker_id=TrackerId.SAM31,
                    status=CapabilityStatus.BLOCKED_WSL_MISSING,
                    detail="missing",
                ),
            ):
                require_available(backend)


def test_sam31_available_wsl2_when_probe_ok() -> None:
    from visionforge.wsl import WSLProbeResult

    backend = select_tracker_backend("sam31")
    probe = WSLProbeResult(ok=True, detail="WSL2 ready mock")
    with (
        patch("visionforge.wsl.distro_installed", return_value=True),
        patch("visionforge.wsl.probe_sam31_runtime", return_value=probe),
    ):
        cap = backend.capability()
        assert cap.status == CapabilityStatus.AVAILABLE_WSL2
        require_available(backend)


def test_sam31_no_edgetam_fallback_on_track() -> None:
    backend = select_tracker_backend("sam31")
    with pytest.raises(TrackerBackendError, match="not loaded|BLOCKED|refusing|fallback|WSL"):
        backend.track(
            Path("."),
            box_xyxy=(0, 0, 1, 1),
            frame_width=8,
            frame_height=8,
        )


def test_unknown_tracker() -> None:
    with pytest.raises(TrackerBackendError, match="Unknown tracker"):
        select_tracker_backend("magic-tracker")
