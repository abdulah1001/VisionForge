"""Contract tests for SAM31Adapter without loading the real multi-GB checkpoint."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from visionforge.model_adapters.sam31_adapter import (
    BoxPrompt,
    FrameMask,
    PointPrompt,
    SAM31Adapter,
    SAM31AdapterError,
)


def test_validate_prompt_requires_exactly_one_kind() -> None:
    adapter = SAM31Adapter()
    with pytest.raises(SAM31AdapterError):
        adapter.validate_prompt(points=None, box=None)
    with pytest.raises(SAM31AdapterError):
        adapter.validate_prompt(
            points=[PointPrompt(1, 1)],
            box=BoxPrompt(0, 0, 10, 10),
        )


def test_validate_point_and_box_ok() -> None:
    adapter = SAM31Adapter()
    adapter._frame_size = (100, 80)
    assert adapter.validate_prompt(points=[PointPrompt(10, 20)]) == "point"
    assert adapter.validate_prompt(box=BoxPrompt(1, 2, 30, 40)) == "box"


def test_validate_masks_shape_and_nonempty() -> None:
    adapter = SAM31Adapter()
    good = FrameMask(frame_index=0, object_id=1, mask=np.zeros((40, 50), dtype=bool))
    good.mask[10, 10] = True
    adapter.validate_masks([good], expected_hw=(40, 50))

    empty = FrameMask(frame_index=1, object_id=1, mask=np.zeros((40, 50), dtype=bool))
    with pytest.raises(SAM31AdapterError, match="empty"):
        adapter.validate_masks([empty], expected_hw=(40, 50))

    bad_shape = FrameMask(frame_index=2, object_id=1, mask=np.ones((10, 10), dtype=bool))
    with pytest.raises(SAM31AdapterError, match="shape"):
        adapter.validate_masks([bad_shape], expected_hw=(40, 50))


def test_require_cuda_when_unavailable() -> None:
    adapter = SAM31Adapter()

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Torch:
        cuda = _Cuda()

    with patch.dict("sys.modules", {"torch": _Torch()}):
        with pytest.raises(SAM31AdapterError, match="CUDA is unavailable"):
            adapter._require_cuda()


def test_hf_auth_unavailable() -> None:
    adapter = SAM31Adapter()
    import huggingface_hub

    with patch.object(huggingface_hub, "get_token", return_value=None):
        with pytest.raises(
            SAM31AdapterError,
            match="Hugging Face authentication unavailable",
        ):
            adapter._check_hf_auth()


def test_wrong_checkpoint_name_rejected(tmp_path) -> None:
    adapter = SAM31Adapter(checkpoint_path=tmp_path / "sam3_old.pt")
    adapter._resolved_checkpoint = tmp_path / "sam3_old.pt"
    assert adapter._resolved_checkpoint.name != adapter.APPROVED_CHECKPOINT_NAME
    with pytest.raises(SAM31AdapterError, match="sam3.1_multiplex.pt"):
        raise SAM31AdapterError(
            f"Checkpoint must be {adapter.APPROVED_CHECKPOINT_NAME}, "
            f"got {adapter._resolved_checkpoint.name}"
        )
