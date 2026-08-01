"""Lightweight GPU / CUDA memory metric helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class GpuSnapshot:
    cuda_available: bool
    device_name: str | None
    total_memory_bytes: int | None
    allocated_bytes: int | None
    reserved_bytes: int | None
    max_allocated_bytes: int | None
    max_reserved_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def reset_peak_stats() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def snapshot_gpu(device_index: int = 0) -> GpuSnapshot:
    try:
        import torch
    except ImportError:
        return GpuSnapshot(
            cuda_available=False,
            device_name=None,
            total_memory_bytes=None,
            allocated_bytes=None,
            reserved_bytes=None,
            max_allocated_bytes=None,
            max_reserved_bytes=None,
        )

    if not torch.cuda.is_available():
        return GpuSnapshot(
            cuda_available=False,
            device_name=None,
            total_memory_bytes=None,
            allocated_bytes=None,
            reserved_bytes=None,
            max_allocated_bytes=None,
            max_reserved_bytes=None,
        )

    props = torch.cuda.get_device_properties(device_index)
    return GpuSnapshot(
        cuda_available=True,
        device_name=props.name,
        total_memory_bytes=int(props.total_memory),
        allocated_bytes=int(torch.cuda.memory_allocated(device_index)),
        reserved_bytes=int(torch.cuda.memory_reserved(device_index)),
        max_allocated_bytes=int(torch.cuda.max_memory_allocated(device_index)),
        max_reserved_bytes=int(torch.cuda.max_memory_reserved(device_index)),
    )


def empty_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
