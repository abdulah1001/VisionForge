"""Feature normalization and cosine similarity helpers."""
from __future__ import annotations

import numpy as np


class FeatureError(Exception):
    pass


def l2_normalize(vectors: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise FeatureError(f"Expected 1D or 2D array, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise FeatureError("Non-finite values in feature vectors")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    out = arr / norms
    if not np.isfinite(out).all():
        raise FeatureError("Non-finite values after L2 normalization")
    return out.astype(np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = l2_normalize(a)
    bb = l2_normalize(b)
    return (aa @ bb.T).astype(np.float32)


def pairwise_cosine(vectors: np.ndarray) -> np.ndarray:
    """Return cosine similarity of each row against the first row and consecutive pairs."""
    v = l2_normalize(vectors)
    vs_first = (v @ v[0:1].T).reshape(-1)
    consecutive = np.ones(len(v), dtype=np.float32)
    if len(v) > 1:
        consecutive[1:] = np.sum(v[1:] * v[:-1], axis=1)
    return vs_first.astype(np.float32), consecutive.astype(np.float32)


def summarize_similarities(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise FeatureError("No similarity values to summarize")
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
