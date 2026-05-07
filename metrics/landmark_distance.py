"""Shared landmark distance helpers for metric scripts."""

from __future__ import annotations

import numpy as np


LIP_68 = np.arange(48, 68)
FACE_68 = np.arange(68)


def landmark_distance(gt: np.ndarray, generated: np.ndarray, indices: np.ndarray | None = None) -> float:
    gt = np.asarray(gt, dtype=np.float32)
    generated = np.asarray(generated, dtype=np.float32)
    frames = min(gt.shape[0], generated.shape[0])
    gt = gt[:frames]
    generated = generated[:frames]
    if indices is not None:
        gt = gt[:, indices]
        generated = generated[:, indices]
    if gt.shape != generated.shape:
        raise ValueError(f"Landmark shape mismatch: {gt.shape} vs {generated.shape}")
    return float(np.linalg.norm(gt - generated, axis=-1).mean())
