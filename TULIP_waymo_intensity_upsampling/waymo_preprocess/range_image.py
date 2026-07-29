"""Pure NumPy operations for native Waymo TOP LiDAR range images."""

from __future__ import annotations

import numpy as np


RING_IDS_32 = np.arange(0, 64, 2, dtype=np.int32)


def extract_range_intensity(
    native_range_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    native = np.asarray(native_range_image, dtype=np.float32)
    if native.ndim != 3 or native.shape[0] != 64 or native.shape[2] < 2:
        raise ValueError(
            "Expected TOP range image [64,W,C>=2], "
            f"got {list(native.shape)}"
        )
    range_64 = native[..., 0]
    intensity_64 = native[..., 1]
    valid_mask_64 = range_64 > 0
    return range_64, intensity_64, valid_mask_64


def select_even_rings(
    range_64: np.ndarray,
    intensity_64: np.ndarray,
    valid_mask_64: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shapes = {
        tuple(np.asarray(range_64).shape),
        tuple(np.asarray(intensity_64).shape),
        tuple(np.asarray(valid_mask_64).shape),
    }
    if len(shapes) != 1 or next(iter(shapes))[0] != 64:
        raise ValueError(
            "range, intensity and mask must share shape [64,W], "
            f"got {sorted(shapes)}"
        )
    return (
        np.asarray(range_64)[RING_IDS_32],
        np.asarray(intensity_64)[RING_IDS_32],
        np.asarray(valid_mask_64, dtype=bool)[RING_IDS_32],
        RING_IDS_32.copy(),
    )
