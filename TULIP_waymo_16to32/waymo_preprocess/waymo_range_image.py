"""Native Waymo 64-to-32-to-16 line transformations."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _validate_native_range_image(range_image: np.ndarray) -> np.ndarray:
    """Validate a native Range Image and cast it to float32 HWC.

    Args:
        range_image: Array with expected shape [64, W, C], C >= 2.

    Returns:
        Float32 array with the same shape.

    Raises:
        ValueError: If dimensionality, height, width, or channels are invalid.
    """
    array = np.asarray(range_image, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected native Range Image [64,W,C], got {array.shape}.")
    if array.shape[0] != 64:
        raise ValueError(f"Expected native Range Image height 64, got {array.shape[0]}.")
    if array.shape[1] <= 0:
        raise ValueError("Native Range Image width must be positive.")
    if array.shape[2] < 2:
        raise ValueError(f"Need range and intensity channels, got C={array.shape[2]}.")
    return array


def extract_range_intensity(
    range_image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract range, intensity, and valid mask from native Range Image.

    Args:
        range_image: Native first-return array [64, W, C]. Channel 0 is range
            and channel 1 is intensity.

    Returns:
        range [64,W] float32, intensity [64,W] float32, and mask [64,W] bool.
        A pixel is valid when range > 0 and both channels are finite. Invalid
        range and intensity values are set to zero.

    Raises:
        ValueError: If the input shape or channel count is invalid.
    """
    native = _validate_native_range_image(range_image)
    raw_range = native[..., 0]
    raw_intensity = native[..., 1]
    valid = (
        (raw_range > 0.0)
        & np.isfinite(raw_range)
        & np.isfinite(raw_intensity)
    )
    return (
        np.where(valid, raw_range, 0.0).astype(np.float32),
        np.where(valid, raw_intensity, 0.0).astype(np.float32),
        valid.astype(bool, copy=False),
    )


def _validate_components(
    range_image: np.ndarray,
    intensity_image: np.ndarray,
    mask: np.ndarray,
    expected_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a range/intensity/mask triplet.

    Args:
        range_image: Range array [expected_height,W].
        intensity_image: Intensity array [expected_height,W].
        mask: Boolean mask [expected_height,W].
        expected_height: Required number of rows.

    Returns:
        Float32 range, float32 intensity, and bool mask.

    Raises:
        ValueError: If shapes or finite-value constraints are invalid.
    """
    range_array = np.asarray(range_image, dtype=np.float32)
    intensity_array = np.asarray(intensity_image, dtype=np.float32)
    mask_array = np.asarray(mask, dtype=bool)
    if range_array.ndim != 2 or range_array.shape[0] != expected_height:
        raise ValueError(
            f"range_image must have shape [{expected_height},W], got {range_array.shape}."
        )
    if range_array.shape[1] <= 0:
        raise ValueError("Range Image width must be positive.")
    if intensity_array.shape != range_array.shape:
        raise ValueError(
            f"intensity shape {intensity_array.shape} != range shape {range_array.shape}."
        )
    if mask_array.shape != range_array.shape:
        raise ValueError(
            f"mask shape {mask_array.shape} != range shape {range_array.shape}."
        )
    if not np.all(np.isfinite(range_array)):
        raise ValueError("range_image contains NaN or Inf.")
    if not np.all(np.isfinite(intensity_array)):
        raise ValueError("intensity_image contains NaN or Inf.")
    return range_array, intensity_array, mask_array


def make_32line_gt(
    range_64: np.ndarray,
    intensity_64: np.ndarray,
    mask_64: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create 32-line GT from native even rows.

    Args:
        range_64: Float32 range [64,W].
        intensity_64: Float32 intensity [64,W].
        mask_64: Bool validity mask [64,W].

    Returns:
        range_32 [32,W] float32, intensity_32 [32,W] float32,
        mask_32 [32,W] bool, and ring_ids_32 [32] int32 equal to
        [0,2,4,...,62].
    """
    range_array, intensity_array, mask_array = _validate_components(
        range_64, intensity_64, mask_64, expected_height=64
    )
    ring_ids_32 = np.arange(0, 64, 2, dtype=np.int32)
    return (
        range_array[ring_ids_32].copy(),
        intensity_array[ring_ids_32].copy(),
        mask_array[ring_ids_32].copy(),
        ring_ids_32,
    )


def make_16line_input(
    range_32: np.ndarray,
    intensity_32: np.ndarray,
    mask_32: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create 16-line input from even rows of the 32-line GT.

    Args:
        range_32: Float32 range [32,W].
        intensity_32: Float32 intensity [32,W].
        mask_32: Bool validity mask [32,W].

    Returns:
        range_16 [16,W] float32, intensity_16 [16,W] float32,
        mask_16 [16,W] bool, and ring_ids_16 [16] int32 equal to
        [0,4,8,...,60] in the original 64-line image.
    """
    range_array, intensity_array, mask_array = _validate_components(
        range_32, intensity_32, mask_32, expected_height=32
    )
    ring_ids_16 = np.arange(0, 64, 4, dtype=np.int32)
    return (
        range_array[::2].copy(),
        intensity_array[::2].copy(),
        mask_array[::2].copy(),
        ring_ids_16,
    )
