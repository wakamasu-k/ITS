"""Unit tests for native Waymo 64/32/16 line selection."""

import numpy as np
import pytest

from waymo_preprocess.waymo_range_image import (
    extract_range_intensity,
    make_16line_input,
    make_32line_gt,
)


def _synthetic_native_range_image(width: int = 5) -> np.ndarray:
    """Create a deterministic native [64,W,2] test image."""
    image = np.zeros((64, width, 2), dtype=np.float32)
    image[..., 0] = np.arange(64, dtype=np.float32)[:, None] + 1.0
    image[..., 1] = np.arange(width, dtype=np.float32)[None, :] + 10.0
    return image


def test_extract_returns_float32_and_bool_mask() -> None:
    """Extraction returns required dtypes and sanitizes invalid pixels."""
    native = _synthetic_native_range_image()
    native[3, 1, 0] = 0.0
    native[4, 2, 0] = -1.0
    native[5, 3, 0] = np.nan
    native[6, 4, 1] = np.inf

    range_image, intensity_image, mask = extract_range_intensity(native)

    assert range_image.shape == (64, 5)
    assert intensity_image.shape == (64, 5)
    assert mask.shape == (64, 5)
    assert range_image.dtype == np.float32
    assert intensity_image.dtype == np.float32
    assert mask.dtype == np.bool_
    assert not np.isnan(range_image).any()
    assert not np.isinf(range_image).any()
    assert not np.isnan(intensity_image).any()
    assert not np.isinf(intensity_image).any()
    for row, col in ((3, 1), (4, 2), (5, 3), (6, 4)):
        assert intensity_image[row, col] == 0.0
        assert not mask[row, col]


def test_make_32line_gt_selects_even_native_rows() -> None:
    """The 32-line GT equals native rows 0,2,...,62."""
    native = _synthetic_native_range_image()
    range_64, intensity_64, mask_64 = extract_range_intensity(native)
    range_32, intensity_32, mask_32, ring_ids_32 = make_32line_gt(
        range_64, intensity_64, mask_64
    )

    assert range_32.shape == (32, 5)
    assert intensity_32.shape == (32, 5)
    assert mask_32.shape == (32, 5)
    np.testing.assert_array_equal(range_32, range_64[0::2])
    np.testing.assert_array_equal(intensity_32, intensity_64[0::2])
    np.testing.assert_array_equal(mask_32, mask_64[0::2])
    np.testing.assert_array_equal(ring_ids_32, np.arange(0, 64, 2, dtype=np.int32))


def test_make_16line_input_selects_even_32line_rows() -> None:
    """The 16-line input equals 32-line rows 0,2,...,30."""
    native = _synthetic_native_range_image()
    range_64, intensity_64, mask_64 = extract_range_intensity(native)
    range_32, intensity_32, mask_32, _ = make_32line_gt(
        range_64, intensity_64, mask_64
    )
    range_16, intensity_16, mask_16, ring_ids_16 = make_16line_input(
        range_32, intensity_32, mask_32
    )

    assert range_16.shape == (16, 5)
    assert intensity_16.shape == (16, 5)
    assert mask_16.shape == (16, 5)
    np.testing.assert_array_equal(range_16, range_32[0::2])
    np.testing.assert_array_equal(intensity_16, intensity_32[0::2])
    np.testing.assert_array_equal(mask_16, mask_32[0::2])
    np.testing.assert_array_equal(ring_ids_16, np.arange(0, 64, 4, dtype=np.int32))


def test_invalid_native_height_is_rejected() -> None:
    """A native image with height other than 64 is rejected."""
    with pytest.raises(ValueError, match="height 64"):
        extract_range_intensity(np.zeros((32, 5, 2), dtype=np.float32))
