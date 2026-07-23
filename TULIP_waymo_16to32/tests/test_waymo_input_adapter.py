"""Tests for the Phase 4A Waymo input adapter."""

import numpy as np
import pytest
import torch

from tulip_adapter.waymo_input_adapter import (
    crop_to_native_width,
    hwc_to_chw_tensor,
    normalize_range_intensity,
    pad_width_for_tulip,
)


def _raw() -> np.ndarray:
    """Return a small deterministic raw HWC sample."""
    raw = np.zeros((32, 4, 2), dtype=np.float32)
    raw[..., 0] = 10.0
    raw[..., 1] = np.arange(4, dtype=np.float32)[None, :]
    raw[0, 0] = 0.0
    return raw


def test_range_normalization_and_invalid_zero() -> None:
    """Range divides by caller maximum and invalid pixels remain zero."""
    result = normalize_range_intensity(_raw(), maximum_range=20.0, intensity_mode="none")
    assert result.shape == (32, 4, 2)
    assert result.dtype == np.float32
    assert result[1, 0, 0] == 0.5
    assert result[1, 0, 1] == 0.0
    assert result[0, 0].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("mode", ["none", "clip_1", "log1p"])
def test_intensity_modes(mode: str) -> None:
    """All supported intensity modes return finite float32 HWC arrays."""
    raw = _raw()
    raw[..., 1] = 2.0
    result = normalize_range_intensity(raw, maximum_range=20.0, intensity_mode=mode)  # type: ignore[arg-type]
    expected = {"none": 2.0, "clip_1": 1.0, "log1p": float(np.log1p(2.0))}[mode]
    assert np.allclose(result[1, :, 1], expected)
    assert np.isfinite(result).all()


def test_hwc_to_chw_float32() -> None:
    """HWC channel order becomes CHW torch.float32."""
    tensor = hwc_to_chw_tensor(_raw())
    assert tensor.shape == (2, 32, 4)
    assert tensor.dtype == torch.float32
    assert tensor[0, 1, 0] == 10.0
    assert tensor[1, 1, 2] == 2.0


def test_circular_padding_2650_to_2688() -> None:
    """Padding appends the first 38 columns to the right."""
    tensor = torch.arange(2 * 2 * 5, dtype=torch.float32).reshape(2, 2, 5)
    padded = pad_width_for_tulip(tensor, 8)
    assert padded.shape == (2, 2, 8)
    torch.testing.assert_close(padded[..., :5], tensor)
    torch.testing.assert_close(padded[..., 5:], tensor[..., :3])


def test_padding_bchw_and_crop() -> None:
    """BCHW padding and native-width crop preserve batch/channel dimensions."""
    tensor = torch.zeros((2, 2, 16, 2650), dtype=torch.float32)
    padded = pad_width_for_tulip(tensor, 2688)
    cropped = crop_to_native_width(padded, 2650)
    assert padded.shape == (2, 2, 16, 2688)
    assert cropped.shape == (2, 2, 16, 2650)


def test_adapter_rejects_bad_parameters() -> None:
    """Invalid modes, maximum range, and shrinking widths fail clearly."""
    with pytest.raises(ValueError, match="maximum_range"):
        normalize_range_intensity(_raw(), 0.0)
    with pytest.raises(ValueError, match="intensity_mode"):
        normalize_range_intensity(_raw(), 20.0, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="smaller"):
        pad_width_for_tulip(torch.zeros((2, 4, 8)), 7)
    with pytest.raises(ValueError, match="larger"):
        crop_to_native_width(torch.zeros((2, 4, 8)), 9)
