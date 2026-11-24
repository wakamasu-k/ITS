"""Tests for scaled Waymo experiment helpers."""
from __future__ import annotations

import pytest
import torch

from tools.evaluate_waymo_16_to_32_scaled import restore_domain


def test_restore_domain_range_and_log1p() -> None:
    """Restore range meters and log1p intensity without changing shape."""
    value = torch.tensor([[[[0.5]], [[0.0]]]], dtype=torch.float32)
    restored = restore_domain(value, range_scale_m=75.0, intensity_transform="log1p")
    assert restored.shape == value.shape
    assert restored[0, 0, 0, 0].item() == pytest.approx(37.5)
    assert restored[0, 1, 0, 0].item() == pytest.approx(0.0)


def test_restore_domain_rejects_invalid_scale() -> None:
    """Reject an invalid range scale explicitly."""
    with pytest.raises(ValueError):
        restore_domain(torch.zeros((1, 2, 1, 1)), range_scale_m=0.0, intensity_transform="raw")
