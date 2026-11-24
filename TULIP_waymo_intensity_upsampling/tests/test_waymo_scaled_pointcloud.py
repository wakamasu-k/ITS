"""Tests for scale-aware point-cloud evaluation helpers."""
from __future__ import annotations

import pytest
import torch

from tools.evaluate_waymo_16_to_32_scaled_pointcloud import restore_domain


def test_restore_domain_for_pointcloud() -> None:
    """Range is restored to meters and log1p intensity to raw values."""
    value = torch.tensor([[[[0.5]], [[torch.log(torch.tensor(3.0))]]]])
    restored = restore_domain(value, range_scale_m=75.0, intensity_transform="log1p")
    assert restored[0, 0, 0, 0].item() == pytest.approx(37.5)
    assert restored[0, 1, 0, 0].item() == pytest.approx(2.0)


def test_restore_domain_rejects_invalid_transform() -> None:
    """Unknown transformations are rejected explicitly."""
    with pytest.raises(ValueError):
        restore_domain(torch.zeros((1, 2, 1, 1)), range_scale_m=75.0, intensity_transform="bad")
