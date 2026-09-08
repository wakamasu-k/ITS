"""Tests for the controlled Waymo 16-to-32 intensity experiment."""
from __future__ import annotations

import os

import pytest
import torch

from tools.evaluate_waymo_16_to_32_controlled import (
    binary_counts,
    generated_only_mask,
    resolve_checkpoint_contract,
    restore_raw_domain,
)
from tools.train_waymo_16_to_32_log1p_moderate_tail import (
    build_intensity_weights,
    controlled_generated_loss,
    pointwise_intensity_objective,
    set_reproducible_seed,
)


def test_moderate_band_boundaries_use_raw_target_intensity() -> None:
    """Assign every threshold boundary to the intended moderate band."""
    raw = torch.tensor(
        [[[0.0, 1.999, 2.0, 19.999, 20.0, 99.999, 100.0, 999.9, 1000.0]]],
        dtype=torch.float32,
    )
    mask = torch.ones_like(raw, dtype=torch.bool)
    result = build_intensity_weights(torch.log1p(raw), mask, profile="moderate")
    expected = torch.tensor(
        [[[1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 2.0, 2.0, 2.0]]],
        dtype=torch.float32,
    )
    assert torch.allclose(result, expected)


def test_baseline_weights_are_one_and_invalid_pixels_are_zero() -> None:
    """Keep baseline unweighted while excluding invalid pixels entirely."""
    target = torch.log1p(torch.tensor([[[0.0, 1000.0, 2.0]]]))
    mask = torch.tensor([[[True, False, True]]])
    result = build_intensity_weights(target, mask, profile="baseline")
    assert torch.equal(result, torch.tensor([[[1.0, 0.0, 1.0]]]))


def test_controlled_loss_excludes_observed_and_invalid_rows() -> None:
    """Calculate both channels only on valid generated odd rows."""
    target = torch.zeros((1, 2, 4, 1), dtype=torch.float32)
    prediction = torch.zeros_like(target)
    prediction[:, 0, 0, 0] = 1000.0
    prediction[:, 1, 2, 0] = 1000.0
    prediction[:, 0, 1, 0] = 1.0
    prediction[:, 0, 3, 0] = 3.0
    prediction[:, 1, 1, 0] = 2.0
    prediction[:, 1, 3, 0] = 4.0
    valid = torch.ones((1, 4, 1), dtype=torch.bool)
    generated = torch.zeros_like(valid)
    generated[:, 1::2] = True
    result = controlled_generated_loss(
        prediction,
        target,
        valid,
        generated,
        profile="baseline",
        range_weight=1.0,
        intensity_weight=1.0,
    )
    assert result["range"].item() == pytest.approx(2.0)
    assert result["intensity_unweighted_log1p"].item() == pytest.approx(3.0)
    assert result["intensity_weighted_log1p"].item() == pytest.approx(3.0)
    assert result["total"].item() == pytest.approx(2.5)


def test_moderate_loss_is_normalized_by_sum_of_weights() -> None:
    """Use moderate weights without changing loss magnitude by pixel count."""
    target = torch.zeros((1, 2, 4, 1), dtype=torch.float32)
    prediction = torch.zeros_like(target)
    target[:, 1, 1, 0] = torch.log1p(torch.tensor(1.0))
    target[:, 1, 3, 0] = torch.log1p(torch.tensor(1000.0))
    valid = torch.ones((1, 4, 1), dtype=torch.bool)
    generated = torch.zeros_like(valid)
    generated[:, 1::2] = True
    result = controlled_generated_loss(
        prediction,
        target,
        valid,
        generated,
        profile="moderate",
        range_weight=1.0,
        intensity_weight=1.0,
    )
    expected = (
        torch.log1p(torch.tensor(1.0)) + 2.0 * torch.log1p(torch.tensor(1000.0))
    ) / 3.0
    assert result["intensity_weighted_log1p"].item() == pytest.approx(
        expected.item(), rel=1e-6
    )
    assert result["weighted_pixel_fraction"].item() == pytest.approx(0.5)


def test_empty_generated_mask_is_rejected() -> None:
    """Fail explicitly when no generated target contributes to training."""
    value = torch.zeros((1, 2, 4, 1), dtype=torch.float32)
    mask = torch.ones((1, 4, 1), dtype=torch.bool)
    with pytest.raises(ValueError, match="contains no pixels"):
        controlled_generated_loss(
            value,
            value,
            mask,
            torch.zeros_like(mask),
            profile="baseline",
            range_weight=1.0,
            intensity_weight=1.0,
        )


def test_huber_objective_caps_large_log_error_gradient() -> None:
    """Use standard Huber values without dropping any high-error element."""
    error = torch.tensor([0.0, 0.1, 0.2, 1.0], requires_grad=True)
    result = pointwise_intensity_objective(
        error,
        loss_name="huber",
        huber_delta=0.2,
    )
    expected = torch.tensor([0.0, 0.005, 0.02, 0.18])
    assert torch.allclose(result, expected)
    result.sum().backward()
    assert torch.allclose(error.grad, torch.tensor([0.0, 0.1, 0.2, 0.2]))


def test_huber_loss_keeps_high_intensity_target_in_mask() -> None:
    """Retain a high-reflectivity target while robustifying its log error."""
    target = torch.zeros((1, 2, 2, 1), dtype=torch.float32)
    prediction = torch.zeros_like(target)
    target[:, 1, 1, 0] = torch.log1p(torch.tensor(100000.0))
    valid = torch.ones((1, 2, 1), dtype=torch.bool)
    generated = torch.tensor([[[False], [True]]])
    result = controlled_generated_loss(
        prediction,
        target,
        valid,
        generated,
        profile="baseline",
        range_weight=1.0,
        intensity_weight=1.0,
        intensity_loss="huber",
        huber_delta=0.2,
    )
    assert result["valid_generated_pixels"].item() == 1
    assert result["intensity_unweighted_log1p"].item() > 10.0
    assert result["intensity_objective"].item() > 0.0
    assert result["intensity_objective"].item() < 3.0


def test_generated_only_mask_keeps_odd_rows() -> None:
    """Retain rows 1 and 3 in a synthetic four-row view after adaptation."""
    mask = torch.ones((1, 32, 2), dtype=torch.bool)
    mask[:, 3, 1] = False
    result = generated_only_mask(mask)
    assert not result[:, 0::2].any()
    assert result[0, 1, 0]
    assert result[0, 3, 0]
    assert not result[0, 3, 1]


def test_restore_raw_domain_round_trip() -> None:
    """Restore metres and raw intensity without clamping negative predictions."""
    value = torch.tensor(
        [[[[0.5]], [[torch.log1p(torch.tensor(3.0)).item()]]]], dtype=torch.float32
    )
    restored = restore_raw_domain(
        value, range_scale_m=75.0, intensity_transform="log1p"
    )
    assert restored.shape == value.shape
    assert restored[0, 0, 0, 0].item() == pytest.approx(37.5)
    assert restored[0, 1, 0, 0].item() == pytest.approx(3.0)


def test_binary_counts_are_correct() -> None:
    """Compute precision/recall source counts on the supplied mask only."""
    target = torch.tensor([[[0.0, 100.0, 1000.0, 2000.0]]])
    prediction = torch.tensor([[[1000.0, 0.0, 1000.0, 500.0]]])
    mask = torch.tensor([[[False, True, True, True]]])
    result = binary_counts(target, prediction, mask, threshold=1000.0)
    assert result == {
        "target_positive": 2,
        "predicted_positive": 1,
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 1,
    }


def test_checkpoint_contract_rejects_preprocessing_mismatch() -> None:
    """Prevent evaluation with a transform different from training."""
    checkpoint = {
        "args": {"range_scale_m": 75.0, "intensity_transform": "log1p"}
    }
    assert resolve_checkpoint_contract(
        checkpoint,
        range_scale_override=None,
        intensity_transform_override=None,
    ) == (75.0, "log1p")
    with pytest.raises(ValueError, match="range scale mismatch"):
        resolve_checkpoint_contract(
            checkpoint,
            range_scale_override=120.0,
            intensity_transform_override="log1p",
        )
    with pytest.raises(ValueError, match="intensity transform mismatch"):
        resolve_checkpoint_contract(
            checkpoint,
            range_scale_override=75.0,
            intensity_transform_override="raw",
        )


def test_seed_reproduces_torch_random_values() -> None:
    """Generate identical initial values when the same seed is reused."""
    set_reproducible_seed(42)
    first = torch.rand(8)
    set_reproducible_seed(42)
    second = torch.rand(8)
    assert torch.equal(first, second)


def test_seed_configures_deterministic_cublas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure the CUDA workspace before deterministic CuBLAS operations."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    set_reproducible_seed(7)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
