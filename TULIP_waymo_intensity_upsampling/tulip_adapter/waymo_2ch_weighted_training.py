"""Channel-weighted loss helpers for a Waymo 2-channel experiment."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WeightedTwoChannelLosses:
    """Weighted all-valid and generated-row losses."""

    all_valid: torch.Tensor
    generated: torch.Tensor


def _validate_weights(range_weight: float, intensity_weight: float) -> tuple[float, float]:
    """Validate non-negative channel weights with a positive total."""
    range_value = float(range_weight)
    intensity_value = float(intensity_weight)
    if not torch.isfinite(torch.tensor([range_value, intensity_value])).all():
        raise ValueError("channel weights must be finite")
    if range_value < 0 or intensity_value < 0:
        raise ValueError("channel weights must be non-negative")
    if range_value + intensity_value <= 0:
        raise ValueError("at least one channel weight must be positive")
    return range_value, intensity_value


def weighted_two_channel_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    range_weight: float,
    intensity_weight: float,
) -> torch.Tensor:
    """Compute weighted mean L1 over valid pixels and two channels.

    Args:
        prediction: Float tensor ``[B,2,64,W]``.
        target: Float tensor ``[B,2,64,W]``.
        valid_mask: Boolean tensor ``[B,64,W]``.
        range_weight: Weight applied to channel 0.
        intensity_weight: Weight applied to channel 1.

    Returns:
        Scalar weighted mean L1 loss.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError("prediction and target must be [B,2,64,W]")
    expected_mask = prediction.shape[:1] + prediction.shape[2:]
    if valid_mask.shape != expected_mask:
        raise ValueError("valid_mask must be [B,64,W]")
    range_value, intensity_value = _validate_weights(
        range_weight, intensity_weight
    )
    channel_weights = prediction.new_tensor(
        [range_value, intensity_value]
    ).view(1, 2, 1, 1)
    mask = valid_mask.bool().unsqueeze(1).to(prediction.dtype)
    denominator = mask.sum() * channel_weights.sum()
    if denominator.item() == 0:
        raise ValueError("valid_mask contains no pixels")
    return (
        (prediction - target).abs() * mask * channel_weights
    ).sum() / denominator


def compute_weighted_two_channel_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    generated_mask: torch.Tensor,
    *,
    range_weight: float,
    intensity_weight: float,
) -> WeightedTwoChannelLosses:
    """Compute weighted losses for all valid and generated rows."""
    return WeightedTwoChannelLosses(
        all_valid=weighted_two_channel_l1(
            prediction, target, target_mask,
            range_weight=range_weight, intensity_weight=intensity_weight,
        ),
        generated=weighted_two_channel_l1(
            prediction, target, generated_mask,
            range_weight=range_weight, intensity_weight=intensity_weight,
        ),
    )
