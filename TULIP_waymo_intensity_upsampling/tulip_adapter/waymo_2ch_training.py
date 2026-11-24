"""Training and evaluation helpers for Waymo two-channel TULIP."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .waymo_32_to_64_tulip import masked_l1_2ch


@dataclass
class TwoChannelLosses:
    """All-valid and generated-row losses for range and intensity together."""

    all_valid: torch.Tensor
    generated: torch.Tensor


@dataclass
class TwoChannelMetrics:
    """Per-channel masked L1 metrics in the data value domain."""

    all_valid_range: float
    generated_range: float
    all_valid_intensity: float
    generated_intensity: float


def validate_range_scale(range_scale: float) -> float:
    """Validate the range-only normalization scale."""
    scale = float(range_scale)
    if not torch.isfinite(torch.tensor(scale)) or scale <= 0:
        raise ValueError("range_scale must be finite and positive")
    return scale


def normalize_two_channel(
    input_signal: torch.Tensor,
    target_signal: torch.Tensor,
    range_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize only channel 0 and leave intensity channel unchanged.

    Args:
        input_signal: Tensor ``[B,2,32,W]``.
        target_signal: Tensor ``[B,2,64,W]``.
        range_scale: Positive divisor for range values in meters.

    Returns:
        New tensors with the same shapes and dtype.
    """
    scale = validate_range_scale(range_scale)
    normalized_input = input_signal.clone()
    normalized_target = target_signal.clone()
    normalized_input[:, 0] = normalized_input[:, 0] / scale
    normalized_target[:, 0] = normalized_target[:, 0] / scale
    return normalized_input, normalized_target


def restore_two_channel(
    prediction: torch.Tensor, range_scale: float
) -> torch.Tensor:
    """Restore channel 0 to meters while leaving intensity unchanged."""
    scale = validate_range_scale(range_scale)
    restored = prediction.clone()
    restored[:, 0] = restored[:, 0] * scale
    return restored


def prepare_two_channel_batch(
    batch: dict[str, torch.Tensor], device: str
) -> tuple[torch.Tensor, ...]:
    """Move a Dataset batch to a device and zero invalid input/target values.

    Shapes are input ``[B,2,32,W]``, target ``[B,2,64,W]``, and masks
    ``[B,32,W]``/``[B,64,W]``.
    """
    input_signal = batch["input"].to(device)
    target_signal = batch["target"].to(device)
    input_mask = batch["input_mask"].to(device).bool()
    target_mask = batch["target_mask"].to(device).bool()
    generated_mask = batch["generated_mask"].to(device).bool()
    input_signal = input_signal.masked_fill(
        ~input_mask.unsqueeze(1), 0.0
    )
    target_signal = target_signal.masked_fill(
        ~target_mask.unsqueeze(1), 0.0
    )
    return input_signal, target_signal, target_mask, generated_mask, input_mask


def compute_two_channel_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    generated_mask: torch.Tensor,
) -> TwoChannelLosses:
    """Compute mean L1 over both channels for two spatial masks."""
    return TwoChannelLosses(
        all_valid=masked_l1_2ch(prediction, target, target_mask),
        generated=masked_l1_2ch(prediction, target, generated_mask),
    )


def select_two_channel_loss(
    losses: TwoChannelLosses, scope: str
) -> torch.Tensor:
    """Select the training objective without changing the model loss."""
    if scope == "all_valid":
        return losses.all_valid
    if scope == "generated":
        return losses.generated
    raise ValueError(f"unsupported loss scope: {scope}")


def interpolate_32_to_64_2ch(
    input_signal: torch.Tensor,
    input_mask: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Interpolate both channels from observed even rows to 64 rows.

    Args:
        input_signal: Tensor ``[B,2,32,W]``.
        input_mask: Boolean tensor ``[B,32,W]``.
        mode: ``nearest`` or ``linear``.

    Returns:
        Tensor ``[B,2,64,W]`` in the same value domain as the input.
    """
    if input_signal.ndim != 4 or input_signal.shape[1:3] != (2, 32):
        raise ValueError("input_signal must be [B,2,32,W]")
    if input_mask.shape != input_signal.shape[:1] + input_signal.shape[2:]:
        raise ValueError("input_mask must be [B,32,W]")
    valid = input_mask.bool()
    output = input_signal.new_zeros(
        (input_signal.shape[0], 2, 64, input_signal.shape[-1])
    )
    output[:, :, 0::2] = input_signal
    left, right = input_signal[:, :, :-1], input_signal[:, :, 1:]
    left_ok = valid[:, :-1].unsqueeze(1)
    right_ok = valid[:, 1:].unsqueeze(1)
    if mode == "nearest":
        between = torch.where(left_ok, left, torch.where(right_ok, right, 0.0))
    elif mode == "linear":
        count = left_ok.to(input_signal.dtype) + right_ok.to(input_signal.dtype)
        between = torch.where(
            count > 0,
            (left * left_ok + right * right_ok) / count.clamp_min(1),
            0.0,
        )
    else:
        raise ValueError(f"unsupported interpolation mode: {mode}")
    output[:, :, 1:63:2] = between
    output[:, :, 63] = torch.where(
        valid[:, -1].unsqueeze(1), input_signal[:, :, -1], 0.0
    )
    return output


def two_channel_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    generated_mask: torch.Tensor,
) -> TwoChannelMetrics:
    """Return range/intensity MAE for all-valid and generated rows."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must share [B,2,64,W]")
    expected = prediction.shape[:1] + prediction.shape[2:]
    if target_mask.shape != expected or generated_mask.shape != expected:
        raise ValueError("masks must be [B,64,W]")

    def mean(channel: int, mask: torch.Tensor) -> float:
        values = (prediction[:, channel] - target[:, channel]).abs()[mask]
        if values.numel() == 0:
            raise ValueError("evaluation mask contains no pixels")
        return float(values.mean().item())

    return TwoChannelMetrics(
        all_valid_range=mean(0, target_mask.bool()),
        generated_range=mean(0, generated_mask.bool()),
        all_valid_intensity=mean(1, target_mask.bool()),
        generated_intensity=mean(1, generated_mask.bool()),
    )


def channel_prediction_statistics(
    prediction: torch.Tensor, valid_mask: torch.Tensor
) -> dict[str, float | int]:
    """Summarize valid prediction values separately for both channels."""
    if prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError("prediction must be [B,2,H,W]")
    if valid_mask.shape != prediction.shape[:1] + prediction.shape[2:]:
        raise ValueError("valid_mask must be [B,H,W]")
    result: dict[str, float | int] = {}
    for channel, name in ((0, "range"), (1, "intensity")):
        values = prediction[:, channel][valid_mask.bool()]
        if values.numel() == 0:
            raise ValueError("valid_mask contains no pixels")
        result[f"prediction_{name}_valid_min"] = float(values.min())
        result[f"prediction_{name}_valid_max"] = float(values.max())
        result[f"prediction_{name}_valid_mean"] = float(values.mean())
        result[f"prediction_{name}_valid_negative_count"] = int(
            (values < 0).sum()
        )
    return result
