"""Training helpers for the Waymo range-only TULIP baseline."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class RangeLosses:
    all_valid: torch.Tensor
    generated: torch.Tensor


def masked_l1(prediction: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("range tensors must be [B,1,H,W]")
    if mask.shape != prediction.shape[:1] + prediction.shape[2:]:
        raise ValueError("mask must be [B,H,W]")
    weights = mask.unsqueeze(1).to(prediction.dtype)
    count = weights.sum()
    if count.item() == 0:
        raise ValueError("mask contains no valid pixels")
    return ((prediction - target).abs() * weights).sum() / count


def prepare_range_batch(batch: dict[str, torch.Tensor], device: str
                        ) -> tuple[torch.Tensor, ...]:
    input_range = batch["input"].to(device)
    target_range = batch["target"].to(device)
    input_mask = batch["input_mask"].to(device)
    target_mask = batch["target_mask"].to(device)
    generated_mask = batch["generated_mask"].to(device)
    input_range = input_range.masked_fill(~input_mask.unsqueeze(1), 0.0)
    target_range = target_range.masked_fill(~target_mask.unsqueeze(1), 0.0)
    return input_range, target_range, target_mask, generated_mask


def compute_losses(prediction: torch.Tensor, target: torch.Tensor,
                   target_mask: torch.Tensor,
                   generated_mask: torch.Tensor) -> RangeLosses:
    return RangeLosses(
        all_valid=masked_l1(prediction, target, target_mask),
        generated=masked_l1(prediction, target, generated_mask),
    )


def select_training_loss(losses: RangeLosses, scope: str) -> torch.Tensor:
    if scope == "all_valid":
        return losses.all_valid
    if scope == "generated":
        return losses.generated
    raise ValueError(f"unsupported loss scope: {scope}")
