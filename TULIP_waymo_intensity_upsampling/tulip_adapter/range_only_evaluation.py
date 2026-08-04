"""Fair evaluation utilities for Waymo range-only 32-to-64 models."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

@dataclass(frozen=True)
class EvaluationMetrics:
    all_valid_l1: float
    generated_l1: float

def _check_input(values, mask):
    if values.ndim == 3:
        values = values.unsqueeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if values.ndim != 4 or values.shape[1] != 1 or values.shape[2] != 32:
        raise ValueError("input_range must be [B,1,32,W]")
    if mask.shape != values.shape[:1] + values.shape[2:]:
        raise ValueError("input_mask must be [B,32,W]")
    return values, mask.bool()

def interpolate_32_to_64(input_range, input_mask, *, mode):
    values, valid = _check_input(input_range, input_mask)
    output = values.new_zeros((values.shape[0], 1, 64, values.shape[-1]))
    output[:, :, 0::2] = values
    left, right = values[:, :, :-1], values[:, :, 1:]
    left_ok, right_ok = valid[:, :-1].unsqueeze(1), valid[:, 1:].unsqueeze(1)
    if mode == "nearest":
        between = torch.where(left_ok, left, torch.where(right_ok, right, 0.0))
    elif mode == "linear":
        count = left_ok.to(values.dtype) + right_ok.to(values.dtype)
        between = torch.where(count > 0,
            (left * left_ok + right * right_ok) / count.clamp_min(1), 0.0)
    else:
        raise ValueError(f"unsupported interpolation mode: {mode}")
    output[:, :, 1:63:2] = between
    output[:, :, 63] = torch.where(valid[:, -1].unsqueeze(1),
                                    values[:, :, -1], 0.0)
    return output

def masked_metrics(prediction, target, target_mask, generated_mask):
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must share [B,1,H,W] shape")
    expected = prediction.shape[:1] + prediction.shape[2:]
    if target_mask.shape != expected or generated_mask.shape != expected:
        raise ValueError("masks must be [B,H,W]")
    error = (prediction - target).abs()
    def mean(mask):
        mask = mask.bool().unsqueeze(1)
        if not mask.any():
            raise ValueError("evaluation mask contains no pixels")
        return float(error[mask].mean().item())
    return EvaluationMetrics(mean(target_mask), mean(generated_mask))

def prediction_statistics(prediction, valid_mask):
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("prediction must be [B,1,H,W]")
    if valid_mask.shape != prediction.shape[:1] + prediction.shape[2:]:
        raise ValueError("valid_mask must be [B,H,W]")
    values = prediction[valid_mask.bool().unsqueeze(1)]
    if values.numel() == 0:
        raise ValueError("valid_mask contains no pixels")
    return {"prediction_valid_min_m": float(values.min()),
            "prediction_valid_max_m": float(values.max()),
            "prediction_valid_mean_m": float(values.mean()),
            "prediction_valid_negative_count": int((values < 0).sum()),
            "prediction_valid_count": int(values.numel())}

def save_difference_image(path: Path, target, prediction, linear,
                          target_mask, generated_mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def arr(value): return value.detach().cpu().squeeze().numpy()
    gt, pred, lin = arr(target), arr(prediction), arr(linear)
    valid, generated = arr(target_mask).astype(bool), arr(generated_mask).astype(bool)
    panels = [
        (np.where(valid, gt, np.nan), "Ground truth range", "viridis"),
        (np.where(valid, pred, np.nan), "TULIP prediction", "viridis"),
        (np.where(generated, np.abs(pred - gt), np.nan),
         "|TULIP - GT| on generated rows", "magma"),
        (np.where(generated, np.abs(lin - gt), np.nan),
         "|linear - GT| on generated rows", "magma")]
    fig, axes = plt.subplots(4, 1, figsize=(16, 9), constrained_layout=True)
    for axis, (values, title, cmap) in zip(axes, panels):
        image = axis.imshow(values, aspect="auto", vmin=0, cmap=cmap)
        axis.set_title(title); axis.set_ylabel("row")
        fig.colorbar(image, ax=axis, label="range [m]")
    axes[-1].set_xlabel("range-image column")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120); plt.close(fig)
