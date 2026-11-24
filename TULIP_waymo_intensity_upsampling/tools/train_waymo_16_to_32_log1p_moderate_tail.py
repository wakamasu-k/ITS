#!/usr/bin/env python3
"""Train reproducible Waymo 16-to-32 TULIP intensity experiments.

The existing two-channel Waymo TULIP adapter is reused unchanged. Only the
loss contract is controlled: valid generated odd rows, 75 m range scaling,
log1p intensity, and optional bounded target-dependent intensity weights.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model


SCRIPT_VERSION = "1.4.0"
PROFILES = ("baseline", "moderate", "low_0_1", "band_balanced")
INTENSITY_LOSSES = ("l1", "huber")
LOW_INTENSITY_MAX_RAW = 1.0
MODERATE_INTENSITY_BANDS = (
    (0.0, 2.0, 1.0),
    (2.0, 20.0, 1.2),
    (20.0, 100.0, 1.5),
    (100.0, 1000.0, 2.0),
    (1000.0, float("inf"), 2.0),
)
BALANCED_INTENSITY_BANDS = (
    ("0_1", 0.0, 1.0, 0.90),
    ("1_20", 1.0, 20.0, 0.05),
    ("20_100", 20.0, 100.0, 0.02),
    ("100_1000", 100.0, 1000.0, 0.02),
    ("1000_inf", 1000.0, float("inf"), 0.01),
)


def parse_args() -> argparse.Namespace:
    """Parse controlled training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Load model weights only before fine-tuning; optimizer state is not restored.",
    )
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--max-train-samples", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--range-weight", type=float, default=1.0)
    parser.add_argument("--intensity-weight", type=float, default=1.0)
    parser.add_argument("--intensity-loss", choices=INTENSITY_LOSSES, default="l1")
    parser.add_argument("--huber-delta", type=float, default=0.2)
    parser.add_argument("--range-scale-m", type=float, default=75.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels.

    Input shape: scalar integer seed.
    Output shape: none; global RNG and backend state are updated.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    # Required by deterministic CuBLAS matrix multiplication on CUDA >= 10.2.
    # Set this before any CUDA RNG or model operation initializes the context.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def intensity_band_spec(profile: str) -> tuple[tuple[float, float, float], ...]:
    """Return raw-intensity band weights for one experiment profile.

    Input shape: profile string.
    Output shape: tuple of ``(lower, upper, weight)`` scalar triples.
    """
    if profile == "baseline":
        return ((0.0, float("inf"), 1.0),)
    if profile == "moderate":
        return MODERATE_INTENSITY_BANDS
    raise ValueError(f"unsupported profile: {profile}")


def build_intensity_weights(
    target_log1p: torch.Tensor,
    valid_generated_mask: torch.Tensor,
    *,
    profile: str,
) -> torch.Tensor:
    """Build weights using the selected raw-target intensity contract.

    Input shapes: target and mask are ``[B,H,W]`` in log1p and bool domains.
    Output shape: floating-point weights ``[B,H,W]``. Invalid pixels are zero;
    ``low_0_1`` also assigns zero to valid targets above raw intensity 1.
    """
    if target_log1p.ndim != 3:
        raise ValueError("target_log1p must be [B,H,W]")
    if tuple(valid_generated_mask.shape) != tuple(target_log1p.shape):
        raise ValueError("target_log1p and mask shapes differ")
    if not torch.isfinite(target_log1p).all():
        raise ValueError("target_log1p must be finite")
    mask = valid_generated_mask.to(dtype=torch.bool)
    raw_target = torch.expm1(target_log1p)
    if torch.any(raw_target[mask] < -1e-6):
        raise ValueError("valid log1p target decodes to negative intensity")
    weights = torch.zeros_like(target_log1p)
    if profile == "low_0_1":
        selected = mask & (raw_target <= LOW_INTENSITY_MAX_RAW)
        weights[selected] = 1.0
        return weights
    for lower, upper, weight in intensity_band_spec(profile):
        lower_log1p = torch.log1p(target_log1p.new_tensor(lower))
        selected = mask & (target_log1p >= lower_log1p)
        if np.isfinite(upper):
            upper_log1p = torch.log1p(target_log1p.new_tensor(upper))
            selected &= target_log1p < upper_log1p
        weights[selected] = weight
    if mask.any() and torch.any(weights[mask] <= 0):
        raise RuntimeError("intensity bands did not cover every valid target")
    return weights


def pointwise_intensity_objective(
    absolute_error: torch.Tensor,
    *,
    loss_name: str,
    huber_delta: float,
) -> torch.Tensor:
    """Return the pointwise log-domain intensity training objective.

    Input/output shape: arbitrary matching tensor shape. ``huber`` uses the
    standard Huber definition, whose gradient is capped at ``huber_delta``.
    """
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    if torch.any(absolute_error < 0) or not torch.isfinite(absolute_error).all():
        raise ValueError("absolute_error must be finite and non-negative")
    if loss_name == "l1":
        return absolute_error
    if loss_name == "huber":
        delta = absolute_error.new_tensor(huber_delta)
        return torch.where(
            absolute_error <= delta,
            0.5 * absolute_error.square(),
            delta * (absolute_error - 0.5 * delta),
        )
    raise ValueError(f"unsupported intensity loss: {loss_name}")


def band_balanced_intensity_loss(
    pointwise_objective: torch.Tensor,
    target_log1p: torch.Tensor,
    valid_generated_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Combine per-band mean intensity losses with fixed coefficients.

    Input shapes: objective and target are ``[B,H,W]`` floating-point tensors;
    mask is ``[B,H,W]`` bool-compatible. Output shape: a dictionary of scalar
    tensors containing the combined loss and per-band loss/count diagnostics.
    Intervals are ``[0,1]``, ``(1,20]``, ``(20,100]``, ``(100,1000]``, and
    ``(1000,inf)`` in raw Waymo intensity.
    """
    if pointwise_objective.ndim != 3 or target_log1p.ndim != 3:
        raise ValueError("objective and target_log1p must be [B,H,W]")
    if pointwise_objective.shape != target_log1p.shape:
        raise ValueError("objective and target_log1p shapes differ")
    if valid_generated_mask.shape != target_log1p.shape:
        raise ValueError("target_log1p and mask shapes differ")
    if not torch.isfinite(pointwise_objective).all():
        raise ValueError("pointwise objective must be finite")
    if torch.any(pointwise_objective < 0):
        raise ValueError("pointwise objective must be non-negative")
    if not torch.isfinite(target_log1p).all():
        raise ValueError("target_log1p must be finite")

    mask = valid_generated_mask.to(dtype=torch.bool)
    if not mask.any():
        raise ValueError("band-balanced intensity mask contains no pixels")
    raw_target = torch.expm1(target_log1p)
    if torch.any(raw_target[mask] < -1e-6):
        raise ValueError("valid log1p target decodes to negative intensity")

    zero = pointwise_objective.new_zeros(())
    weighted_sum = zero
    active_coefficient_sum = zero
    covered = torch.zeros_like(mask)
    metrics: dict[str, torch.Tensor] = {}
    for label, lower, upper, coefficient in BALANCED_INTENSITY_BANDS:
        lower_log1p = torch.log1p(target_log1p.new_tensor(lower))
        if lower == 0.0:
            selected = mask & (target_log1p >= lower_log1p)
        else:
            selected = mask & (target_log1p > lower_log1p)
        if np.isfinite(upper):
            upper_log1p = torch.log1p(target_log1p.new_tensor(upper))
            selected &= target_log1p <= upper_log1p
        if torch.any(covered & selected):
            raise RuntimeError(f"overlapping balanced intensity band: {label}")
        covered |= selected
        pixel_count = selected.sum()
        active = pixel_count > 0
        if active.item():
            band_loss = pointwise_objective[selected].mean()
            coefficient_tensor = pointwise_objective.new_tensor(coefficient)
            weighted_sum = weighted_sum + coefficient_tensor * band_loss
            active_coefficient_sum = active_coefficient_sum + coefficient_tensor
        else:
            band_loss = zero
        metrics[f"band_{label}_loss_log1p"] = band_loss
        metrics[f"band_{label}_pixels"] = pixel_count.to(
            dtype=pointwise_objective.dtype
        )
        metrics[f"band_{label}_active"] = active.to(
            dtype=pointwise_objective.dtype
        )

    if torch.any(mask & ~covered):
        raise RuntimeError("balanced intensity bands did not cover every valid target")
    if active_coefficient_sum.item() <= 0:
        raise RuntimeError("no balanced intensity band was active")
    metrics["loss"] = weighted_sum / active_coefficient_sum
    return metrics


def controlled_generated_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    generated_mask: torch.Tensor,
    *,
    profile: str,
    range_weight: float,
    intensity_weight: float,
    intensity_loss: str = "l1",
    huber_delta: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Compute a generated-row two-channel loss in the model domain.

    Input shapes: prediction/target ``[B,2,32,W]`` and masks ``[B,32,W]``.
    Output shape: dictionary of scalar tensors, including ``total``.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError("prediction and target must be [B,2,H,W]")
    expected_mask = prediction.shape[:1] + prediction.shape[2:]
    if tuple(target_mask.shape) != expected_mask:
        raise ValueError("target_mask must be [B,H,W]")
    if tuple(generated_mask.shape) != expected_mask:
        raise ValueError("generated_mask must be [B,H,W]")
    if range_weight <= 0 or intensity_weight <= 0:
        raise ValueError("channel weights must be positive")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must be finite")

    mask = target_mask.to(dtype=torch.bool) & generated_mask.to(dtype=torch.bool)
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("valid generated mask contains no pixels")
    mask_f = mask.to(dtype=prediction.dtype)
    range_loss = ((prediction[:, 0] - target[:, 0]).abs() * mask_f).sum() / count
    intensity_error = (prediction[:, 1] - target[:, 1]).abs()
    unweighted_intensity_loss = (intensity_error * mask_f).sum() / count
    pointwise_objective = pointwise_intensity_objective(
        intensity_error,
        loss_name=intensity_loss,
        huber_delta=huber_delta,
    )
    band_metrics: dict[str, torch.Tensor] = {}
    if profile == "band_balanced":
        balanced = band_balanced_intensity_loss(
            pointwise_objective,
            target[:, 1],
            mask,
        )
        weighted_intensity_loss = balanced.pop("loss")
        band_metrics = balanced
        intensity_optimized_count = count
        raw_target = torch.expm1(target[:, 1])
        weighted_fraction = ((raw_target > 1.0) & mask).to(
            dtype=prediction.dtype
        ).sum() / count
    else:
        intensity_weights = build_intensity_weights(target[:, 1], mask, profile=profile)
        intensity_optimized_mask = intensity_weights > 0
        intensity_optimized_count = intensity_optimized_mask.sum()
        weighted_count = intensity_weights.sum()
        if weighted_count.item() <= 0:
            raise ValueError("intensity loss scope contains no pixels")
        weighted_intensity_loss = (
            pointwise_objective * intensity_weights
        ).sum() / weighted_count
        weighted_fraction = (intensity_weights[mask] > 1.0).to(
            dtype=prediction.dtype
        ).mean()
    total = (
        range_weight * range_loss + intensity_weight * weighted_intensity_loss
    ) / (range_weight + intensity_weight)
    optimized_fraction = intensity_optimized_count.to(dtype=prediction.dtype) / count
    result = {
        "total": total,
        "range": range_loss,
        "intensity_unweighted_log1p": unweighted_intensity_loss,
        "intensity_weighted_log1p": weighted_intensity_loss,
        "intensity_objective": weighted_intensity_loss,
        "weighted_pixel_fraction": weighted_fraction,
        "intensity_optimized_pixel_fraction": optimized_fraction,
        "valid_generated_pixels": count.to(dtype=prediction.dtype),
        "intensity_optimized_pixels": intensity_optimized_count.to(
            dtype=prediction.dtype
        ),
    }
    result.update(band_metrics)
    return result


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert CLI arguments to JSON-safe checkpoint metadata."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(path: Path, value: dict[str, Any]) -> None:
    """Save a checkpoint through a temporary file and atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model_initialization(
    model: torch.nn.Module,
    checkpoint_path: Path | None,
) -> dict[str, Any] | None:
    """Strictly initialize a model from a controlled-training checkpoint.

    Input shape: model with an arbitrary parameter tree and an optional scalar
    checkpoint path. Output shape: optional metadata dictionary. Only the
    model parameters are restored; optimizer state is intentionally ignored.
    """
    if checkpoint_path is None:
        return None
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"initialization checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("initialization checkpoint must contain a dictionary")
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise KeyError("initialization checkpoint does not contain model weights")
    model.load_state_dict(state_dict, strict=True)
    contract = checkpoint.get("training_contract")
    source_profile = contract.get("profile") if isinstance(contract, dict) else None
    return {
        "source_checkpoint": str(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "source_profile": source_profile,
        "model_state_strict": True,
        "optimizer_restored": False,
    }


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid or unsafe controlled-training arguments."""
    if args.max_train_samples < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("sample, epoch, and batch-size values must be positive")
    if (
        args.lr <= 0
        or args.range_scale_m <= 0
        or args.grad_clip <= 0
        or args.huber_delta <= 0
    ):
        raise ValueError(
            "lr, range-scale-m, grad-clip, and huber-delta must be positive"
        )
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative")
    if args.range_weight <= 0 or args.intensity_weight <= 0:
        raise ValueError("range and intensity weights must be positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.is_file():
            raise FileNotFoundError(
                f"initialization checkpoint not found: {args.init_checkpoint}"
            )
        if args.init_checkpoint.resolve() == args.output.resolve():
            raise ValueError("init-checkpoint and output must be different files")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output}; pass --overwrite explicitly"
        )


def main() -> None:
    """Train one controlled intensity experiment without changing TULIP."""
    args = parse_args()
    _validate_args(args)
    set_reproducible_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    dataset = Waymo16To32Dataset(
        args.index,
        dataset_role="train",
        intensity_transform="log1p",
        range_scale_m=args.range_scale_m,
    )
    count = min(args.max_train_samples, len(dataset))
    if count < 1:
        raise ValueError("no training samples")
    loader = DataLoader(
        Subset(dataset, range(count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = build_waymo_16_to_32_model(args.tulip_root)
    initialization = load_model_initialization(model, args.init_checkpoint)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    epoch_metrics: list[dict[str, Any]] = []
    if args.profile == "low_0_1":
        contract_bands = None
        band_aggregation = "mean_over_selected_pixels"
    elif args.profile == "band_balanced":
        contract_bands = [
            {
                "label": label,
                "lower": lower,
                "upper": None if not np.isfinite(upper) else upper,
                "coefficient": coefficient,
            }
            for label, lower, upper, coefficient in BALANCED_INTENSITY_BANDS
        ]
        band_aggregation = "coefficient_weighted_mean_of_active_band_means"
    else:
        band_spec = intensity_band_spec(args.profile)
        contract_bands = [
            {
                "lower_inclusive": lower,
                "upper_exclusive": None if not np.isfinite(upper) else upper,
                "weight": weight,
            }
            for lower, upper, weight in band_spec
        ]
        band_aggregation = "weighted_mean_over_pixels"
    contract = {
        "script_version": SCRIPT_VERSION,
        "profile": args.profile,
        "loss_rows": "valid_generated_odd_rows_only",
        "range_scale_m": args.range_scale_m,
        "intensity_transform": "log1p",
        "intensity_band_domain": "raw_target_intensity",
        "intensity_error_domain": "log1p",
        "intensity_loss": args.intensity_loss,
        "huber_delta": args.huber_delta if args.intensity_loss == "huber" else None,
        "intensity_loss_target_scope": (
            {
                "raw_min_inclusive": 0.0,
                "raw_max_inclusive": LOW_INTENSITY_MAX_RAW,
                "raw_above_max_policy": "excluded_from_intensity_loss_only",
            }
            if args.profile == "low_0_1"
            else "all_valid_generated_pixels"
        ),
        "intensity_bands": contract_bands,
        "intensity_band_aggregation": band_aggregation,
        "range_weight": args.range_weight,
        "intensity_weight": args.intensity_weight,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": None,
        "warmup": None,
        "amp": False,
        "gradient_clip_norm": args.grad_clip,
        "seed": args.seed,
        "shuffle": False,
        "num_workers": 0,
        "initialization": initialization,
    }
    print(
        json.dumps(
            {
                "device": str(device),
                "samples": count,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "checkpoint": str(args.output),
                "initialization": initialization,
                "contract": contract,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        totals = {
            "loss": 0.0,
            "range": 0.0,
            "intensity_unweighted_log1p": 0.0,
            "intensity_weighted_log1p": 0.0,
            "weighted_pixel_fraction": 0.0,
            "intensity_optimized_pixel_fraction": 0.0,
        }
        band_loss_sums = {
            label: 0.0 for label, _, _, _ in BALANCED_INTENSITY_BANDS
        }
        band_active_batches = {
            label: 0.0 for label, _, _, _ in BALANCED_INTENSITY_BANDS
        }
        band_pixel_counts = {
            label: 0.0 for label, _, _, _ in BALANCED_INTENSITY_BANDS
        }
        valid_generated_pixel_count = 0.0
        steps = 0
        for batch in loader:
            input_signal = batch["input"].to(device)
            target_signal = batch["target"].to(device)
            target_mask = batch["target_mask"].to(device)
            generated_mask = batch["generated_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(input_signal, target_signal, target_mask)
            losses = controlled_generated_loss(
                prediction,
                target_signal,
                target_mask,
                generated_mask,
                profile=args.profile,
                range_weight=args.range_weight,
                intensity_weight=args.intensity_weight,
                intensity_loss=args.intensity_loss,
                huber_delta=args.huber_delta,
            )
            loss = losses["total"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for name in totals:
                if name != "loss":
                    totals[name] += float(losses[name].detach())
            if args.profile == "band_balanced":
                valid_generated_pixel_count += float(
                    losses["valid_generated_pixels"].detach()
                )
                for label, _, _, _ in BALANCED_INTENSITY_BANDS:
                    active = float(losses[f"band_{label}_active"].detach())
                    band_active_batches[label] += active
                    band_pixel_counts[label] += float(
                        losses[f"band_{label}_pixels"].detach()
                    )
                    if active:
                        band_loss_sums[label] += float(
                            losses[f"band_{label}_loss_log1p"].detach()
                        )
            steps += 1
        if steps == 0:
            raise RuntimeError("training DataLoader produced no batches")
        metrics = {name: value / steps for name, value in totals.items()}
        if args.profile == "band_balanced":
            for label, _, _, _ in BALANCED_INTENSITY_BANDS:
                active_batches = band_active_batches[label]
                metrics[f"band_{label}_loss_log1p"] = (
                    band_loss_sums[label] / active_batches
                    if active_batches > 0
                    else None
                )
                metrics[f"band_{label}_pixel_count"] = int(
                    band_pixel_counts[label]
                )
                metrics[f"band_{label}_pixel_fraction"] = (
                    band_pixel_counts[label] / valid_generated_pixel_count
                    if valid_generated_pixel_count > 0
                    else None
                )
        epoch_metrics.append(metrics)
        print(json.dumps({"epoch": epoch, **metrics}), flush=True)

    save_checkpoint(
        args.output,
        {
            "task": "Waymo TOP LiDAR 16-to-32 controlled two-channel TULIP",
            "format_version": 1,
            "script_version": SCRIPT_VERSION,
            "epoch": args.epochs - 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": _jsonable_args(args),
            "training_contract": contract,
            "epoch_metrics": epoch_metrics,
            "torch_version": torch.__version__,
        },
    )
    print(
        json.dumps(
            {
                "checkpoint_saved": str(args.output),
                "profile": args.profile,
                "initial_loss": epoch_metrics[0]["loss"],
                "final_loss": epoch_metrics[-1]["loss"],
                "decreased": epoch_metrics[-1]["loss"] < epoch_metrics[0]["loss"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
