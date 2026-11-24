#!/usr/bin/env python3
"""Train Waymo 16-to-32 TULIP with target-dependent tail weighting.

The TULIP model and its Transformer structure are reused unchanged.  This
experiment changes only the training loss: valid generated rows are used,
and target intensity values above a configurable threshold receive an
additional weight.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model


def parse_args() -> argparse.Namespace:
    """Parse tail-weighted training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-train-samples", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--range-weight", type=float, default=0.05)
    parser.add_argument("--intensity-weight", type=float, default=1.0)
    parser.add_argument(
        "--tail-threshold", type=float, default=20.0,
        help="Raw target intensity threshold for the extra tail weight.",
    )
    parser.add_argument(
        "--tail-weight", type=float, default=10.0,
        help="Multiplicative weight for target intensity above the threshold.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--loss-rows", choices=("all", "generated"), default="generated",
        help="Use all valid target rows or generated odd rows only.",
    )
    return parser.parse_args()


def tail_weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    generated_mask: torch.Tensor,
    *,
    range_weight: float,
    intensity_weight: float,
    tail_threshold: float,
    tail_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute generated-row 2-channel L1 with an intensity tail weight.

    Input shapes: ``prediction`` and ``target`` are ``[B,2,32,W]``;
    ``valid_mask`` and ``generated_mask`` are ``[B,32,W]``.
    Output shapes: total loss, range loss, intensity loss, tail fraction,
    and weighted-tail intensity loss are scalar tensors.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError("prediction and target must be [B,2,H,W]")
    expected_mask_shape = prediction.shape[:1] + prediction.shape[2:]
    if tuple(valid_mask.shape) != expected_mask_shape:
        raise ValueError("valid_mask must be [B,H,W]")
    if tuple(generated_mask.shape) != expected_mask_shape:
        raise ValueError("generated_mask must be [B,H,W]")
    if range_weight <= 0 or intensity_weight <= 0:
        raise ValueError("range and intensity weights must be positive")
    if tail_threshold < 0 or tail_weight < 1:
        raise ValueError("tail threshold must be non-negative and tail weight >= 1")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must be finite")

    mask = valid_mask.to(dtype=torch.bool) & generated_mask.to(dtype=torch.bool)
    pixel_count = mask.sum()
    if pixel_count.item() == 0:
        raise ValueError("generated valid mask contains no pixels")
    mask_f = mask.to(dtype=prediction.dtype)
    range_loss = ((prediction[:, 0] - target[:, 0]).abs() * mask_f).sum() / pixel_count

    tail = mask & (target[:, 1] > tail_threshold)
    intensity_weights = torch.ones_like(mask_f)
    intensity_weights[tail] = tail_weight
    weighted_mask = mask_f * intensity_weights
    weighted_count = weighted_mask.sum()
    intensity_error = (prediction[:, 1] - target[:, 1]).abs()
    intensity_loss = (intensity_error * weighted_mask).sum() / weighted_count
    unweighted_intensity_loss = (intensity_error * mask_f).sum() / pixel_count
    total = (range_weight * range_loss + intensity_weight * intensity_loss) / (
        range_weight + intensity_weight
    )
    tail_fraction = tail.to(dtype=prediction.dtype).sum() / pixel_count
    return total, range_loss, unweighted_intensity_loss, tail_fraction, intensity_loss


def save_checkpoint(path: Path, value: dict) -> None:
    """Save a checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Run one tail-weighted raw-domain training experiment."""
    args = parse_args()
    if args.max_train_samples < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("sample, epoch, and batch-size values must be positive")
    if args.lr <= 0 or args.range_weight <= 0 or args.intensity_weight <= 0:
        raise ValueError("lr and channel loss weights must be positive")
    if args.tail_threshold < 0 or args.tail_weight < 1:
        raise ValueError("tail threshold must be non-negative and tail weight >= 1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    dataset = Waymo16To32Dataset(args.index, dataset_role="train", intensity_transform="raw")
    count = min(args.max_train_samples, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(count)), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    epoch_losses: list[dict[str, float]] = []
    print(json.dumps({
        "device": str(device), "samples": count, "epochs": args.epochs,
        "batch_size": args.batch_size, "range_weight": args.range_weight,
        "intensity_weight": args.intensity_weight,
        "tail_threshold": args.tail_threshold, "tail_weight": args.tail_weight,
        "loss_rows": args.loss_rows,
        "intensity_domain": "raw",
        "metric_value_domain": "raw_range_and_raw_intensity",
        "checkpoint": str(args.output),
    }, ensure_ascii=False), flush=True)

    for epoch in range(args.epochs):
        model.train()
        totals = {
            "loss": 0.0, "range": 0.0, "intensity": 0.0,
            "weighted_intensity": 0.0, "tail_fraction": 0.0,
        }
        steps = 0
        for batch in loader:
            input_signal = batch["input"].to(device)
            target_signal = batch["target"].to(device)
            target_mask = batch["target_mask"].to(device)
            generated_mask = batch["generated_mask"].to(device)
            loss_mask = generated_mask if args.loss_rows == "generated" else target_mask
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(input_signal, target_signal, target_mask)
            loss, range_loss, intensity_loss, tail_fraction, weighted_intensity = tail_weighted_loss(
                prediction, target_signal, target_mask, loss_mask,
                range_weight=args.range_weight,
                intensity_weight=args.intensity_weight,
                tail_threshold=args.tail_threshold,
                tail_weight=args.tail_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["range"] += float(range_loss.detach())
            totals["intensity"] += float(intensity_loss.detach())
            totals["weighted_intensity"] += float(weighted_intensity.detach())
            totals["tail_fraction"] += float(tail_fraction.detach())
            steps += 1
        metrics = {name: value / steps for name, value in totals.items()}
        epoch_losses.append(metrics)
        print(json.dumps({"epoch": epoch, **metrics}, ensure_ascii=False), flush=True)

    save_checkpoint(args.output, {
        "task": "Waymo TOP LiDAR 16-to-32 tail-weighted intensity upsampling",
        "epoch": args.epochs - 1,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "args": vars(args), "epoch_losses": epoch_losses,
    })
    print(json.dumps({
        "checkpoint_saved": str(args.output),
        "initial_loss": epoch_losses[0]["loss"],
        "final_loss": epoch_losses[-1]["loss"],
        "decreased": epoch_losses[-1]["loss"] < epoch_losses[0]["loss"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
