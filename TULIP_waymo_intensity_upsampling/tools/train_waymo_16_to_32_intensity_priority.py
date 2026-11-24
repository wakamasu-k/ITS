#!/usr/bin/env python3
"""Train Waymo 16→32 with explicit range/intensity loss weights."""
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
    """Parse intensity-priority training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-train-samples", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--range-weight", type=float, default=0.05)
    parser.add_argument("--intensity-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def weighted_loss(prediction: torch.Tensor, target: torch.Tensor,
                  valid_mask: torch.Tensor, range_weight: float,
                  intensity_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute weighted 2-channel masked L1.

    Input shapes: prediction/target ``[B,2,32,W]`` and mask ``[B,32,W]``.
    Output shapes: total, range, and intensity scalar tensors.
    """
    if range_weight <= 0 or intensity_weight <= 0:
        raise ValueError("loss weights must be positive")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    mask = valid_mask.to(dtype=prediction.dtype)
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("valid mask contains no pixels")
    range_loss = ((prediction[:, 0] - target[:, 0]).abs() * mask).sum() / count
    intensity_loss = ((prediction[:, 1] - target[:, 1]).abs() * mask).sum() / count
    total = (range_weight * range_loss + intensity_weight * intensity_loss) / (
        range_weight + intensity_weight
    )
    return total, range_loss, intensity_loss


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
    """Run a raw-domain intensity-priority training experiment."""
    args = parse_args()
    if args.max_train_samples < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("sample, epoch, and batch-size values must be positive")
    if args.lr <= 0 or args.range_weight <= 0 or args.intensity_weight <= 0:
        raise ValueError("lr and loss weights must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    dataset = Waymo16To32Dataset(args.index, dataset_role="train")
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
        "metric_value_domain": "raw_range_and_raw_intensity",
        "checkpoint": str(args.output),
    }, ensure_ascii=False), flush=True)

    for epoch in range(args.epochs):
        model.train()
        totals = {"loss": 0.0, "range": 0.0, "intensity": 0.0}
        steps = 0
        for batch in loader:
            input_signal = batch["input"].to(device)
            target_signal = batch["target"].to(device)
            target_mask = batch["target_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(input_signal, target_signal, target_mask)
            loss, range_loss, intensity_loss = weighted_loss(
                prediction, target_signal, target_mask,
                args.range_weight, args.intensity_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["range"] += float(range_loss.detach())
            totals["intensity"] += float(intensity_loss.detach())
            steps += 1
        metrics = {name: value / steps for name, value in totals.items()}
        epoch_losses.append(metrics)
        print(json.dumps({"epoch": epoch, **metrics}, ensure_ascii=False), flush=True)

    save_checkpoint(args.output, {
        "task": "Waymo TOP LiDAR 16-to-32 intensity-priority upsampling",
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
