#!/usr/bin/env python3
"""Train Waymo 16→32 with log1p intensity and equal channel loss."""
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
    """Parse log1p training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-train-samples", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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
    """Run equal-weight log1p-domain training."""
    args = parse_args()
    if args.max_train_samples < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("sample, epoch, and batch-size values must be positive")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    dataset = Waymo16To32Dataset(
        args.index, dataset_role="train", intensity_transform="log1p"
    )
    count = min(args.max_train_samples, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(count)), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    epoch_losses: list[float] = []
    print(json.dumps({
        "device": str(device), "samples": count, "epochs": args.epochs,
        "batch_size": args.batch_size, "intensity_transform": "log1p",
        "metric_value_domain": "log1p_training_raw_evaluation",
        "checkpoint": str(args.output),
    }, ensure_ascii=False), flush=True)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        steps = 0
        for batch in loader:
            input_signal = batch["input"].to(device)
            target_signal = batch["target"].to(device)
            target_mask = batch["target_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, loss = model(input_signal, target_signal, target_mask)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        mean_loss = total / steps
        epoch_losses.append(mean_loss)
        print(json.dumps({"epoch": epoch, "mean_loss": mean_loss}, ensure_ascii=False), flush=True)

    save_checkpoint(args.output, {
        "task": "Waymo TOP LiDAR 16-to-32 log1p intensity upsampling",
        "epoch": args.epochs - 1,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "args": vars(args), "epoch_losses": epoch_losses,
    })
    print(json.dumps({
        "checkpoint_saved": str(args.output),
        "initial_mean_loss": epoch_losses[0],
        "final_mean_loss": epoch_losses[-1],
        "decreased": epoch_losses[-1] < epoch_losses[0],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
