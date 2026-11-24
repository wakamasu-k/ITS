#!/usr/bin/env python3
"""Overfit one Waymo 16→32 sample to validate the training path."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model


def parse_args() -> argparse.Namespace:
    """Parse one-sample overfit arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def save_checkpoint(path: Path, value: dict) -> None:
    """Save a checkpoint using an atomic same-directory replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Run one-sample forward/backward steps and save the final checkpoint."""
    args = parse_args()
    if args.epochs < 1 or args.lr <= 0:
        raise ValueError("epochs must be positive and lr must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    dataset = Waymo16To32Dataset(args.index, dataset_role="train")
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample_index {args.sample_index} outside dataset length {len(dataset)}"
        )
    sample = dataset[args.sample_index]
    input_signal = sample["input"].unsqueeze(0).to(device)
    target_signal = sample["target"].unsqueeze(0).to(device)
    target_mask = sample["target_mask"].unsqueeze(0).to(device)

    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses: list[float] = []
    print(json.dumps({
        "device": str(device),
        "sample_index": args.sample_index,
        "epochs": args.epochs,
        "input_shape": list(input_signal.shape),
        "target_shape": list(target_signal.shape),
        "checkpoint": str(args.output),
    }, ensure_ascii=False), flush=True)

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction, loss = model(input_signal, target_signal, target_mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}: {float(loss)}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        print(json.dumps({"epoch": epoch, "loss": value}, ensure_ascii=False), flush=True)

    save_checkpoint(args.output, {
        "task": "Waymo TOP LiDAR 16-to-32 range and intensity upsampling",
        "epoch": args.epochs - 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "losses": losses,
    })
    print(json.dumps({
        "checkpoint_saved": str(args.output),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "decreased": losses[-1] < losses[0],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
