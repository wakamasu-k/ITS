#!/usr/bin/env python3
"""Train a 2-channel experiment with a configurable intensity-priority loss."""
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

from tulip_adapter.range_only_tulip import required_padded_width
from tulip_adapter.waymo_32_to_64_dataset import Waymo32To64Dataset
from tulip_adapter.waymo_32_to_64_tulip import Waymo32To64TULIP
from tulip_adapter.waymo_2ch_training import (
    normalize_two_channel,
    prepare_two_channel_batch,
    select_two_channel_loss,
    validate_range_scale,
)
from tulip_adapter.waymo_2ch_weighted_training import (
    compute_weighted_two_channel_losses,
)


def parse_args() -> argparse.Namespace:
    """Parse the intensity-priority experiment arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-train-samples", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--loss-scope", choices=("all_valid", "generated"),
                        default="all_valid")
    parser.add_argument("--range-loss-weight", type=float, default=0.05)
    parser.add_argument("--intensity-loss-weight", type=float, default=1.0)
    parser.add_argument("--range-scale", type=float, default=1.0)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"),
                        default="raw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(tulip_root: Path, native_width: int = 2650) -> Waymo32To64TULIP:
    """Build the unchanged TULIP backbone through the Waymo 2ch adapter."""
    sys.path.insert(0, str(tulip_root.resolve()))
    from model.tulip import tulip_base

    padded_width = required_padded_width(native_width)
    backbone = tulip_base(
        img_size=(32, padded_width), target_img_size=(64, padded_width),
        patch_size=(1, 4), in_chans=2, window_size=(2, 8),
        pixel_shuffle=False, circular_padding=True, swin_v2=False,
        log_transform=False, patch_unmerging=False,
    )
    return Waymo32To64TULIP(
        backbone, native_width=native_width, padded_width=padded_width
    )


def save_checkpoint(path: Path, value: dict) -> None:
    """Save a checkpoint with an atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    """Run the configurable channel-weight experiment."""
    args = parse_args()
    if args.max_train_samples < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("sample, epoch and batch counts must be positive")
    range_scale = validate_range_scale(args.range_scale)
    torch.manual_seed(args.seed)
    dataset = Waymo32To64Dataset(
        args.index, dataset_role="train", signal_mode="range_intensity",
        intensity_transform=args.intensity_transform,
    )
    count = min(args.max_train_samples, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(count)), batch_size=args.batch_size,
        shuffle=True, num_workers=0,
    )
    model = build_model(args.tulip_root).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    print(json.dumps({
        "samples": count, "epochs": args.epochs,
        "range_loss_weight": args.range_loss_weight,
        "intensity_loss_weight": args.intensity_loss_weight,
        "range_scale": range_scale,
        "intensity_transform": args.intensity_transform,
        "device": args.device, "checkpoint": str(args.output),
    }), flush=True)
    for epoch in range(args.epochs):
        model.train()
        all_total = generated_total = 0.0
        steps = 0
        for batch in loader:
            input_signal, target_signal, target_mask, generated_mask, _ = (
                prepare_two_channel_batch(batch, args.device)
            )
            normalized_input, normalized_target = normalize_two_channel(
                input_signal, target_signal, range_scale
            )
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(
                normalized_input, normalized_target, target_mask
            )
            losses = compute_weighted_two_channel_losses(
                prediction, normalized_target, target_mask, generated_mask,
                range_weight=args.range_loss_weight,
                intensity_weight=args.intensity_loss_weight,
            )
            loss = select_two_channel_loss(losses, args.loss_scope)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            all_total += float(losses.all_valid.detach())
            generated_total += float(losses.generated.detach())
            steps += 1
        metrics = {
            "epoch": epoch,
            "all_valid_weighted_l1": all_total / steps,
            "generated_weighted_l1": generated_total / steps,
        }
        print(json.dumps(metrics), flush=True)
        save_checkpoint(args.output, {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "args": vars(args),
            "metrics": metrics,
        })


if __name__ == "__main__":
    main()
