#!/usr/bin/env python3
"""Overfit a few Waymo range-only samples to validate TULIP training."""
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
from tulip_adapter import Waymo32To64Dataset
from tulip_adapter.range_only_tulip import (
    WaymoRangeOnlyTULIP, required_padded_width)
from tulip_adapter.range_only_training import (
    compute_losses, prepare_range_batch, select_training_loss)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--tulip-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-train-samples", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--loss-scope", choices=("all_valid", "generated"),
                   default="all_valid")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def build_model(tulip_root: Path, native_width: int):
    sys.path.insert(0, str(tulip_root.resolve()))
    from model.tulip import tulip_base
    padded_width = required_padded_width(native_width)
    backbone = tulip_base(
        img_size=(32, padded_width), target_img_size=(64, padded_width),
        patch_size=(1, 4), in_chans=1, window_size=(2, 8),
        pixel_shuffle=False, circular_padding=True, swin_v2=False,
        log_transform=False, patch_unmerging=False)
    return WaymoRangeOnlyTULIP(
        backbone, native_width=native_width, padded_width=padded_width)


def save_checkpoint(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    args = parse_args()
    if args.max_train_samples < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("sample, epoch and batch counts must be positive")
    torch.manual_seed(args.seed)
    dataset = Waymo32To64Dataset(
        args.index, dataset_role="train", signal_mode="range_only")
    count = min(args.max_train_samples, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(count)), batch_size=args.batch_size,
        shuffle=True, num_workers=0)
    model = build_model(args.tulip_root, 2650).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    if args.resume:
        if not args.output.is_file():
            raise FileNotFoundError(args.output)
        checkpoint = torch.load(args.output, map_location=args.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1

    print(json.dumps({
        "samples": count, "epochs": args.epochs, "start_epoch": start_epoch,
        "batch_size": args.batch_size, "loss_scope": args.loss_scope,
        "device": args.device, "checkpoint": str(args.output),
    }), flush=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        all_total = generated_total = 0.0
        steps = 0
        for batch in loader:
            input_range, target_range, target_mask, generated_mask = (
                prepare_range_batch(batch, args.device))
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(input_range, target_range, target_mask)
            losses = compute_losses(
                prediction, target_range, target_mask, generated_mask)
            loss = select_training_loss(losses, args.loss_scope)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            all_total += float(losses.all_valid.detach())
            generated_total += float(losses.generated.detach())
            steps += 1
        metrics = {
            "epoch": epoch, "all_valid_l1": all_total / steps,
            "generated_l1": generated_total / steps,
        }
        print(json.dumps(metrics), flush=True)
        save_checkpoint(args.output, {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "args": vars(args),
            "metrics": metrics,
        })


if __name__ == "__main__":
    main()
