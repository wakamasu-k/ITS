#!/usr/bin/env python3
"""Run one untrained range-only Waymo batch through the TULIP backbone."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter import Waymo32To64Dataset
from tulip_adapter.range_only_tulip import (
    WaymoRangeOnlyTULIP, required_padded_width,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--dataset-role", default="train",
                        choices=("train", "validation", "test"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    torch.manual_seed(args.seed)
    dataset = Waymo32To64Dataset(
        args.index, dataset_role=args.dataset_role,
        signal_mode="range_only")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    tulip_package = args.tulip_root.resolve()
    sys.path.insert(0, str(tulip_package))
    from model.tulip import tulip_base

    native_width = int(batch["input"].shape[-1])
    padded_width = required_padded_width(native_width)
    backbone = tulip_base(
        img_size=(32, padded_width),
        target_img_size=(64, padded_width),
        patch_size=(1, 4),
        in_chans=1,
        window_size=(2, 8),
        pixel_shuffle=False,
        circular_padding=True,
        swin_v2=False,
        log_transform=False,
        patch_unmerging=False,
    )
    model = WaymoRangeOnlyTULIP(
        backbone, native_width=native_width,
        padded_width=padded_width).to(args.device)
    model.eval()
    input_range = batch["input"].to(args.device)
    target_range = batch["target"].to(args.device)
    target_mask = batch["target_mask"].to(args.device)
    with torch.no_grad():
        prediction, loss = model(input_range, target_range, target_mask)
    print(json.dumps({
        "dataset_role": args.dataset_role,
        "dataset_size": len(dataset),
        "input_shape": list(input_range.shape),
        "padded_width": padded_width,
        "prediction_shape": list(prediction.shape),
        "target_shape": list(target_range.shape),
        "masked_l1_untrained": float(loss),
        "device": args.device,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
