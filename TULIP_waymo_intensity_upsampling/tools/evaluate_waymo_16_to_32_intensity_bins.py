#!/usr/bin/env python3
"""Evaluate intensity MAE by target-intensity value bins."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_waymo_16_to_32_generated_only import (
    raw_signals,
    resolve_range_scale_m,
)
from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model


BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0),
        (5.0, 20.0), (20.0, 100.0), (100.0, 1000.0), (1000.0, float("inf")))


def parse_args() -> argparse.Namespace:
    """Parse binned evaluation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"), default="raw")
    parser.add_argument("--dataset-role", default="validation")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--range-scale-m",
        type=float,
        help=(
            "Range normalization used during training. By default, read it "
            "from the checkpoint."
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Run raw-domain binned intensity evaluation."""
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    range_scale_m = resolve_range_scale_m(checkpoint, args.range_scale_m)
    dataset = Waymo16To32Dataset(
        args.index,
        dataset_role=args.dataset_role,
        intensity_transform=args.intensity_transform,
        range_scale_m=range_scale_m,
    )
    count = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))
    if count < 1:
        raise ValueError("no evaluation samples")
    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    accum = {
        f"{low:g}_{high:g}": {"count": 0, "target_sum": 0.0,
                              "tulip_abs_sum": 0.0, "nearest_abs_sum": 0.0,
                              "linear_abs_sum": 0.0}
        for low, high in BINS
    }

    with torch.inference_mode():
        for index in range(count):
            sample = dataset[index]
            x = sample["input"].unsqueeze(0).to(device)
            y = sample["target"].unsqueeze(0).to(device)
            mask = sample["target_mask"].unsqueeze(0).to(device)
            prediction, _ = model(x, y, mask)
            x_raw, y_raw, pred_raw = raw_signals(
                x, y, prediction, args.intensity_transform, range_scale_m
            )
            x_raw = x_raw[..., :2650]
            y_raw = y_raw[..., :2650]
            pred_raw = pred_raw[..., :2650]
            valid = mask[..., :2650].bool()
            nearest = F.interpolate(x_raw, size=(32, 2650), mode="nearest")
            linear = F.interpolate(x_raw, size=(32, 2650), mode="bilinear", align_corners=False)
            target = y_raw[0, 1][valid[0]].detach().cpu().numpy()
            tulip = pred_raw[0, 1][valid[0]].detach().cpu().numpy()
            nearest_values = nearest[0, 1][valid[0]].detach().cpu().numpy()
            linear_values = linear[0, 1][valid[0]].detach().cpu().numpy()
            for low, high in BINS:
                key = f"{low:g}_{high:g}"
                selected = (target >= low) & (target < high)
                if not np.any(selected):
                    continue
                item = accum[key]
                item["count"] += int(selected.sum())
                item["target_sum"] += float(target[selected].sum())
                item["tulip_abs_sum"] += float(np.abs(tulip[selected] - target[selected]).sum())
                item["nearest_abs_sum"] += float(np.abs(nearest_values[selected] - target[selected]).sum())
                item["linear_abs_sum"] += float(np.abs(linear_values[selected] - target[selected]).sum())
            print(json.dumps({"evaluated": index + 1, "frames": count}), flush=True)

    bins = {}
    total_count = sum(v["count"] for v in accum.values())
    for key, item in accum.items():
        count_value = item["count"]
        if count_value:
            bins[key] = {
                "count": count_value,
                "fraction": count_value / total_count,
                "target_mean": item["target_sum"] / count_value,
                "tulip_mae": item["tulip_abs_sum"] / count_value,
                "nearest_mae": item["nearest_abs_sum"] / count_value,
                "linear_mae": item["linear_abs_sum"] / count_value,
            }
        else:
            bins[key] = {"count": 0, "fraction": 0.0}
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_role": args.dataset_role,
        "evaluated_frames": count,
        "range_scale_m": range_scale_m,
        "intensity_transform": args.intensity_transform,
        "metric_value_domain": "raw_intensity",
        "mask_scope": "all_valid_target_rows",
        "bins": bins,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
