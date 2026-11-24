#!/usr/bin/env python3
"""Evaluate a scaled Waymo 16-to-32 checkpoint in raw value space."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model


def restore_domain(
    tensor: torch.Tensor,
    *,
    range_scale_m: float,
    intensity_transform: str,
) -> torch.Tensor:
    """Restore ``[B,2,H,W]`` model-domain values to raw units."""
    if tensor.ndim != 4 or tensor.shape[1] != 2:
        raise ValueError("tensor must be [B,2,H,W]")
    if range_scale_m <= 0:
        raise ValueError("range_scale_m must be positive")
    output = tensor.clone()
    output[:, 0] *= range_scale_m
    if intensity_transform == "log1p":
        output[:, 1] = torch.expm1(output[:, 1]).clamp_min(0.0)
    elif intensity_transform != "raw":
        raise ValueError(f"unsupported intensity_transform: {intensity_transform}")
    return output


def channel_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    channel: int,
) -> float:
    """Calculate one-channel raw-domain MAE."""
    valid = mask.to(dtype=prediction.dtype)
    count = valid.sum()
    if count.item() == 0:
        raise ValueError("evaluation mask contains no valid pixels")
    return float(((prediction[:, channel] - target[:, channel]).abs() * valid).sum() / count)


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-role", default="validation")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--range-scale-m", type=float, default=75.0)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"), default="log1p")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Evaluate TULIP and interpolation baselines in raw units."""
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max_frames must be positive")
    if args.range_scale_m <= 0:
        raise ValueError("range-scale-m must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dataset = Waymo16To32Dataset(
        args.index,
        dataset_role=args.dataset_role,
        intensity_transform=args.intensity_transform,
        range_scale_m=args.range_scale_m,
    )
    count = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))
    if count < 1:
        raise ValueError("no evaluation samples")
    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    with torch.inference_mode():
        for index in range(count):
            sample = dataset[index]
            input_signal = sample["input"].unsqueeze(0).to(device)
            target_signal = sample["target"].unsqueeze(0).to(device)
            target_mask = sample["target_mask"].unsqueeze(0).to(device)
            prediction, _ = model(input_signal, target_signal, target_mask)
            prediction = restore_domain(
                prediction,
                range_scale_m=args.range_scale_m,
                intensity_transform=args.intensity_transform,
            )
            target = restore_domain(
                target_signal,
                range_scale_m=args.range_scale_m,
                intensity_transform=args.intensity_transform,
            )
            input_raw = restore_domain(
                input_signal,
                range_scale_m=args.range_scale_m,
                intensity_transform=args.intensity_transform,
            )
            nearest = F.interpolate(input_raw, size=(32, 2688), mode="nearest")
            linear = F.interpolate(input_raw, size=(32, 2688), mode="bilinear", align_corners=False)
            prediction = prediction[..., :2650]
            target = target[..., :2650]
            nearest = nearest[..., :2650]
            linear = linear[..., :2650]
            mask = target_mask[..., :2650]
            if not torch.isfinite(prediction).all():
                raise RuntimeError(f"non-finite prediction at frame {index}")
            records.append({
                "segment_id": sample["segment_id"],
                "frame_index": sample["frame_index"],
                "tulip_range_mae": channel_mae(prediction, target, mask, 0),
                "tulip_intensity_mae": channel_mae(prediction, target, mask, 1),
                "nearest_range_mae": channel_mae(nearest, target, mask, 0),
                "nearest_intensity_mae": channel_mae(nearest, target, mask, 1),
                "linear_range_mae": channel_mae(linear, target, mask, 0),
                "linear_intensity_mae": channel_mae(linear, target, mask, 1),
            })
            print(json.dumps({"evaluated": index + 1, "frames": count}), flush=True)

    names = (
        "tulip_range_mae", "tulip_intensity_mae",
        "nearest_range_mae", "nearest_intensity_mae",
        "linear_range_mae", "linear_intensity_mae",
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_role": args.dataset_role,
        "evaluated_frames": count,
        "range_scale_m": args.range_scale_m,
        "intensity_transform": args.intensity_transform,
        "metric_value_domain": "raw_range_and_raw_intensity",
        **{name: sum(float(row[name]) for row in records) / count for name in names},
    }
    with (args.output_root / "per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
