#!/usr/bin/env python3
"""Evaluate Waymo 16→32 TULIP against nearest and linear baselines."""
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


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-role", default="validation")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def channel_mae(prediction: torch.Tensor, target: torch.Tensor,
                valid_mask: torch.Tensor, channel: int) -> float:
    """Calculate one-channel MAE over valid native-width pixels."""
    mask = valid_mask.to(dtype=prediction.dtype)
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("validation mask contains no valid pixels")
    return float(((prediction[:, channel] - target[:, channel]).abs() * mask).sum() / count)


def crop_native(tensor: torch.Tensor) -> torch.Tensor:
    """Crop a padded tensor from width 2688 to native width 2650."""
    return tensor[..., :2650]


def main() -> None:
    """Evaluate TULIP and interpolation baselines on one dataset role."""
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max_frames must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dataset = Waymo16To32Dataset(args.index, dataset_role=args.dataset_role)
    if args.max_frames is not None:
        count = min(args.max_frames, len(dataset))
    else:
        count = len(dataset)
    if count == 0:
        raise ValueError("no evaluation samples")

    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    per_frame: list[dict[str, object]] = []

    with torch.inference_mode():
        for index in range(count):
            sample = dataset[index]
            input_signal = sample["input"].unsqueeze(0).to(device)
            target_signal = sample["target"].unsqueeze(0).to(device)
            target_mask = sample["target_mask"].unsqueeze(0).to(device)
            prediction, _ = model(input_signal, target_signal, target_mask)
            nearest = F.interpolate(input_signal, size=(32, 2688), mode="nearest")
            linear = F.interpolate(
                input_signal, size=(32, 2688), mode="bilinear", align_corners=False
            )
            prediction = crop_native(prediction)
            nearest = crop_native(nearest)
            linear = crop_native(linear)
            target = crop_native(target_signal)
            mask = crop_native(target_mask)
            if not torch.isfinite(prediction).all():
                raise RuntimeError(f"non-finite prediction at frame {index}")
            per_frame.append({
                "segment_id": sample["segment_id"],
                "frame_index": sample["frame_index"],
                "tulip_range_mae": channel_mae(prediction, target, mask, 0),
                "tulip_intensity_mae": channel_mae(prediction, target, mask, 1),
                "nearest_range_mae": channel_mae(nearest, target, mask, 0),
                "nearest_intensity_mae": channel_mae(nearest, target, mask, 1),
                "linear_range_mae": channel_mae(linear, target, mask, 0),
                "linear_intensity_mae": channel_mae(linear, target, mask, 1),
                "valid_pixels": int(mask.sum()),
            })
            print(json.dumps({"evaluated": index + 1, "frames": count}), flush=True)

    metric_names = (
        "tulip_range_mae", "tulip_intensity_mae",
        "nearest_range_mae", "nearest_intensity_mae",
        "linear_range_mae", "linear_intensity_mae",
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_role": args.dataset_role,
        "evaluated_frames": count,
        "metric_value_domain": "raw_range_and_raw_intensity",
        **{name: sum(float(row[name]) for row in per_frame) / count for name in metric_names},
    }
    with (args.output_root / "per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(per_frame[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_frame)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
