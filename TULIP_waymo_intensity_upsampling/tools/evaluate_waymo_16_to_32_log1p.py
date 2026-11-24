#!/usr/bin/env python3
"""Evaluate a log1p-trained Waymo 16→32 model in raw value space."""
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


def channel_mae(prediction: torch.Tensor, target: torch.Tensor,
                valid_mask: torch.Tensor, channel: int) -> float:
    """Calculate raw-domain channel MAE over valid pixels."""
    mask = valid_mask.to(dtype=prediction.dtype)
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("validation mask contains no valid pixels")
    return float(((prediction[:, channel] - target[:, channel]).abs() * mask).sum() / count)


def parse_args() -> argparse.Namespace:
    """Parse log1p evaluation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-role", default="validation")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Evaluate log1p TULIP after inverse transformation to raw values."""
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dataset = Waymo16To32Dataset(
        args.index, dataset_role=args.dataset_role, intensity_transform="log1p"
    )
    count = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))
    if count < 1:
        raise ValueError("no evaluation samples")
    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    with torch.inference_mode():
        for index in range(count):
            sample = dataset[index]
            input_log = sample["input"].unsqueeze(0).to(device)
            target_log = sample["target"].unsqueeze(0).to(device)
            mask = sample["target_mask"].unsqueeze(0).to(device)
            prediction_log, _ = model(input_log, target_log, mask)

            input_raw = input_log.clone()
            input_raw[:, 1] = torch.expm1(input_raw[:, 1])
            target_raw = target_log.clone()
            target_raw[:, 1] = torch.expm1(target_raw[:, 1])
            prediction_raw = prediction_log.clone()
            prediction_raw[:, 1] = torch.expm1(prediction_raw[:, 1]).clamp_min(0.0)
            nearest = F.interpolate(input_raw, size=(32, 2688), mode="nearest")
            linear = F.interpolate(input_raw, size=(32, 2688), mode="bilinear", align_corners=False)
            prediction_raw = prediction_raw[..., :2650]
            target_raw = target_raw[..., :2650]
            nearest = nearest[..., :2650]
            linear = linear[..., :2650]
            valid_mask = mask[..., :2650]
            if not torch.isfinite(prediction_raw).all():
                raise RuntimeError(f"non-finite prediction at frame {index}")
            records.append({
                "segment_id": sample["segment_id"],
                "frame_index": sample["frame_index"],
                "tulip_range_mae": channel_mae(prediction_raw, target_raw, valid_mask, 0),
                "tulip_intensity_mae": channel_mae(prediction_raw, target_raw, valid_mask, 1),
                "nearest_range_mae": channel_mae(nearest, target_raw, valid_mask, 0),
                "nearest_intensity_mae": channel_mae(nearest, target_raw, valid_mask, 1),
                "linear_range_mae": channel_mae(linear, target_raw, valid_mask, 0),
                "linear_intensity_mae": channel_mae(linear, target_raw, valid_mask, 1),
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
        "intensity_transform": "log1p",
        "metric_value_domain": "raw_range_and_raw_intensity",
        **{name: sum(float(record[name]) for record in records) / count for name in names},
    }
    with (args.output_root / "per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(records[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
