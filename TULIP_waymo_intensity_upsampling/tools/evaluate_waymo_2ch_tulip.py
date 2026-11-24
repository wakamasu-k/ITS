#!/usr/bin/env python3
"""Evaluate a Waymo 2-channel checkpoint in the raw value domain."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_32_to_64_dataset import Waymo32To64Dataset
from tulip_adapter.waymo_32_to_64_tulip import Waymo32To64TULIP
from tulip_adapter.range_only_tulip import required_padded_width
from tulip_adapter.waymo_2ch_training import (
    channel_prediction_statistics,
    interpolate_32_to_64_2ch,
    normalize_two_channel,
    prepare_two_channel_batch,
    restore_two_channel,
    two_channel_metrics,
    validate_range_scale,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for two-channel checkpoint evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-role", choices=("train", "validation", "test"),
                        default="validation")
    parser.add_argument("--max-frames", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"),
                        default=None)
    return parser.parse_args()


def build_model(tulip_root: Path, native_width: int = 2650) -> Waymo32To64TULIP:
    """Build the same new 2-channel architecture used during training."""
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


def restore_intensity_values(
    signal: torch.Tensor, intensity_transform: str
) -> torch.Tensor:
    """Convert Dataset intensity values back to raw intensity values.

    Args:
        signal: Tensor ``[B,2,H,W]`` whose channel 1 may be log1p-transformed.
        intensity_transform: ``raw`` or ``log1p``.

    Returns:
        A new tensor in the raw intensity value domain.
    """
    if signal.ndim != 4 or signal.shape[1] != 2:
        raise ValueError("signal must be [B,2,H,W]")
    if intensity_transform == "raw":
        return signal
    if intensity_transform != "log1p":
        raise ValueError(
            f"unsupported intensity_transform: {intensity_transform}"
        )
    restored = signal.clone()
    restored[:, 1] = torch.expm1(restored[:, 1])
    if not torch.isfinite(restored).all():
        raise ValueError("inverse log1p produced non-finite intensity values")
    return restored


def main() -> None:
    """Evaluate TULIP and interpolation baselines in raw value units."""
    args = parse_args()
    if args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    saved_args = checkpoint.get("args", {})
    range_scale = validate_range_scale(saved_args.get("range_scale", 1.0))
    intensity_transform = args.intensity_transform or saved_args.get(
        "intensity_transform", "raw"
    )
    dataset = Waymo32To64Dataset(
        args.index, dataset_role=args.dataset_role,
        signal_mode="range_intensity", intensity_transform=intensity_transform,
    )
    count = min(args.max_frames, len(dataset))
    model = build_model(args.tulip_root).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for number in range(count):
            sample = dataset[number]
            batch = {
                key: value.unsqueeze(0) if torch.is_tensor(value) else value
                for key, value in sample.items()
            }
            input_signal, target_signal, target_mask, generated_mask, input_mask = (
                prepare_two_channel_batch(batch, args.device)
            )
            normalized_input, normalized_target = normalize_two_channel(
                input_signal, target_signal, range_scale
            )
            normalized_prediction, _ = model(
                normalized_input, normalized_target, target_mask
            )
            prediction = restore_intensity_values(
                restore_two_channel(normalized_prediction, range_scale),
                intensity_transform,
            )
            target_values = restore_intensity_values(
                target_signal, intensity_transform
            )
            input_values = restore_intensity_values(
                input_signal, intensity_transform
            )
            nearest = interpolate_32_to_64_2ch(
                input_values, input_mask, mode="nearest"
            )
            linear = interpolate_32_to_64_2ch(
                input_values, input_mask, mode="linear"
            )
            tulip_metrics = two_channel_metrics(
                prediction, target_values, target_mask, generated_mask
            )
            nearest_metrics = two_channel_metrics(
                nearest, target_values, target_mask, generated_mask
            )
            linear_metrics = two_channel_metrics(
                linear, target_values, target_mask, generated_mask
            )
            row = {
                "dataset_role": sample["dataset_role"],
                "segment_id": sample["segment_id"],
                "frame_index": sample["frame_index"],
                "timestamp_micros": sample["timestamp_micros"],
                "tulip_all_valid_range_l1_m": tulip_metrics.all_valid_range,
                "tulip_generated_range_l1_m": tulip_metrics.generated_range,
                "tulip_all_valid_intensity_l1": tulip_metrics.all_valid_intensity,
                "tulip_generated_intensity_l1": tulip_metrics.generated_intensity,
                "nearest_generated_range_l1_m": nearest_metrics.generated_range,
                "nearest_generated_intensity_l1": nearest_metrics.generated_intensity,
                "linear_generated_range_l1_m": linear_metrics.generated_range,
                "linear_generated_intensity_l1": linear_metrics.generated_intensity,
                **channel_prediction_statistics(prediction, target_mask),
            }
            rows.append(row)
            print(json.dumps({"evaluated": number + 1, "frames": count}), flush=True)
    csv_path = args.output_root / "per_frame.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "range_scale_m": range_scale,
        "intensity_transform": intensity_transform,
        "metric_value_domain": "raw_range_and_raw_intensity",
        "dataset_role": args.dataset_role,
        "evaluated_frames": len(rows),
        "model_training": model.training,
        "mean_tulip_generated_range_l1_m": mean("tulip_generated_range_l1_m"),
        "mean_tulip_generated_intensity_l1": mean("tulip_generated_intensity_l1"),
        "mean_nearest_generated_range_l1_m": mean("nearest_generated_range_l1_m"),
        "mean_nearest_generated_intensity_l1": mean("nearest_generated_intensity_l1"),
        "mean_linear_generated_range_l1_m": mean("linear_generated_range_l1_m"),
        "mean_linear_generated_intensity_l1": mean("linear_generated_intensity_l1"),
        "per_frame_csv": str(csv_path),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
