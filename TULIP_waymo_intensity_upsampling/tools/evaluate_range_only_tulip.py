#!/usr/bin/env python3
"""Evaluate a fixed TULIP checkpoint against interpolation baselines."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tulip_adapter import Waymo32To64Dataset
from tulip_adapter.range_only_evaluation import (interpolate_32_to_64,
    masked_metrics, prediction_statistics, save_difference_image)
from tulip_adapter.range_only_training import (normalize_ranges,
    prepare_range_batch, restore_range_meters, validate_range_scale)
from tulip_adapter.range_only_tulip import WaymoRangeOnlyTULIP, required_padded_width

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--tulip-root", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--dataset-role", choices=("train", "validation", "test"),
                   default="validation")
    p.add_argument("--max-frames", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-images", action="store_true")
    return p.parse_args()

def build_model(tulip_root, native_width=2650):
    sys.path.insert(0, str(tulip_root.resolve()))
    from model.tulip import tulip_base
    padded = required_padded_width(native_width)
    backbone = tulip_base(img_size=(32, padded), target_img_size=(64, padded),
        patch_size=(1, 4), in_chans=1, window_size=(2, 8),
        pixel_shuffle=False, circular_padding=True, swin_v2=False,
        log_transform=False, patch_unmerging=False)
    return WaymoRangeOnlyTULIP(backbone, native_width=native_width,
                               padded_width=padded)

def main():
    args = parse_args()
    if args.max_frames < 1: raise ValueError("max-frames must be positive")
    dataset = Waymo32To64Dataset(args.index, dataset_role=args.dataset_role,
                                 signal_mode="range_only")
    count = min(args.max_frames, len(dataset))
    model = build_model(args.tulip_root).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    range_scale = validate_range_scale(
        checkpoint.get("args", {}).get("range_scale", 1.0))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for number in range(count):
            sample = dataset[number]
            batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else value
                     for key, value in sample.items()}
            input_range, target, target_mask, generated_mask = \
                prepare_range_batch(batch, args.device)
            input_mask = batch["input_mask"].to(args.device).bool()
            normalized_input, normalized_target = normalize_ranges(
                input_range, target, range_scale)
            normalized_prediction, _ = model(
                normalized_input, normalized_target, target_mask)
            prediction = restore_range_meters(
                normalized_prediction, range_scale)
            nearest = interpolate_32_to_64(input_range, input_mask, mode="nearest")
            linear = interpolate_32_to_64(input_range, input_mask, mode="linear")
            tm = masked_metrics(prediction, target, target_mask, generated_mask)
            nm = masked_metrics(nearest, target, target_mask, generated_mask)
            lm = masked_metrics(linear, target, target_mask, generated_mask)
            image = args.output_root / "images" / f"frame_{number:04d}.png"
            if not args.no_images:
                save_difference_image(image, target, prediction, linear,
                                      target_mask, generated_mask)
            rows.append({"dataset_role": sample["dataset_role"],
                "segment_id": sample["segment_id"],
                "frame_index": sample["frame_index"],
                "timestamp_micros": sample["timestamp_micros"],
                "tulip_all_valid_l1_m": tm.all_valid_l1,
                "tulip_generated_l1_m": tm.generated_l1,
                "nearest_all_valid_l1_m": nm.all_valid_l1,
                "nearest_generated_l1_m": nm.generated_l1,
                "linear_all_valid_l1_m": lm.all_valid_l1,
                "linear_generated_l1_m": lm.generated_l1,
                **prediction_statistics(prediction, target_mask),
                "difference_image": "" if args.no_images else str(image)})
            print(json.dumps({"evaluated": number + 1, "frames": count}), flush=True)
    csv_path = args.output_root / "per_frame.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    mean = lambda key: sum(float(row[key]) for row in rows) / len(rows)
    linear_mean = mean("linear_generated_l1_m")
    summary = {"checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "range_scale_m": range_scale,
        "dataset_role": args.dataset_role, "evaluated_frames": len(rows),
        "model_training": model.training,
        "mean_tulip_generated_l1_m": mean("tulip_generated_l1_m"),
        "mean_nearest_generated_l1_m": mean("nearest_generated_l1_m"),
        "mean_linear_generated_l1_m": linear_mean,
        "tulip_to_linear_generated_l1_ratio":
            mean("tulip_generated_l1_m") / linear_mean if linear_mean > 0 else None,
        "total_negative_predictions": sum(
            int(row["prediction_valid_negative_count"]) for row in rows),
        "per_frame_csv": str(csv_path)}
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
