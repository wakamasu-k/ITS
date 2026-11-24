#!/usr/bin/env python3
"""Diagnose raw intensity prediction distributions for Waymo 16→32."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model


def stats(values: np.ndarray) -> dict[str, float | int]:
    """Return finite-value count and distribution statistics."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite values for statistics")
    percentiles = np.percentile(values, [50, 95, 99, 99.5, 99.9, 99.99])
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(percentiles[0]),
        "p95": float(percentiles[1]),
        "p99": float(percentiles[2]),
        "p99_5": float(percentiles[3]),
        "p99_9": float(percentiles[4]),
        "p99_99": float(percentiles[5]),
    }


def parse_args() -> argparse.Namespace:
    """Parse diagnostic arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--dataset-role", default="validation")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Collect target/prediction intensity distributions and correlations."""
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dataset = Waymo16To32Dataset(args.index, dataset_role=args.dataset_role)
    count = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))
    if count < 1:
        raise ValueError("no diagnostic samples")
    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    target_values: list[np.ndarray] = []
    prediction_values: list[np.ndarray] = []
    linear_values: list[np.ndarray] = []
    frame_mae: list[float] = []
    negative_prediction = 0
    total_prediction = 0

    with torch.inference_mode():
        for index in range(count):
            sample = dataset[index]
            x = sample["input"].unsqueeze(0).to(device)
            y = sample["target"].unsqueeze(0).to(device)
            mask = sample["target_mask"].unsqueeze(0).to(device)
            prediction, _ = model(x, y, mask)
            prediction = prediction[..., :2650]
            y = y[..., :2650]
            mask = mask[..., :2650]
            valid = mask[0].bool()
            pred_intensity = prediction[0, 1][valid].detach().cpu().numpy()
            target_intensity = y[0, 1][valid].detach().cpu().numpy()
            target_values.append(target_intensity)
            prediction_values.append(pred_intensity)
            frame_mae.append(float(np.abs(pred_intensity - target_intensity).mean()))
            negative_prediction += int((pred_intensity < 0).sum())
            total_prediction += int(pred_intensity.size)
            print(json.dumps({"diagnosed": index + 1, "frames": count}), flush=True)

    target_flat = np.concatenate(target_values)
    prediction_flat = np.concatenate(prediction_values)
    if target_flat.size != prediction_flat.size:
        raise RuntimeError("target and prediction sample counts differ")
    if target_flat.std() > 0 and prediction_flat.std() > 0:
        correlation = float(np.corrcoef(target_flat, prediction_flat)[0, 1])
    else:
        correlation = None
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_role": args.dataset_role,
        "evaluated_frames": count,
        "target_intensity": stats(target_flat),
        "prediction_intensity": stats(prediction_flat),
        "prediction_negative_fraction": negative_prediction / total_prediction,
        "prediction_target_correlation": correlation,
        "frame_mae_min": float(min(frame_mae)),
        "frame_mae_max": float(max(frame_mae)),
        "frame_mae_mean": float(np.mean(frame_mae)),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
