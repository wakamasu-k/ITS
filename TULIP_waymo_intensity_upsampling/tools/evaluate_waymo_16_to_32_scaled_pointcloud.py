#!/usr/bin/env python3
"""Evaluate a scaled Waymo 16-to-32 checkpoint in vehicle-frame 3D."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model
from tulip_adapter.waymo_pointcloud import (
    build_top_geometry,
    paired_vehicle_points,
    pointwise_metrics,
)
from tools.reconstruct_waymo_16_to_32_pointcloud import (
    _read_frame,
    _source_from_pair_json,
    _top_pose,
)


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


def parse_args() -> argparse.Namespace:
    """Parse scaled point-cloud evaluation arguments."""
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


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load a local checkpoint across supported PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def evaluate_one(
    dataset: Waymo16To32Dataset,
    model: torch.nn.Module,
    index: int,
    device: torch.device,
    range_scale_m: float,
    intensity_transform: str,
    output_root: Path,
) -> Dict[str, Any]:
    """Reconstruct one sample and calculate raw-domain point metrics.

    Input shapes: input ``[2,16,2688]`` and target ``[2,32,2688]``.
    Output: JSON-compatible metrics and an NPZ with paired vehicle points.
    """
    sample = dataset[index]
    input_signal = sample["input"].unsqueeze(0).to(device)
    target_signal = sample["target"].unsqueeze(0).to(device)
    target_mask = sample["target_mask"].unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction, _ = model(input_signal, target_signal, target_mask)
        target_raw = restore_domain(
            target_signal,
            range_scale_m=range_scale_m,
            intensity_transform=intensity_transform,
        )
        prediction_raw = restore_domain(
            prediction,
            range_scale_m=range_scale_m,
            intensity_transform=intensity_transform,
        )
        input_raw = restore_domain(
            input_signal,
            range_scale_m=range_scale_m,
            intensity_transform=intensity_transform,
        )
        nearest_raw = F.interpolate(input_raw, size=(32, 2688), mode="nearest")
        linear_raw = F.interpolate(
            input_raw, size=(32, 2688), mode="bilinear", align_corners=False
        )

    target_native = target_raw[..., :2650][0].detach().cpu().numpy()
    prediction_native = prediction_raw[..., :2650][0].detach().cpu().numpy()
    nearest_native = nearest_raw[..., :2650][0].detach().cpu().numpy()
    linear_native = linear_raw[..., :2650][0].detach().cpu().numpy()
    mask_native = target_mask[..., :2650][0].detach().cpu().numpy().astype(bool)
    generated_mask = np.zeros_like(mask_native)
    generated_mask[1::2] = mask_native[1::2]
    target_range, target_intensity = target_native
    prediction_range, prediction_intensity = prediction_native
    nearest_range, nearest_intensity = nearest_native
    linear_range, linear_intensity = linear_native
    common = (
        generated_mask
        & np.isfinite(target_range) & (target_range > 0.0)
        & np.isfinite(prediction_range) & (prediction_range > 0.0)
        & np.isfinite(target_intensity) & np.isfinite(prediction_intensity)
    )
    if not common.any():
        raise ValueError(f"no common valid generated pixels at sample {index}")

    pair_json = Path(dataset.rows[index]["pair_json"])
    source_tfrecord, source_frame_index = _source_from_pair_json(pair_json)
    frame = _read_frame(source_tfrecord, source_frame_index)
    geometry = build_top_geometry(
        frame,
        _top_pose(frame),
        np.arange(0, 64, 2, dtype=np.int32),
        native_width=2650,
    )

    points_gt, intensity_gt = paired_vehicle_points(
        target_range, target_intensity, geometry, common
    )
    points_tulip, intensity_tulip = paired_vehicle_points(
        prediction_range, prediction_intensity, geometry, common
    )
    nearest_mask = common & np.isfinite(nearest_range) & (nearest_range > 0.0)
    linear_mask = common & np.isfinite(linear_range) & (linear_range > 0.0)
    points_nearest, intensity_nearest = paired_vehicle_points(
        nearest_range, nearest_intensity, geometry, nearest_mask
    )
    points_linear, intensity_linear = paired_vehicle_points(
        linear_range, linear_intensity, geometry, linear_mask
    )
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_root / "waymo_16_to_32_scaled_pointcloud.npz",
        points_gt_vehicle=points_gt,
        intensity_gt=intensity_gt,
        points_tulip_vehicle=points_tulip,
        intensity_tulip=intensity_tulip,
    )
    result: Dict[str, Any] = {
        "sample_index": int(index),
        "segment_id": sample["segment_id"],
        "frame_index": int(sample["frame_index"]),
        "source_tfrecord": str(source_tfrecord),
        "source_frame_index": int(source_frame_index),
        "range_scale_m": range_scale_m,
        "intensity_transform": intensity_transform,
        "coordinate_frame": "Waymo vehicle frame",
        "ring_ids": list(range(0, 64, 2)),
        "mask_scope": "generated rows with common GT/predicted finite positive range",
        "paired_valid_count": int(common.sum()),
        "tulip_valid_count": int(points_tulip.shape[0]),
        "nearest_valid_count": int(points_nearest.shape[0]),
        "linear_valid_count": int(points_linear.shape[0]),
        "tulip_metrics": pointwise_metrics(
            points_tulip, points_gt, intensity_tulip, intensity_gt
        ),
        "nearest_metrics": pointwise_metrics(
            points_nearest,
            points_gt[nearest_mask[common]],
            intensity_nearest,
            intensity_gt[nearest_mask[common]],
        ) if points_nearest.shape[0] else None,
        "linear_metrics": pointwise_metrics(
            points_linear,
            points_gt[linear_mask[common]],
            intensity_linear,
            intensity_gt[linear_mask[common]],
        ) if points_linear.shape[0] else None,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate point metrics weighted by paired point count."""
    if not rows:
        raise ValueError("cannot aggregate empty point-cloud evaluation")
    result: Dict[str, Any] = {"frames": len(rows)}
    for method in ("tulip", "nearest", "linear"):
        total = 0
        sums = {name: 0.0 for name in ("coordinate_mae_m", "euclidean_mae_m", "intensity_mae")}
        frames = 0
        for row in rows:
            metrics = row.get(f"{method}_metrics")
            if not metrics:
                continue
            count = int(metrics["count"])
            total += count
            frames += 1
            for name in sums:
                sums[name] += float(metrics[name]) * count
        result[method] = {
            "frames": frames,
            "count": total,
            **({name: sums[name] / total for name in sums} if total else {}),
        }
    return result


def main() -> None:
    """Evaluate a scaled checkpoint and save per-frame/aggregate results."""
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max-frames must be positive")
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
    checkpoint = _load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        row = evaluate_one(
            dataset, model, index, device, args.range_scale_m,
            args.intensity_transform, args.output_root / f"sample_{index:06d}"
        )
        rows.append(row)
        print(json.dumps({"evaluated": index + 1, "frames": count}), flush=True)

    fields = [
        "sample_index", "segment_id", "frame_index", "paired_valid_count",
        "tulip_euclidean_mae_m", "tulip_intensity_mae",
        "nearest_euclidean_mae_m", "nearest_intensity_mae",
        "linear_euclidean_mae_m", "linear_intensity_mae",
    ]
    with (args.output_root / "per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sample_index": row["sample_index"],
                "segment_id": row["segment_id"],
                "frame_index": row["frame_index"],
                "paired_valid_count": row["paired_valid_count"],
                **{
                    f"{method}_{metric}": (
                        row[f"{method}_metrics"][metric]
                        if row.get(f"{method}_metrics") else None
                    )
                    for method in ("tulip", "nearest", "linear")
                    for metric in ("euclidean_mae_m", "intensity_mae")
                },
            })
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_role": args.dataset_role,
        "evaluated_frames": count,
        "range_scale_m": args.range_scale_m,
        "intensity_transform": args.intensity_transform,
        "coordinate_frame": "Waymo vehicle frame",
        "metric_value_domain": "raw_range_and_raw_intensity",
        "aggregate": _aggregate(rows),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
