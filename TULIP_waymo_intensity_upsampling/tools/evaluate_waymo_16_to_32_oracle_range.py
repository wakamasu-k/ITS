#!/usr/bin/env python3
"""Separate TULIP intensity error from predicted-range geometry error."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

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
    _raw_signals,
    _read_frame,
    _source_from_pair_json,
    _top_pose,
)


def parse_args() -> argparse.Namespace:
    """Parse oracle-range diagnostic arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-role", default="validation")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"), default="raw")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load a local checkpoint across PyTorch versions."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def evaluate_one(
    dataset: Waymo16To32Dataset,
    model: torch.nn.Module,
    index: int,
    device: torch.device,
    intensity_transform: str,
) -> Dict[str, Any]:
    """Evaluate one sample with GT and predicted range geometry.

    Input shapes: input ``[2,16,2688]`` and target ``[2,32,2688]``.
    Output: JSON-compatible per-frame metrics.  Both modes use the same
    generated-row pixels; only the range used for 3D conversion differs.
    """

    sample = dataset[index]
    x = sample["input"].unsqueeze(0).to(device)
    y = sample["target"].unsqueeze(0).to(device)
    target_mask = sample["target_mask"].unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction, _ = model(x, y, target_mask)
        _, target_raw, prediction_raw = _raw_signals(
            x, y, prediction, intensity_transform
        )
    target_native = target_raw[..., :2650][0].detach().cpu().numpy()
    prediction_native = prediction_raw[..., :2650][0].detach().cpu().numpy()
    mask = target_mask[..., :2650][0].detach().cpu().numpy().astype(bool)
    generated = np.zeros_like(mask)
    generated[1::2] = mask[1::2]
    target_range = target_native[0]
    target_intensity = target_native[1]
    predicted_range = prediction_native[0]
    predicted_intensity = prediction_native[1]
    common = (
        generated
        & np.isfinite(target_range)
        & (target_range > 0.0)
        & np.isfinite(predicted_range)
        & (predicted_range > 0.0)
        & np.isfinite(target_intensity)
        & np.isfinite(predicted_intensity)
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
    gt_points, gt_intensity = paired_vehicle_points(
        target_range, target_intensity, geometry, common
    )
    oracle_points, oracle_intensity = paired_vehicle_points(
        target_range, predicted_intensity, geometry, common
    )
    predicted_points, predicted_intensity_values = paired_vehicle_points(
        predicted_range, predicted_intensity, geometry, common
    )
    return {
        "sample_index": int(index),
        "segment_id": sample["segment_id"],
        "frame_index": int(sample["frame_index"]),
        "paired_valid_count": int(common.sum()),
        "oracle_range_metrics": pointwise_metrics(
            oracle_points, gt_points, oracle_intensity, gt_intensity
        ),
        "predicted_range_metrics": pointwise_metrics(
            predicted_points, gt_points, predicted_intensity_values, gt_intensity
        ),
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate oracle/predicted-range metrics weighted by valid pixels."""

    if not rows:
        raise ValueError("cannot aggregate empty diagnostics")
    result: Dict[str, Any] = {"frames": len(rows)}
    for mode in ("oracle_range", "predicted_range"):
        total = 0
        sums = {name: 0.0 for name in ("coordinate_mae_m", "euclidean_mae_m", "intensity_mae")}
        for row in rows:
            metrics = row[f"{mode}_metrics"]
            count = int(metrics["count"])
            total += count
            for name in sums:
                sums[name] += float(metrics[name]) * count
        result[mode] = {
            "count": total,
            **{name: sums[name] / total for name in sums},
        }
    result["range_geometry_penalty_euclidean_mae_m"] = (
        result["predicted_range"]["euclidean_mae_m"]
        - result["oracle_range"]["euclidean_mae_m"]
    )
    return result


def main() -> None:
    """Run the range-isolation diagnostic and save JSON/CSV outputs."""

    args = parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max_frames must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dataset = Waymo16To32Dataset(
        args.index,
        dataset_role=args.dataset_role,
        intensity_transform=args.intensity_transform,
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
        row = evaluate_one(dataset, model, index, device, args.intensity_transform)
        rows.append(row)
        print(json.dumps({"evaluated": index + 1, "frames": count}), flush=True)

    with (args.output_root / "per_frame.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "sample_index", "segment_id", "frame_index", "paired_valid_count",
            "oracle_range_euclidean_mae_m", "oracle_range_intensity_mae",
            "predicted_range_euclidean_mae_m", "predicted_range_intensity_mae",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sample_index": row["sample_index"],
                "segment_id": row["segment_id"],
                "frame_index": row["frame_index"],
                "paired_valid_count": row["paired_valid_count"],
                "oracle_range_euclidean_mae_m": row["oracle_range_metrics"]["euclidean_mae_m"],
                "oracle_range_intensity_mae": row["oracle_range_metrics"]["intensity_mae"],
                "predicted_range_euclidean_mae_m": row["predicted_range_metrics"]["euclidean_mae_m"],
                "predicted_range_intensity_mae": row["predicted_range_metrics"]["intensity_mae"],
            })
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_role": args.dataset_role,
        "evaluated_frames": count,
        "intensity_transform": args.intensity_transform,
        "mask_scope": "generated rows with common GT/predicted finite positive range",
        "interpretation": {
            "oracle_range": "GT range + TULIP intensity",
            "predicted_range": "TULIP range + TULIP intensity",
        },
        "aggregate": aggregate(rows),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
