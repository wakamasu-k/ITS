#!/usr/bin/env python3
"""Evaluate Waymo 16-to-32 TULIP reconstruction in vehicle-frame 3D."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model
from tools.reconstruct_waymo_16_to_32_pointcloud import reconstruct_sample


METHODS = ("tulip", "nearest", "linear")
METRICS = ("coordinate_mae_m", "euclidean_mae_m", "intensity_mae")


def parse_args() -> argparse.Namespace:
    """Parse multi-frame point-cloud evaluation arguments."""

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


def aggregate_summaries(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate point-cloud metrics weighted by paired point count.

    Input shape: iterable of per-frame metadata dictionaries produced by
    ``reconstruct_sample``.  Output shape: nested scalar summary dictionary.
    Frames without a method result are excluded from that method only.
    """

    row_list = list(rows)
    if not row_list:
        raise ValueError("cannot aggregate an empty point-cloud evaluation")
    result: Dict[str, Any] = {"frames": len(row_list)}
    for method in METHODS:
        total_count = 0
        weighted = {metric: 0.0 for metric in METRICS}
        valid_frames = 0
        for row in row_list:
            metrics = row.get(f"{method}_metrics")
            if not metrics:
                continue
            count = int(metrics["count"])
            total_count += count
            valid_frames += 1
            for metric in METRICS:
                weighted[metric] += float(metrics[metric]) * count
        if total_count == 0:
            result[method] = {"frames": 0, "count": 0}
        else:
            result[method] = {
                "frames": valid_frames,
                "count": total_count,
                **{
                    metric: weighted[metric] / total_count
                    for metric in METRICS
                },
            }
    return result


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load a trusted local checkpoint across PyTorch versions."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    """Evaluate a bounded role subset and save per-frame/aggregate JSON."""

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
        sample_root = args.output_root / f"sample_{index:06d}"
        summary = reconstruct_sample(
            dataset=dataset,
            sample_index=index,
            model=model,
            device=device,
            intensity_transform=args.intensity_transform,
            output_root=sample_root,
        )
        rows.append(summary)
        print(json.dumps({"evaluated": index + 1, "frames": count}), flush=True)

    per_frame_fields = [
        "sample_index", "segment_id", "frame_index", "paired_valid_count",
        "tulip_euclidean_mae_m", "tulip_intensity_mae",
        "nearest_euclidean_mae_m", "nearest_intensity_mae",
        "linear_euclidean_mae_m", "linear_intensity_mae",
    ]
    with (args.output_root / "per_frame.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=per_frame_fields)
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
                    for method in METHODS
                    for metric in ("euclidean_mae_m", "intensity_mae")
                },
            })
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_role": args.dataset_role,
        "evaluated_frames": count,
        "intensity_transform": args.intensity_transform,
        "coordinate_frame": "Waymo vehicle frame",
        "mask_scope": "generated rows with GT-valid and TULIP-positive finite range",
        "aggregate": aggregate_summaries(rows),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
