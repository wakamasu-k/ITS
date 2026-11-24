#!/usr/bin/env python3
"""Fair, generated-row-only 3-D comparison of TULIP, nearest, and linear.

Unlike the older point-cloud evaluator, this script separates each method's
validity/coverage from a *shared* GT-valid mask.  The reported shared metrics
therefore compare TULIP, nearest, and linear on exactly the same point pixels.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_waymo_16_to_32_scaled_pointcloud import restore_domain
from tools.reconstruct_waymo_16_to_32_pointcloud import _read_frame, _source_from_pair_json, _top_pose
from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model
from tulip_adapter.waymo_pointcloud import build_top_geometry, paired_vehicle_points, pointwise_metrics


METHODS = ("tulip", "nearest", "linear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-role", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--range-scale-m", type=float, default=75.0)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"), default="log1p")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def valid_signal_mask(signal: np.ndarray, target_domain: np.ndarray) -> np.ndarray:
    """Return finite positive range and finite intensity on a supplied GT domain."""
    values = np.asarray(signal, dtype=np.float32)
    if values.shape[0] != 2:
        raise ValueError("signal must be [2,H,W]")
    domain = np.asarray(target_domain, dtype=bool)
    if values.shape[1:] != domain.shape:
        raise ValueError("signal and target domain shape mismatch")
    return domain & np.isfinite(values[0]) & (values[0] > 0.0) & np.isfinite(values[1])


def shared_valid_mask(target_domain: np.ndarray, method_signals: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return per-method masks and their common fixed comparison mask."""
    masks = {name: valid_signal_mask(signal, target_domain) for name, signal in method_signals.items()}
    shared = np.asarray(target_domain, dtype=bool).copy()
    for mask in masks.values():
        shared &= mask
    return masks, shared


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metrics_for_mask(
    signal: np.ndarray, target: np.ndarray, geometry: Any, mask: np.ndarray
) -> dict[str, float | int] | None:
    if not mask.any():
        return None
    points_gt, intensity_gt = paired_vehicle_points(target[0], target[1], geometry, mask)
    points_pred, intensity_pred = paired_vehicle_points(signal[0], signal[1], geometry, mask)
    return pointwise_metrics(points_pred, points_gt, intensity_pred, intensity_gt)


def evaluate_one(
    dataset: Waymo16To32Dataset, model: torch.nn.Module, sample_index: int,
    device: torch.device, range_scale_m: float, intensity_transform: str,
) -> dict[str, Any]:
    sample = dataset[sample_index]
    input_signal = sample["input"].unsqueeze(0).to(device)
    target_signal = sample["target"].unsqueeze(0).to(device)
    target_mask = sample["target_mask"].unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction, _ = model(input_signal, target_signal, target_mask)
        target_raw = restore_domain(target_signal, range_scale_m=range_scale_m, intensity_transform=intensity_transform)
        tulip_raw = restore_domain(prediction, range_scale_m=range_scale_m, intensity_transform=intensity_transform)
        input_raw = restore_domain(input_signal, range_scale_m=range_scale_m, intensity_transform=intensity_transform)
        nearest_raw = F.interpolate(input_raw, size=(32, 2688), mode="nearest")
        linear_raw = F.interpolate(input_raw, size=(32, 2688), mode="bilinear", align_corners=False)

    target = target_raw[..., :2650][0].cpu().numpy()
    signals = {
        "tulip": tulip_raw[..., :2650][0].cpu().numpy(),
        "nearest": nearest_raw[..., :2650][0].cpu().numpy(),
        "linear": linear_raw[..., :2650][0].cpu().numpy(),
    }
    target_mask_native = target_mask[..., :2650][0].cpu().numpy().astype(bool)
    generated_domain = np.zeros_like(target_mask_native)
    generated_domain[1::2] = target_mask_native[1::2]
    masks, shared = shared_valid_mask(generated_domain, signals)
    pair_json = Path(dataset.rows[sample_index]["pair_json"])
    source_tfrecord, source_frame_index = _source_from_pair_json(pair_json)
    frame = _read_frame(source_tfrecord, source_frame_index)
    geometry = build_top_geometry(frame, _top_pose(frame), np.arange(0, 64, 2, dtype=np.int32), native_width=2650)
    result: dict[str, Any] = {
        "sample_index": sample_index,
        "segment_id": sample["segment_id"],
        "frame_index": int(sample["frame_index"]),
        "target_generated_valid_count": int(generated_domain.sum()),
        "shared_valid_count": int(shared.sum()),
        "shared_valid_fraction": float(shared.sum() / generated_domain.sum()) if generated_domain.any() else None,
    }
    for name in METHODS:
        result[f"{name}_valid_count"] = int(masks[name].sum())
        result[f"{name}_valid_fraction"] = float(masks[name].sum() / generated_domain.sum()) if generated_domain.any() else None
        result[f"{name}_native_metrics"] = _metrics_for_mask(signals[name], target, geometry, masks[name])
        result[f"{name}_shared_metrics"] = _metrics_for_mask(signals[name], target, geometry, shared)
    return result


def aggregate(rows: list[dict[str, Any]], metric_key: str) -> dict[str, Any]:
    result: dict[str, Any] = {"frames": len(rows)}
    for name in METHODS:
        available = [row[f"{name}_{metric_key}"] for row in rows if row[f"{name}_{metric_key}"]]
        total = sum(int(metric["count"]) for metric in available)
        summary: dict[str, Any] = {"frames": len(available), "count": total}
        for metric_name in ("coordinate_mae_m", "euclidean_mae_m", "intensity_mae"):
            summary[metric_name] = (
                sum(float(metric[metric_name]) * int(metric["count"]) for metric in available) / total
                if total else None
            )
        result[name] = summary
    return result


def main() -> None:
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    if args.range_scale_m <= 0:
        raise ValueError("range-scale-m must be positive")
    outputs = (args.output_root / "summary.json", args.output_root / "per_frame.csv")
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("output exists; pass --overwrite to replace it")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dataset = Waymo16To32Dataset(args.index, dataset_role=args.dataset_role,
                                 intensity_transform=args.intensity_transform,
                                 range_scale_m=args.range_scale_m)
    count = min(args.max_frames, len(dataset)) if args.max_frames else len(dataset)
    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = []
    for index in range(count):
        rows.append(evaluate_one(dataset, model, index, device, args.range_scale_m, args.intensity_transform))
        print(json.dumps({"evaluated": index + 1, "frames": count}), flush=True)
    shared_fraction = sum(row["shared_valid_count"] for row in rows) / sum(row["target_generated_valid_count"] for row in rows)
    summary = {
        "task": "Waymo 16-to-32 controlled generated-row point-cloud evaluation",
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_role": args.dataset_role, "evaluated_frames": count,
        "range_scale_m": args.range_scale_m, "intensity_transform": args.intensity_transform,
        "coordinate_frame": "Waymo vehicle frame",
        "target_scope": "valid generated odd rows only",
        "shared_mask_definition": "GT-valid generated rows and finite positive range/finiteness for all three methods",
        "shared_valid_fraction": shared_fraction,
        "method_native_mask_aggregate": aggregate(rows, "native_metrics"),
        "shared_mask_aggregate": aggregate(rows, "shared_metrics"),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_root / "summary.json", summary)
    fields = ["sample_index", "segment_id", "frame_index", "target_generated_valid_count", "shared_valid_count", "shared_valid_fraction"]
    for name in METHODS:
        fields += [f"{name}_valid_count", f"{name}_valid_fraction"]
        for scope in ("native", "shared"):
            for metric in ("coordinate_mae_m", "euclidean_mae_m", "intensity_mae"):
                fields.append(f"{name}_{scope}_{metric}")
    flat_rows = []
    for row in rows:
        flat = {key: row.get(key) for key in fields}
        for name in METHODS:
            for scope in ("native", "shared"):
                metrics = row.get(f"{name}_{scope}_metrics") or {}
                for metric in ("coordinate_mae_m", "euclidean_mae_m", "intensity_mae"):
                    flat[f"{name}_{scope}_{metric}"] = metrics.get(metric)
        flat_rows.append(flat)
    _atomic_csv(args.output_root / "per_frame.csv", flat_rows, fields)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
