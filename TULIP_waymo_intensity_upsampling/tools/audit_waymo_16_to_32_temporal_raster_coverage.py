#!/usr/bin/env python3
"""Read-only audit of preceding static 16-line points rasterized to current input rays.

The tool does not create a Dataset or change TULIP.  It asks a stricter
question than 3-D proximity: after pose compensation, can preceding static
TOP-LiDAR points be placed on *current 16-line input pixels* that are invalid
in the current scan?  Only this rasterized, non-GT-derived evidence could be
given to a later temporal-input Dataset.
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.audit_waymo_16_to_32_high_intensity import _neighbor_support, _resolve_pair, load_index
from tools.audit_waymo_16_to_32_temporal_high_intensity import transform_points
from tools.reconstruct_waymo_16_to_32_pointcloud import _read_frame, _top_pose
from tulip_adapter.tulip_static_map import static_point_mask
from tulip_adapter.waymo_pointcloud import (
    build_top_geometry,
    paired_vehicle_points,
    range_image_to_vehicle_points,
)
from waymo_preprocess.export_one_frame_64_32 import extract_range_intensity, top_first_return


RING_IDS_16 = np.arange(0, 64, 4, dtype=np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-role", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--high-threshold", type=float, default=1000.0)
    parser.add_argument("--previous-frame-offset", type=int, default=1)
    parser.add_argument("--max-angular-distance-deg", type=float, default=0.25)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def _load_pair(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "input_16", "input_valid_mask_16", "target_32", "target_valid_mask_32",
            "observed_target_rows", "generated_target_rows",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"pair is missing arrays: {sorted(missing)}")
        return {name: np.asarray(data[name]) for name in required}


def _source_identity(pair_json: Path) -> tuple[Path, int]:
    metadata = json.loads(pair_json.read_text(encoding="utf-8"))
    source = metadata.get("source_tfrecord")
    frame_index = metadata.get("frame_index")
    if not source or frame_index is None:
        raise KeyError("pair JSON must contain source_tfrecord and frame_index")
    return Path(str(source)), int(frame_index)


def current_input_rays(geometry: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return current TOP sensor origin and one official unit-bearing vector per input pixel."""
    height, width = int(geometry.ring_ids.size), int(geometry.native_width)
    ones = np.ones((height, width), dtype=np.float32)
    all_pixels = np.ones((height, width), dtype=bool)
    unit_points = range_image_to_vehicle_points(ones, geometry, all_pixels)
    if unit_points.shape != (height * width, 3):
        raise ValueError("official unit-ray conversion did not preserve the full input grid")
    extrinsic = np.asarray(geometry.extrinsic[0].numpy(), dtype=np.float32)
    origin = extrinsic[:3, 3]
    vectors = unit_points - origin[None, :]
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ValueError("non-finite current input ray")
    return origin, (vectors / lengths[:, None]).astype(np.float32)


def rasterize_static_points(
    points_current_vehicle: np.ndarray,
    intensity: np.ndarray,
    *,
    sensor_origin_vehicle: np.ndarray,
    input_unit_rays: np.ndarray,
    input_shape: tuple[int, int],
    max_angular_distance_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-buffer static points onto the closest current input ray within an angular gate."""
    from scipy.spatial import cKDTree

    points = np.asarray(points_current_vehicle, dtype=np.float32)
    values = np.asarray(intensity, dtype=np.float32)
    origin = np.asarray(sensor_origin_vehicle, dtype=np.float32)
    rays = np.asarray(input_unit_rays, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or values.shape != (points.shape[0],):
        raise ValueError("points must be [N,3] and intensity must be [N]")
    if rays.shape != (input_shape[0] * input_shape[1], 3):
        raise ValueError("input_unit_rays shape does not match input_shape")
    if max_angular_distance_deg <= 0:
        raise ValueError("max_angular_distance_deg must be positive")
    output_range = np.zeros(input_shape, dtype=np.float32)
    output_intensity = np.zeros(input_shape, dtype=np.float32)
    output_valid = np.zeros(input_shape, dtype=bool)
    if points.shape[0] == 0:
        return output_range, output_intensity, output_valid

    vectors = points - origin[None, :]
    ranges = np.linalg.norm(vectors, axis=1)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(values) & np.isfinite(ranges) & (ranges > 0)
    if not finite.any():
        return output_range, output_intensity, output_valid
    vectors, ranges, values = vectors[finite], ranges[finite], values[finite]
    directions = vectors / ranges[:, None]
    tree = cKDTree(rays)
    chord, pixel = tree.query(directions, k=1)
    chord = np.clip(np.asarray(chord, dtype=np.float64), 0.0, 2.0)
    angles = np.degrees(2.0 * np.arcsin(chord / 2.0))
    keep = angles <= max_angular_distance_deg
    if not keep.any():
        return output_range, output_intensity, output_valid
    pixel, ranges, values = pixel[keep], ranges[keep], values[keep]
    order = np.lexsort((ranges, pixel))
    pixel, ranges, values = pixel[order], ranges[order], values[order]
    first = np.r_[True, pixel[1:] != pixel[:-1]]
    pixel, ranges, values = pixel[first], ranges[first], values[first]
    output_range.flat[pixel] = ranges
    output_intensity.flat[pixel] = values
    output_valid.flat[pixel] = True
    return output_range, output_intensity, output_valid


def evaluate_temporal_fill(
    pair: dict[str, np.ndarray],
    temporal_intensity: np.ndarray,
    temporal_valid: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """Count input fills and high-GT pixels newly supported by those fills."""
    current = np.asarray(pair["input_16"], dtype=np.float32)
    current_valid = np.asarray(pair["input_valid_mask_16"], dtype=bool)
    target = np.asarray(pair["target_32"], dtype=np.float32)
    target_valid = np.asarray(pair["target_valid_mask_32"], dtype=bool)
    observed_rows = np.asarray(pair["observed_target_rows"], dtype=np.int32)
    generated_rows = np.asarray(pair["generated_target_rows"], dtype=np.int32)
    if current.shape != (16, 2650, 2) or temporal_intensity.shape != (16, 2650):
        raise ValueError("expected native 16-line input and temporal raster")
    fill = ~current_valid & temporal_valid
    combined_intensity = np.zeros((32, 2650), dtype=np.float32)
    combined_valid = np.zeros((32, 2650), dtype=bool)
    combined_intensity[observed_rows] = current[..., 1]
    combined_valid[observed_rows] = current_valid
    combined_intensity[observed_rows][fill] = temporal_intensity[fill]
    combined_valid[observed_rows][fill] = True
    exact_before, local_before = _neighbor_support(target[..., 1], target_valid, generated_rows, threshold)
    exact_after, local_after = _neighbor_support(combined_intensity, combined_valid, generated_rows, threshold)
    generated = np.zeros_like(target_valid)
    generated[generated_rows] = True
    high = generated & target_valid & (target[..., 1] >= threshold)
    no_local_before = high & ~local_before
    return {
        "current_valid_input_pixels": int(current_valid.sum()),
        "temporal_raster_valid_pixels": int(temporal_valid.sum()),
        "temporal_filled_input_pixels": int(fill.sum()),
        "target_high_generated_pixels": int(high.sum()),
        "target_high_no_current_local_support": int(no_local_before.sum()),
        "target_high_temporal_exact_support": int((no_local_before & exact_after).sum()),
        "target_high_temporal_local_only_support": int((no_local_before & ~exact_after & local_after).sum()),
    }


def audit_row(
    row: dict[str, str], *, repo_root: Path, threshold: float,
    previous_offset: int, max_angle_deg: float,
) -> dict[str, Any]:
    pair = _load_pair(_resolve_pair(repo_root, row["pair_npz"]))
    source_tfrecord, current_index = _source_identity(_resolve_pair(repo_root, row["pair_json"]))
    previous_index = current_index - previous_offset
    base = {
        "segment_id": row["q_seg"], "frame_index": current_index,
        "timestamp_micros": int(row["timestamp_micros"]), "previous_frame_index": previous_index,
    }
    if previous_index < 0:
        return {**base, "status": "skipped_no_previous_frame"}
    current_frame = _read_frame(source_tfrecord, current_index)
    previous_frame = _read_frame(source_tfrecord, previous_index)
    current_geometry = build_top_geometry(current_frame, _top_pose(current_frame), RING_IDS_16, native_width=2650)
    previous_geometry = build_top_geometry(previous_frame, _top_pose(previous_frame), RING_IDS_16, native_width=2650)
    previous_native = top_first_return(previous_frame)
    previous_range, previous_intensity, previous_valid = extract_range_intensity(previous_native)
    previous_range = previous_range[RING_IDS_16]
    previous_intensity = previous_intensity[RING_IDS_16]
    previous_valid = previous_valid[RING_IDS_16]
    previous_points, previous_values = paired_vehicle_points(
        previous_range, previous_intensity, previous_geometry, previous_valid
    )
    keep_static = static_point_mask(previous_points, previous_frame.laser_labels)
    current_from_previous = (
        np.linalg.inv(np.asarray(current_frame.pose.transform, dtype=np.float32).reshape(4, 4))
        @ np.asarray(previous_frame.pose.transform, dtype=np.float32).reshape(4, 4)
    )
    points_current = transform_points(previous_points[keep_static], current_from_previous)
    origin, rays = current_input_rays(current_geometry)
    _, temporal_intensity, temporal_valid = rasterize_static_points(
        points_current, previous_values[keep_static], sensor_origin_vehicle=origin,
        input_unit_rays=rays, input_shape=(16, 2650), max_angular_distance_deg=max_angle_deg,
    )
    metrics = evaluate_temporal_fill(pair, temporal_intensity, temporal_valid, threshold)
    return {
        **base,
        "status": "ok",
        "time_delta_s": (int(current_frame.timestamp_micros) - int(previous_frame.timestamp_micros)) / 1e6,
        "previous_points_before_static_filter": int(previous_points.shape[0]),
        "previous_static_points": int(keep_static.sum()),
        **metrics,
    }


def main() -> None:
    args = parse_args()
    if args.high_threshold <= 0 or args.previous_frame_offset < 1 or args.max_angular_distance_deg <= 0:
        raise ValueError("threshold, frame offset, and angular distance must be positive")
    outputs = (args.output_root / "summary.json", args.output_root / "per_frame.csv")
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("audit output exists; pass --overwrite to replace it")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = load_index(args.index, args.dataset_role, args.max_frames)
    records = []
    for count, row in enumerate(rows, start=1):
        record = audit_row(row, repo_root=args.repo_root.resolve(), threshold=args.high_threshold,
                           previous_offset=args.previous_frame_offset,
                           max_angle_deg=args.max_angular_distance_deg)
        records.append(record)
        print(json.dumps({"audited": count, "frames": len(rows), "status": record["status"]}), flush=True)
    active = [record for record in records if record["status"] == "ok"]
    sums = {key: sum(int(record.get(key, 0)) for record in active) for key in (
        "current_valid_input_pixels", "temporal_raster_valid_pixels", "temporal_filled_input_pixels",
        "target_high_generated_pixels", "target_high_no_current_local_support",
        "target_high_temporal_exact_support", "target_high_temporal_local_only_support",
    )}
    new_support = sums["target_high_temporal_exact_support"] + sums["target_high_temporal_local_only_support"]
    denominator = sums["target_high_no_current_local_support"]
    summary = {
        "task": "Waymo 16-to-32 temporal input-raster coverage audit",
        "purpose": "read-only pre-training audit; no TULIP model, Dataset, or pair was modified",
        "dataset_role": args.dataset_role, "evaluated_frames": len(records),
        "processed_frames": len(active), "skipped_no_previous_frame": len(records) - len(active),
        "high_threshold_raw_intensity": args.high_threshold,
        "previous_frame_offset": args.previous_frame_offset,
        "max_angular_distance_deg": args.max_angular_distance_deg,
        "prior_point_policy": "preceding 16-line TOP points; dynamic laser-label boxes and ego roof removed",
        "raster_policy": "closest current official input ray within angular gate; nearest-range z-buffer; fill current-invalid input only",
        "aggregate": {
            **sums,
            "target_high_newly_supported_by_temporal_input": new_support,
            "fraction_of_current_no_local_support_newly_supported": new_support / denominator if denominator else None,
        },
        "caution": "This measures temporal input availability only. MAE, Precision, Recall, and F1 require a separately trained temporal-input checkpoint.",
    }
    _atomic_json(args.output_root / "summary.json", summary)
    fields = list(dict.fromkeys(key for record in records for key in record))
    _atomic_csv(args.output_root / "per_frame.csv", records, fields)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
