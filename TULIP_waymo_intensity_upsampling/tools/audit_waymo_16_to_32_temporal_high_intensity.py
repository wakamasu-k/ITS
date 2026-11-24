#!/usr/bin/env python3
"""Measure whether the preceding 16-line scan supplies high-intensity evidence.

This is a read-only feasibility audit.  It does not change a TULIP input, model,
checkpoint, pair file, or TFRecord.  A current generated-row GT high-intensity
point is counted as temporally supported when a high-intensity point from the
preceding TOP LiDAR 16-line scan is geometrically close after ego-pose
compensation.  The proximity result is a candidate-evidence proxy, not a claim
that the earlier point can safely be fused: moving objects and occlusion still
need handling before a temporal Dataset is implemented.
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

# Keep ``python tools/<script>.py`` compatible with the existing command style.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.audit_waymo_16_to_32_high_intensity import (
    _neighbor_support,
    _resolve_pair,
    load_index,
)
from tools.reconstruct_waymo_16_to_32_pointcloud import _read_frame, _top_pose
from tulip_adapter.waymo_pointcloud import build_top_geometry, paired_vehicle_points
from waymo_preprocess.export_one_frame_64_32 import extract_range_intensity, top_first_return


RING_IDS_16 = np.arange(0, 64, 4, dtype=np.int32)
RING_IDS_32 = np.arange(0, 64, 2, dtype=np.int32)
GROUPS = ("exact_current_support", "local_only_current_support", "no_current_local_support")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-role", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--high-threshold", type=float, default=1000.0)
    parser.add_argument("--previous-frame-offset", type=int, default=1)
    parser.add_argument("--max-distance-m", type=float, default=1.0)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def transform_points(points: np.ndarray, current_from_previous: np.ndarray) -> np.ndarray:
    """Apply a homogeneous current-vehicle <- previous-vehicle transform."""
    values = np.asarray(points, dtype=np.float32)
    matrix = np.asarray(current_from_previous, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"points must be [N,3], got {values.shape}")
    if matrix.shape != (4, 4):
        raise ValueError(f"transform must be [4,4], got {matrix.shape}")
    if not np.isfinite(values).all() or not np.isfinite(matrix).all():
        raise ValueError("points and transform must be finite")
    homogeneous = np.concatenate(
        (values, np.ones((values.shape[0], 1), dtype=np.float32)), axis=1
    )
    return (homogeneous @ matrix.T)[:, :3].astype(np.float32)


def nearest_distances(query: np.ndarray, reference: np.ndarray, *, chunk_size: int = 256) -> np.ndarray:
    """Return exact Euclidean nearest distances without an optional SciPy dependency."""
    query_values = np.asarray(query, dtype=np.float32)
    reference_values = np.asarray(reference, dtype=np.float32)
    if query_values.ndim != 2 or query_values.shape[1] != 3:
        raise ValueError("query must be [N,3]")
    if reference_values.ndim != 2 or reference_values.shape[1] != 3:
        raise ValueError("reference must be [M,3]")
    if reference_values.shape[0] == 0:
        return np.full(query_values.shape[0], np.inf, dtype=np.float32)
    result = np.empty(query_values.shape[0], dtype=np.float32)
    for start in range(0, query_values.shape[0], chunk_size):
        stop = min(start + chunk_size, query_values.shape[0])
        delta = query_values[start:stop, None, :] - reference_values[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        result[start:stop] = np.sqrt(squared.min(axis=1))
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_pair(pair_path: Path) -> dict[str, np.ndarray]:
    with np.load(pair_path, allow_pickle=False) as data:
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


def _current_groups(pair: dict[str, np.ndarray], threshold: float) -> tuple[np.ndarray, dict[str, int]]:
    target = np.asarray(pair["target_32"], dtype=np.float32)
    valid = np.asarray(pair["target_valid_mask_32"], dtype=bool)
    generated_rows = np.asarray(pair["generated_target_rows"], dtype=np.int32)
    if target.shape != (32, 2650, 2) or valid.shape != (32, 2650):
        raise ValueError("expected native target [32,2650,2] and mask [32,2650]")
    exact, local = _neighbor_support(target[..., 1], valid, generated_rows, threshold)
    generated = np.zeros_like(valid)
    generated[generated_rows] = True
    high = generated & valid & (target[..., 1] >= threshold)
    labels = np.full(valid.shape, -1, dtype=np.int8)
    labels[high & exact] = 0
    labels[high & ~exact & local] = 1
    labels[high & ~local] = 2
    return labels, {group: int((labels == number).sum()) for number, group in enumerate(GROUPS)}


def audit_row(
    row: dict[str, str],
    *,
    repo_root: Path,
    high_threshold: float,
    previous_frame_offset: int,
    max_distance_m: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Audit temporal high-intensity evidence for one current pair."""
    pair_path = _resolve_pair(repo_root, row["pair_npz"])
    pair_json = _resolve_pair(repo_root, row["pair_json"])
    pair = _load_pair(pair_path)
    source_tfrecord, current_index = _source_identity(pair_json)
    previous_index = current_index - previous_frame_offset
    identity: dict[str, Any] = {
        "segment_id": row["q_seg"],
        "frame_index": current_index,
        "timestamp_micros": int(row["timestamp_micros"]),
        "previous_frame_index": previous_index,
    }
    empty = {group: 0 for group in GROUPS}
    if previous_index < 0:
        return {
            **identity,
            "status": "skipped_no_previous_frame",
            "previous_high_point_count": 0,
            "target_high_count": 0,
            **{f"{group}_target_count": 0 for group in GROUPS},
            **{f"{group}_temporal_candidate_count": 0 for group in GROUPS},
        }, empty

    labels, group_count = _current_groups(pair, high_threshold)
    current = _read_frame(source_tfrecord, current_index)
    previous = _read_frame(source_tfrecord, previous_index)
    current_geometry = build_top_geometry(current, _top_pose(current), RING_IDS_32, native_width=2650)
    previous_geometry = build_top_geometry(previous, _top_pose(previous), RING_IDS_16, native_width=2650)

    target = np.asarray(pair["target_32"], dtype=np.float32)
    current_distances = np.full(labels.shape, np.inf, dtype=np.float32)
    current_coordinates: dict[str, np.ndarray] = {}
    for number, group in enumerate(GROUPS):
        selected = labels == number
        points, _ = paired_vehicle_points(target[..., 0], target[..., 1], current_geometry, selected)
        current_coordinates[group] = points

    previous_native = top_first_return(previous)
    previous_range, previous_intensity, previous_valid = extract_range_intensity(previous_native)
    previous_range_16 = previous_range[RING_IDS_16]
    previous_intensity_16 = previous_intensity[RING_IDS_16]
    previous_valid_16 = previous_valid[RING_IDS_16]
    previous_high = previous_valid_16 & (previous_intensity_16 >= high_threshold)
    previous_points, _ = paired_vehicle_points(
        previous_range_16, previous_intensity_16, previous_geometry, previous_high
    )
    current_from_previous = (
        np.linalg.inv(np.asarray(current.pose.transform, dtype=np.float32).reshape(4, 4))
        @ np.asarray(previous.pose.transform, dtype=np.float32).reshape(4, 4)
    )
    previous_in_current = transform_points(previous_points, current_from_previous)
    temporal_counts: dict[str, int] = {}
    distance_values: list[np.ndarray] = []
    for number, group in enumerate(GROUPS):
        distances = nearest_distances(current_coordinates[group], previous_in_current)
        temporal_counts[group] = int((distances <= max_distance_m).sum())
        if distances.size:
            distance_values.append(distances)

    all_distances = np.concatenate(distance_values) if distance_values else np.empty(0, dtype=np.float32)
    translation = current_from_previous[:3, 3]
    return {
        **identity,
        "status": "ok",
        "previous_timestamp_micros": int(previous.timestamp_micros),
        "time_delta_s": (int(current.timestamp_micros) - int(previous.timestamp_micros)) / 1e6,
        "relative_translation_m": float(np.linalg.norm(translation)),
        "previous_high_point_count": int(previous_points.shape[0]),
        "target_high_count": int(sum(group_count.values())),
        **{f"{group}_target_count": group_count[group] for group in GROUPS},
        **{f"{group}_temporal_candidate_count": temporal_counts[group] for group in GROUPS},
        "nearest_previous_high_distance_median_m": (
            float(np.median(all_distances)) if all_distances.size else None
        ),
        "nearest_previous_high_distance_p95_m": (
            float(np.percentile(all_distances, 95.0)) if all_distances.size else None
        ),
    }, temporal_counts


def main() -> None:
    args = parse_args()
    if args.high_threshold <= 0 or args.max_distance_m <= 0:
        raise ValueError("high-threshold and max-distance-m must be positive")
    if args.previous_frame_offset < 1:
        raise ValueError("previous-frame-offset must be at least 1")
    outputs = (args.output_root / "summary.json", args.output_root / "per_frame.csv")
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("audit output exists; pass --overwrite to replace it")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = load_index(args.index, args.dataset_role, args.max_frames)
    records: list[dict[str, Any]] = []
    totals = {group: {"target": 0, "temporal_candidate": 0} for group in GROUPS}
    for count, row in enumerate(rows, start=1):
        record, temporal_counts = audit_row(
            row,
            repo_root=args.repo_root.resolve(),
            high_threshold=args.high_threshold,
            previous_frame_offset=args.previous_frame_offset,
            max_distance_m=args.max_distance_m,
        )
        records.append(record)
        if record["status"] == "ok":
            for group in GROUPS:
                totals[group]["target"] += int(record[f"{group}_target_count"])
                totals[group]["temporal_candidate"] += temporal_counts[group]
        print(json.dumps({"audited": count, "frames": len(rows), "status": record["status"]}), flush=True)

    summary_groups = {
        group: {
            "target_high_count": values["target"],
            "temporal_candidate_count": values["temporal_candidate"],
            "temporal_candidate_fraction": (
                values["temporal_candidate"] / values["target"] if values["target"] else None
            ),
        }
        for group, values in totals.items()
    }
    summary = {
        "task": "Waymo 16-to-32 preceding-frame high-intensity evidence audit",
        "purpose": "read-only temporal feasibility audit; not temporal fusion or model evaluation",
        "dataset_role": args.dataset_role,
        "evaluated_frames": len(records),
        "skipped_no_previous_frame": sum(record["status"] != "ok" for record in records),
        "high_threshold_raw_intensity": args.high_threshold,
        "previous_frame_offset": args.previous_frame_offset,
        "max_distance_m": args.max_distance_m,
        "target_scope": "valid generated odd rows with raw intensity >= threshold",
        "temporal_candidate_definition": (
            "preceding-frame 16-line raw-intensity-high point within max_distance_m "
            "after Waymo world-pose compensation; dynamic/occlusion filtering is not applied"
        ),
        "groups": summary_groups,
        "caution": (
            "A temporal candidate is evidence of possible additional input information, "
            "not a valid fused measurement. Dynamic-object and occlusion handling are required "
            "before building a temporal Dataset."
        ),
    }
    _atomic_json(args.output_root / "summary.json", summary)
    fields = list(records[0].keys()) if records else []
    _atomic_csv(args.output_root / "per_frame.csv", records, fields)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
