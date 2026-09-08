#!/usr/bin/env python3
"""Audit rare high-intensity GT pixels in Waymo 16-to-32 pairs."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_THRESHOLDS = (20.0, 100.0, 1000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-role", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--high-threshold", type=float, default=1000.0)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("distribution requires finite non-empty values")
    levels = (50.0, 90.0, 95.0, 99.0, 99.5, 99.9, 99.99)
    quantiles = np.percentile(values, levels)
    result: dict[str, float | int] = {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }
    for level, value in zip(levels, quantiles):
        result["p" + str(level).replace(".", "_")] = float(value)
    return result


def load_index(index_path: Path, role: str, max_frames: int | None) -> list[dict[str, str]]:
    if max_frames is not None and max_frames < 1:
        raise ValueError("max-frames must be positive")
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset_role", "q_seg", "q_frame_index", "timestamp_micros", "pair_npz"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"index is missing columns: {sorted(missing)}")
        rows = [row for row in reader if row["dataset_role"] == role]
    rows.sort(key=lambda row: (row["q_seg"], int(row["q_frame_index"])))
    if max_frames is not None:
        rows = rows[:max_frames]
    if not rows:
        raise ValueError(f"no index rows for dataset_role={role}")
    return rows


def _resolve_pair(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _neighbor_support(
    intensity: np.ndarray,
    valid: np.ndarray,
    generated_rows: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return same-column and ±1-column observed-row support for generated pixels."""
    height, width = intensity.shape
    exact = np.zeros((height, width), dtype=bool)
    local = np.zeros((height, width), dtype=bool)
    generated_set = set(int(value) for value in generated_rows)
    for row in generated_rows:
        observed_neighbors = [candidate for candidate in (int(row) - 1, int(row) + 1)
                              if 0 <= candidate < height and candidate not in generated_set]
        for observed_row in observed_neighbors:
            high_valid = valid[observed_row] & (intensity[observed_row] >= threshold)
            exact[row] |= high_valid
            local[row] |= high_valid
            local[row] |= np.roll(high_valid, 1)
            local[row] |= np.roll(high_valid, -1)
    return exact, local


def audit_pair(
    pair_path: Path,
    identity: dict[str, Any],
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    high_threshold: float = 1000.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """Audit one pair and return frame summary, high-point rows and valid intensities."""
    with np.load(pair_path, allow_pickle=False) as data:
        required = {
            "target_32", "target_valid_mask_32", "ring_ids_32",
            "observed_target_rows", "generated_target_rows",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"pair is missing arrays: {sorted(missing)}")
        target = np.asarray(data["target_32"], dtype=np.float32)
        valid = np.asarray(data["target_valid_mask_32"], dtype=bool)
        ring_ids = np.asarray(data["ring_ids_32"], dtype=np.int32)
        observed_rows = np.asarray(data["observed_target_rows"], dtype=np.int32)
        generated_rows = np.asarray(data["generated_target_rows"], dtype=np.int32)
    if target.ndim != 3 or target.shape[2] != 2:
        raise ValueError(f"target_32 must be [H,W,2], got {target.shape}")
    if valid.shape != target.shape[:2] or ring_ids.shape != (target.shape[0],):
        raise ValueError("target, mask, or ring-id shape mismatch")
    if not np.isfinite(target).all():
        raise ValueError("target contains NaN or Inf")
    if set(observed_rows.tolist()) & set(generated_rows.tolist()):
        raise ValueError("observed and generated rows overlap")
    if sorted(np.concatenate((observed_rows, generated_rows)).tolist()) != list(range(target.shape[0])):
        raise ValueError("observed and generated rows do not cover target height")

    range_m = target[..., 0]
    intensity = target[..., 1]
    generated_domain = np.zeros_like(valid)
    generated_domain[generated_rows] = True
    generated_valid = generated_domain & valid
    valid_values = intensity[generated_valid]
    if valid_values.size == 0:
        raise ValueError("pair has no valid generated pixels")

    frame: dict[str, Any] = {
        **identity,
        "pair_npz": str(pair_path),
        "generated_valid_pixels": int(generated_valid.sum()),
        "generated_invalid_pixels": int((generated_domain & ~valid).sum()),
        "intensity_max": float(valid_values.max()),
        "intensity_p99_9": float(np.percentile(valid_values, 99.9)),
        "intensity_p99_99": float(np.percentile(valid_values, 99.99)),
    }
    support_by_threshold: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for threshold in thresholds:
        selected = generated_valid & (intensity >= threshold)
        exact, local = _neighbor_support(intensity, valid, generated_rows, threshold)
        support_by_threshold[threshold] = (exact, local)
        count = int(selected.sum())
        key = f"ge_{threshold:g}"
        frame[f"{key}_count"] = count
        frame[f"{key}_invalid_count"] = int(
            (generated_domain & ~valid & (intensity >= threshold)).sum()
        )
        frame[f"{key}_exact_observed_support"] = int((selected & exact).sum())
        frame[f"{key}_local_observed_support"] = int((selected & local).sum())

    high_selected = generated_valid & (intensity >= high_threshold)
    if high_threshold in support_by_threshold:
        exact_high, local_high = support_by_threshold[high_threshold]
    else:
        exact_high, local_high = _neighbor_support(
            intensity, valid, generated_rows, high_threshold
        )
    points: list[dict[str, Any]] = []
    generated_set = set(generated_rows.tolist())
    for row, column in np.argwhere(high_selected):
        neighbor_values: list[float] = []
        for observed_row in (int(row) - 1, int(row) + 1):
            if 0 <= observed_row < target.shape[0] and observed_row not in generated_set:
                if valid[observed_row, column]:
                    neighbor_values.append(float(intensity[observed_row, column]))
        points.append({
            **identity,
            "target_row": int(row),
            "native_ring_id": int(ring_ids[row]),
            "column": int(column),
            "range_m": float(range_m[row, column]),
            "raw_intensity": float(intensity[row, column]),
            "observed_neighbor_max": max(neighbor_values) if neighbor_values else "",
            "exact_observed_support": int(exact_high[row, column]),
            "local_observed_support": int(local_high[row, column]),
        })
    return frame, points, valid_values.astype(np.float64, copy=False)


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


def main() -> None:
    args = parse_args()
    if args.high_threshold <= 0:
        raise ValueError("high-threshold must be positive")
    output_files = (
        args.output_root / "summary.json",
        args.output_root / "per_frame.csv",
        args.output_root / "high_points.csv",
    )
    if not args.overwrite and any(path.exists() for path in output_files):
        raise FileExistsError("audit output exists; pass --overwrite to replace it")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = load_index(args.index, args.dataset_role, args.max_frames)
    frame_rows: list[dict[str, Any]] = []
    high_points: list[dict[str, Any]] = []
    value_chunks: list[np.ndarray] = []
    repo_root = args.repo_root.resolve()
    thresholds = tuple(sorted(set(DEFAULT_THRESHOLDS + (float(args.high_threshold),))))
    for index, row in enumerate(rows, start=1):
        identity = {
            "segment_id": row["q_seg"],
            "frame_index": int(row["q_frame_index"]),
            "timestamp_micros": int(row["timestamp_micros"]),
        }
        frame, points, values = audit_pair(
            _resolve_pair(repo_root, row["pair_npz"]),
            identity,
            thresholds=thresholds,
            high_threshold=args.high_threshold,
        )
        frame_rows.append(frame)
        high_points.extend(points)
        value_chunks.append(values)
        print(json.dumps({"audited": index, "frames": len(rows), "high_points": len(high_points)}), flush=True)

    all_values = np.concatenate(value_chunks)
    threshold_summary: dict[str, Any] = {}
    for threshold in thresholds:
        key = f"ge_{threshold:g}"
        count = sum(int(row[f"{key}_count"]) for row in frame_rows)
        exact = sum(int(row[f"{key}_exact_observed_support"]) for row in frame_rows)
        local = sum(int(row[f"{key}_local_observed_support"]) for row in frame_rows)
        invalid = sum(int(row[f"{key}_invalid_count"]) for row in frame_rows)
        threshold_summary[f"{threshold:g}"] = {
            "count": count,
            "fraction_of_generated_valid": count / int(all_values.size),
            "frames_with_points": sum(int(row[f"{key}_count"] > 0) for row in frame_rows),
            "invalid_pixel_count": invalid,
            "exact_observed_support_count": exact,
            "exact_observed_support_fraction": exact / count if count else None,
            "local_observed_support_count": local,
            "local_observed_support_fraction": local / count if count else None,
        }

    high_key = f"ge_{args.high_threshold:g}_count"
    ranked_frames = sorted(frame_rows, key=lambda row: int(row[high_key]), reverse=True)
    total_high = sum(int(row[high_key]) for row in frame_rows)
    summary = {
        "task": "Waymo 16-to-32 generated-ring GT high-intensity audit",
        "dataset_role": args.dataset_role,
        "evaluated_frames": len(frame_rows),
        "segments": len({row["segment_id"] for row in frame_rows}),
        "mask_scope": "valid_generated_odd_rows_only",
        "generated_valid_pixel_count": int(all_values.size),
        "target_intensity_distribution": distribution(all_values),
        "thresholds": threshold_summary,
        "high_threshold": args.high_threshold,
        "high_point_count": total_high,
        "top_frame_share": int(ranked_frames[0][high_key]) / total_high if total_high else None,
        "top_5_frame_share": sum(int(row[high_key]) for row in ranked_frames[:5]) / total_high if total_high else None,
        "top_frames": [
            {
                "segment_id": row["segment_id"],
                "frame_index": row["frame_index"],
                "count": row[high_key],
                "intensity_max": row["intensity_max"],
            }
            for row in ranked_frames[:10]
        ],
    }
    _atomic_json(output_files[0], summary)
    _atomic_csv(output_files[1], frame_rows, list(frame_rows[0].keys()))
    point_fields = list(high_points[0].keys()) if high_points else [
        "segment_id", "frame_index", "timestamp_micros", "target_row",
        "native_ring_id", "column", "range_m", "raw_intensity",
        "observed_neighbor_max", "exact_observed_support", "local_observed_support",
    ]
    _atomic_csv(output_files[2], high_points, point_fields)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
