#!/usr/bin/env python3
"""Rank Waymo TFRecords by ego-motion without decoding images or point clouds."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--pattern", default="*.tfrecord")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=40)
    return parser.parse_args()


def wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def inspect_segment(path: Path, max_frames: int) -> dict[str, object]:
    positions: list[np.ndarray] = []
    yaws: list[float] = []
    dataset = tf.data.TFRecordDataset(str(path), compression_type="")
    for raw in dataset.take(max_frames):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(raw.numpy()))
        pose = np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)
        positions.append(pose[:3, 3])
        yaws.append(math.degrees(math.atan2(pose[1, 0], pose[0, 0])))

    if len(positions) < 2:
        raise ValueError(f"only {len(positions)} frame(s)")
    xyz = np.stack(positions)
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    yaw_steps = np.asarray([
        abs(wrap_degrees(b - a)) for a, b in zip(yaws[:-1], yaws[1:])
    ])
    return {
        "segment": path.stem,
        "tfrecord": str(path),
        "frames_scanned": len(positions),
        "path_length_m": float(steps.sum()),
        "displacement_m": float(np.linalg.norm(xyz[-1] - xyz[0])),
        "max_step_m": float(steps.max()),
        "yaw_change_deg": float(yaw_steps.sum()),
        "net_yaw_deg": float(abs(wrap_degrees(yaws[-1] - yaws[0]))),
        "error": "",
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob(args.pattern))[: args.limit]
    rows: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        try:
            rows.append(inspect_segment(path, args.max_frames))
        except Exception as exc:  # retain failures in the audit table
            rows.append({
                "segment": path.stem,
                "tfrecord": str(path),
                "frames_scanned": 0,
                "path_length_m": 0.0,
                "displacement_m": 0.0,
                "max_step_m": 0.0,
                "yaw_change_deg": 0.0,
                "net_yaw_deg": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            })
    rows.sort(key=lambda row: float(row["path_length_m"]), reverse=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.output_csv.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
