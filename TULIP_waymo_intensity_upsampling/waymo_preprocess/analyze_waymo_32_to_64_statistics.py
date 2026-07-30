#!/usr/bin/env python3
"""Analyze multiple Waymo scenes for stable 32->64 normalization choices."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

if __package__ in (None, ""):
    from statistics_core import (
        EVEN_ROWS,
        ODD_ROWS,
        FrameBalancedSampler,
        choose_frame_indices,
        describe_values,
        frame_statistics,
        valid_values,
    )
else:
    from waymo_preprocess.statistics_core import (
        EVEN_ROWS,
        ODD_ROWS,
        FrameBalancedSampler,
        choose_frame_indices,
        describe_values,
        frame_statistics,
        valid_values,
    )


SCRIPT_VERSION = "waymo-32-to-64-statistics-v1"
GROUP_ROWS = {
    "all_64": np.arange(64, dtype=np.int32),
    "observed_even_32": EVEN_ROWS,
    "generated_odd_32": ODD_ROWS,
}


def matrix_float_to_numpy(message: Any) -> np.ndarray:
    dimensions = [
        int(value.size if hasattr(value, "size") else value)
        for value in message.shape.dims
    ]
    values = np.asarray(message.data, dtype=np.float32)
    expected = int(np.prod(dimensions))
    if values.size != expected:
        raise ValueError(f"MatrixFloat size mismatch: {values.size} != {expected}")
    return values.reshape(dimensions)


def top_first_return(frame: Any) -> np.ndarray:
    range_images, _, _, _ = frame_utils.parse_range_image_and_camera_projection(
        frame
    )
    top = open_dataset.LaserName.TOP
    if top not in range_images or not range_images[top]:
        raise KeyError("TOP LiDAR first return is unavailable")
    native = matrix_float_to_numpy(range_images[top][0])
    if native.ndim != 3 or native.shape[0] != 64 or native.shape[2] < 2:
        raise ValueError(f"unexpected TOP range image shape: {native.shape}")
    return native


def count_records(tfrecord: Path) -> int:
    dataset = tf.data.TFRecordDataset(str(tfrecord), compression_type="")
    return sum(1 for _ in dataset)


def selected_frames(tfrecord: Path, indices: list[int]):
    wanted = set(indices)
    dataset = tf.data.TFRecordDataset(str(tfrecord), compression_type="")
    for frame_index, record in enumerate(dataset):
        if frame_index not in wanted:
            continue
        frame = open_dataset.Frame()
        frame.ParseFromString(record.numpy())
        yield frame_index, frame
        if frame_index == indices[-1]:
            break


def flatten_frame_row(
    *,
    scene: str,
    frame_index: int,
    timestamp_micros: int,
    statistics: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scene": scene,
        "frame_index": frame_index,
        "timestamp_micros": timestamp_micros,
    }
    for group, modalities in statistics.items():
        for modality, metrics in modalities.items():
            for metric, value in metrics.items():
                row[f"{group}.{modality}.{metric}"] = value
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scene_rows(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        grouped[str(row["scene"])].append(row)
    output: list[dict[str, Any]] = []
    for scene, rows in sorted(grouped.items()):
        result: dict[str, Any] = {"scene": scene, "sampled_frames": len(rows)}
        numeric_keys = [
            key
            for key in rows[0]
            if key not in {"scene", "frame_index", "timestamp_micros"}
        ]
        for key in numeric_keys:
            values = [
                float(row[key])
                for row in rows
                if row[key] is not None and np.isfinite(float(row[key]))
            ]
            if values:
                result[f"mean.{key}"] = float(np.mean(values))
                result[f"max.{key}"] = float(np.max(values))
        output.append(result)
    return output


def save_histograms(
    samplers: dict[str, dict[str, FrameBalancedSampler]],
    output_dir: Path,
) -> None:
    colors = {
        "all_64": "#1f77b4",
        "observed_even_32": "#ff7f0e",
        "generated_odd_32": "#2ca02c",
    }

    range_figure, range_axis = plt.subplots(
        figsize=(10, 6), constrained_layout=True
    )
    for group, modalities in samplers.items():
        values = modalities["range_m"].array()
        range_axis.hist(
            values,
            bins=160,
            range=(0, 80),
            density=True,
            histtype="step",
            linewidth=1.6,
            label=group,
            color=colors[group],
        )
    range_axis.set_title("Waymo TOP LiDAR range distribution")
    range_axis.set_xlabel("Range [m]")
    range_axis.set_ylabel("Density")
    range_axis.legend()
    range_figure.savefig(output_dir / "range_histogram.png", dpi=170)
    plt.close(range_figure)

    all_intensity = samplers["all_64"]["intensity"].array()
    display_max = float(np.percentile(all_intensity, 99.9))
    intensity_figure, intensity_axis = plt.subplots(
        figsize=(10, 6), constrained_layout=True
    )
    for group, modalities in samplers.items():
        values = modalities["intensity"].array()
        intensity_axis.hist(
            np.clip(values, 0, display_max),
            bins=160,
            range=(0, display_max),
            density=True,
            histtype="step",
            linewidth=1.6,
            label=group,
            color=colors[group],
        )
    intensity_axis.set_title(
        f"Raw intensity distribution (display clipped at sampled p99.9={display_max:.6g})"
    )
    intensity_axis.set_xlabel("Raw intensity")
    intensity_axis.set_ylabel("Density")
    intensity_axis.legend()
    intensity_figure.savefig(output_dir / "intensity_histogram.png", dpi=170)
    plt.close(intensity_figure)

    log_figure, log_axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for group, modalities in samplers.items():
        values = modalities["intensity"].array()
        log_axis.hist(
            np.log1p(np.clip(values, 0, None)),
            bins=180,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=group,
            color=colors[group],
        )
    log_axis.set_title("Raw intensity log1p distribution")
    log_axis.set_xlabel("log1p(raw intensity)")
    log_axis.set_ylabel("Density")
    log_axis.legend()
    log_figure.savefig(output_dir / "intensity_log_histogram.png", dpi=170)
    plt.close(log_figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-scenes", type=int, default=10)
    parser.add_argument("--frames-per-scene", type=int, default=20)
    parser.add_argument("--sample-values-per-frame", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_scenes <= 0 or args.frames_per_scene <= 0:
        raise ValueError("max-scenes and frames-per-scene must be positive")
    if args.sample_values_per_frame <= 0:
        raise ValueError("sample-values-per-frame must be positive")
    tfrecords = sorted(args.input_root.glob("*.tfrecord"))[: args.max_scenes]
    if not tfrecords:
        raise FileNotFoundError(f"no TFRecords found in {args.input_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samplers = {
        group: {
            "range_m": FrameBalancedSampler(args.sample_values_per_frame),
            "intensity": FrameBalancedSampler(args.sample_values_per_frame),
        }
        for group in GROUP_ROWS
    }
    frame_rows: list[dict[str, Any]] = []
    selected_sources: list[dict[str, Any]] = []

    for scene_number, tfrecord in enumerate(tfrecords, start=1):
        frame_count = count_records(tfrecord)
        indices = choose_frame_indices(frame_count, args.frames_per_scene)
        print(
            f"[{scene_number}/{len(tfrecords)}] "
            f"{tfrecord.name}: total={frame_count}, selected={indices}"
        )
        processed = 0
        for frame_index, frame in selected_frames(tfrecord, indices):
            native = top_first_return(frame)
            range_64 = native[..., 0]
            intensity_64 = native[..., 1]
            mask_64 = range_64 > 0
            statistics = frame_statistics(range_64, intensity_64, mask_64)
            frame_rows.append(
                flatten_frame_row(
                    scene=tfrecord.stem,
                    frame_index=frame_index,
                    timestamp_micros=int(frame.timestamp_micros),
                    statistics=statistics,
                )
            )
            for group, rows in GROUP_ROWS.items():
                group_mask = mask_64[rows]
                samplers[group]["range_m"].add(
                    valid_values(range_64[rows], group_mask)
                )
                samplers[group]["intensity"].add(
                    valid_values(intensity_64[rows], group_mask)
                )
            processed += 1
        selected_sources.append(
            {
                "tfrecord": str(tfrecord.resolve()),
                "total_frames": frame_count,
                "selected_frame_indices": indices,
                "processed_frames": processed,
            }
        )

    global_statistics = {
        group: {
            modality: describe_values(sampler.array())
            for modality, sampler in modalities.items()
        }
        for group, modalities in samplers.items()
    }
    all_intensity_stats = global_statistics["all_64"]["intensity"]
    outlier_reference = float(all_intensity_stats["p99_9"])
    outlier_rows = [
        row
        for row in frame_rows
        if (
            float(row["all_64.intensity.max"]) > 10.0 * outlier_reference
            or int(row["all_64.intensity.nan_count"]) > 0
            or int(row["all_64.intensity.inf_count"]) > 0
            or int(row["all_64.range_m.negative_count"]) > 0
        )
    ]

    write_csv(args.output_dir / "per_frame_statistics.csv", frame_rows)
    write_csv(args.output_dir / "per_scene_statistics.csv", scene_rows(frame_rows))
    if outlier_rows:
        write_csv(args.output_dir / "outlier_frames.csv", outlier_rows)
    else:
        (args.output_dir / "outlier_frames.csv").write_text(
            "scene,frame_index,timestamp_micros\n",
            encoding="utf-8",
        )
    save_histograms(samplers, args.output_dir)

    report = {
        "script_version": SCRIPT_VERSION,
        "input_root": str(args.input_root.resolve()),
        "scene_count": len(tfrecords),
        "processed_frame_count": len(frame_rows),
        "frames_per_scene_requested": args.frames_per_scene,
        "sampling": {
            "type": "deterministic frame-balanced value sample",
            "values_per_frame_per_group": args.sample_values_per_frame,
            "note": (
                "Global percentile/histogram values are frame-balanced estimates; "
                "per-frame CSV values are exact."
            ),
        },
        "group_rows": {
            group: rows.tolist() for group, rows in GROUP_ROWS.items()
        },
        "global_statistics": global_statistics,
        "outlier_rule": {
            "intensity_max_greater_than": 10.0 * outlier_reference,
            "or_nan_inf_negative_range": True,
        },
        "outlier_frame_count": len(outlier_rows),
        "selected_sources": selected_sources,
    }
    report_path = args.output_dir / "statistics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(global_statistics, ensure_ascii=False, indent=2))
    print(f"outlier_frame_count={len(outlier_rows)}")
    print(f"saved_statistics={report_path}")


if __name__ == "__main__":
    main()
