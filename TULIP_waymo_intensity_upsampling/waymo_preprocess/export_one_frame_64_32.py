#!/usr/bin/env python3
"""Export matching native 64-line and even-ring 32-line Waymo arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils


PREPROCESSING_VERSION = "shimizu-compatible-one-frame-v1"
RING_IDS_32 = np.arange(0, 64, 2, dtype=np.int32)


def extract_range_intensity(
    native_range_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    native = np.asarray(native_range_image, dtype=np.float32)
    if native.ndim != 3 or native.shape[0] != 64 or native.shape[2] < 2:
        raise ValueError(
            "Expected TOP range image [64,W,C>=2], "
            f"got {list(native.shape)}"
        )
    range_64 = native[..., 0]
    intensity_64 = native[..., 1]
    valid_mask_64 = range_64 > 0
    return range_64, intensity_64, valid_mask_64


def select_even_rings(
    range_64: np.ndarray,
    intensity_64: np.ndarray,
    valid_mask_64: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shapes = {
        tuple(np.asarray(range_64).shape),
        tuple(np.asarray(intensity_64).shape),
        tuple(np.asarray(valid_mask_64).shape),
    }
    if len(shapes) != 1 or next(iter(shapes))[0] != 64:
        raise ValueError(
            "range, intensity and mask must share shape [64,W], "
            f"got {sorted(shapes)}"
        )
    return (
        np.asarray(range_64)[RING_IDS_32],
        np.asarray(intensity_64)[RING_IDS_32],
        np.asarray(valid_mask_64, dtype=bool)[RING_IDS_32],
        RING_IDS_32.copy(),
    )


def matrix_float_to_numpy(message: Any) -> np.ndarray:
    dimensions = [
        int(value.size if hasattr(value, "size") else value)
        for value in message.shape.dims
    ]
    values = np.asarray(message.data, dtype=np.float32)
    expected = int(np.prod(dimensions))
    if values.size != expected:
        raise ValueError(
            f"MatrixFloat size mismatch: {values.size} != {expected}"
        )
    return values.reshape(dimensions)


def read_frame(tfrecord: Path, frame_index: int) -> Any:
    if frame_index < 0:
        raise ValueError("frame-index must be non-negative")
    dataset = tf.data.TFRecordDataset(str(tfrecord), compression_type="")
    for index, record in enumerate(dataset):
        if index == frame_index:
            frame = open_dataset.Frame()
            frame.ParseFromString(record.numpy())
            return frame
    raise IndexError(f"frame {frame_index} does not exist in {tfrecord}")


def top_first_return(frame: Any) -> np.ndarray:
    range_images, _, _, _ = frame_utils.parse_range_image_and_camera_projection(
        frame
    )
    top = open_dataset.LaserName.TOP
    if top not in range_images or not range_images[top]:
        raise KeyError("TOP LiDAR first return is unavailable")
    return matrix_float_to_numpy(range_images[top][0])


def valid_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {"count": 0}
    return {
        "count": int(selected.size),
        "min": float(np.min(selected)),
        "median": float(np.median(selected)),
        "p99_5": float(np.percentile(selected, 99.5)),
        "max": float(np.max(selected)),
        "mean": float(np.mean(selected)),
        "std": float(np.std(selected)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord", required=True, type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tfrecord.is_file():
        raise FileNotFoundError(args.tfrecord)

    frame = read_frame(args.tfrecord, args.frame_index)
    native = top_first_return(frame)
    range_64, intensity_64, mask_64 = extract_range_intensity(native)
    range_32, intensity_32, mask_32, ring_ids_32 = select_even_rings(
        range_64, intensity_64, mask_64
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / (
        f"waymo_top_frame_{args.frame_index:06d}.npz"
    )
    np.savez_compressed(
        npz_path,
        range_64=range_64.astype(np.float32),
        intensity_64=intensity_64.astype(np.float32),
        valid_mask_64=mask_64,
        range_32=range_32.astype(np.float32),
        intensity_32=intensity_32.astype(np.float32),
        valid_mask_32=mask_32,
        ring_ids_32=ring_ids_32,
    )

    metadata = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "source_tfrecord": str(args.tfrecord.resolve()),
        "source_frame_index": int(args.frame_index),
        "context_name": str(frame.context.name),
        "timestamp_micros": int(frame.timestamp_micros),
        "lidar_name": "TOP",
        "return_index": 0,
        "native_range_image_shape": [int(value) for value in native.shape],
        "range_shape_64": [int(value) for value in range_64.shape],
        "range_shape_32": [int(value) for value in range_32.shape],
        "ring_keep_rule": "even",
        "ring_ids_32": ring_ids_32.tolist(),
        "valid_count_64": int(mask_64.sum()),
        "valid_count_32": int(mask_32.sum()),
        "range_64_valid_m": valid_stats(range_64, mask_64),
        "intensity_64_valid": valid_stats(intensity_64, mask_64),
        "range_32_valid_m": valid_stats(range_32, mask_32),
        "intensity_32_valid": valid_stats(intensity_32, mask_32),
        "normalization": "none",
        "histogram_equalization": False,
        "clahe": False,
        "temporal_accumulation": False,
        "dynamic_object_removal": False,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"saved_npz={npz_path}")
    print(f"saved_metadata={metadata_path}")


if __name__ == "__main__":
    main()
