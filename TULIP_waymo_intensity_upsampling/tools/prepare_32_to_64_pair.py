#!/usr/bin/env python3
"""Prepare one traceable Waymo 32-line input -> 64-line target pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EVEN_ROWS = np.arange(0, 64, 2, dtype=np.int32)
ODD_ROWS = np.arange(1, 64, 2, dtype=np.int32)
REQUIRED_KEYS = {
    "range_64",
    "intensity_64",
    "valid_mask_64",
    "range_32",
    "intensity_32",
    "valid_mask_32",
    "ring_ids_32",
}


def build_pair(data: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    keys = set(data.files)
    missing = REQUIRED_KEYS - keys
    if missing:
        raise KeyError(f"missing arrays: {sorted(missing)}")

    range_64 = np.asarray(data["range_64"], dtype=np.float32)
    intensity_64 = np.asarray(data["intensity_64"], dtype=np.float32)
    mask_64 = np.asarray(data["valid_mask_64"], dtype=bool)
    range_32 = np.asarray(data["range_32"], dtype=np.float32)
    intensity_32 = np.asarray(data["intensity_32"], dtype=np.float32)
    mask_32 = np.asarray(data["valid_mask_32"], dtype=bool)
    ring_ids = np.asarray(data["ring_ids_32"], dtype=np.int32)

    if range_64.ndim != 2 or range_64.shape[0] != 64:
        raise ValueError(f"range_64 must be [64,W], got {range_64.shape}")
    width = range_64.shape[1]
    expected_64 = (64, width)
    expected_32 = (32, width)
    for name, array in (
        ("intensity_64", intensity_64),
        ("valid_mask_64", mask_64),
    ):
        if array.shape != expected_64:
            raise ValueError(f"{name} must be {expected_64}, got {array.shape}")
    for name, array in (
        ("range_32", range_32),
        ("intensity_32", intensity_32),
        ("valid_mask_32", mask_32),
    ):
        if array.shape != expected_32:
            raise ValueError(f"{name} must be {expected_32}, got {array.shape}")
    if not np.array_equal(ring_ids, EVEN_ROWS):
        raise ValueError(f"ring_ids_32 must be even rows, got {ring_ids.tolist()}")

    # The equality checks prevent a mismatched or independently processed
    # 32-line file from being paired with this 64-line target.
    if not np.array_equal(range_32, range_64[EVEN_ROWS]):
        raise ValueError("range_32 is not the exact even-row subset of range_64")
    if not np.array_equal(intensity_32, intensity_64[EVEN_ROWS]):
        raise ValueError(
            "intensity_32 is not the exact even-row subset of intensity_64"
        )
    if not np.array_equal(mask_32, mask_64[EVEN_ROWS]):
        raise ValueError("valid_mask_32 does not match valid_mask_64 even rows")

    arrays = {
        "input_32": np.stack((range_32, intensity_32), axis=-1),
        "input_valid_mask_32": mask_32,
        "target_64": np.stack((range_64, intensity_64), axis=-1),
        "target_valid_mask_64": mask_64,
        "observed_target_rows": EVEN_ROWS,
        "generated_target_rows": ODD_ROWS,
    }
    metadata = {
        "task": "Waymo TOP LiDAR 32-to-64 range and intensity upsampling",
        "input_shape": [32, width, 2],
        "target_shape": [64, width, 2],
        "channel_semantics": {"0": "range_m", "1": "raw_intensity"},
        "observed_target_rows": EVEN_ROWS.tolist(),
        "generated_target_rows": ODD_ROWS.tolist(),
        "normalization": "none",
        "exact_subset_verified": True,
    }
    return arrays, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        arrays, metadata = build_pair(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata_path = args.metadata_output or args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"saved_pair={args.output}")
    print(f"saved_metadata={metadata_path}")


if __name__ == "__main__":
    main()
