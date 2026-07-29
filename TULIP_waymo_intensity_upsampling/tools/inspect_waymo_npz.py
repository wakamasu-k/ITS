#!/usr/bin/env python3
"""Validate one exported Waymo range/intensity frame without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _finite_stats(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    percentiles = np.percentile(finite, [0, 1, 5, 50, 95, 99, 100])
    return {
        "count": int(finite.size),
        "min": float(percentiles[0]),
        "p01": float(percentiles[1]),
        "p05": float(percentiles[2]),
        "median": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "p99": float(percentiles[5]),
        "max": float(percentiles[6]),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def inspect_arrays(
    range_image: np.ndarray,
    intensity_image: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return traceable statistics and fail early on incompatible shapes."""
    range_image = np.asarray(range_image)
    intensity_image = np.asarray(intensity_image)
    if range_image.ndim != 2 or intensity_image.ndim != 2:
        raise ValueError("range and intensity must both be 2-D [H, W]")
    if range_image.shape != intensity_image.shape:
        raise ValueError(
            f"shape mismatch: range={range_image.shape}, "
            f"intensity={intensity_image.shape}"
        )

    if valid_mask is None:
        valid_mask = range_image > 0
        mask_source = "range > 0"
    else:
        valid_mask = np.asarray(valid_mask)
        if valid_mask.shape != range_image.shape:
            raise ValueError(
                f"shape mismatch: valid_mask={valid_mask.shape}, "
                f"range={range_image.shape}"
            )
        valid_mask = valid_mask.astype(bool)
        mask_source = "valid_mask"

    valid_range = range_image[valid_mask]
    valid_intensity = intensity_image[valid_mask]
    invalid_mask = ~valid_mask
    return {
        "shape": [int(v) for v in range_image.shape],
        "mask_source": mask_source,
        "pixel_count": int(range_image.size),
        "valid_count": int(np.count_nonzero(valid_mask)),
        "valid_ratio": float(np.mean(valid_mask)),
        "quality": {
            "range_nan_count": int(np.count_nonzero(np.isnan(range_image))),
            "range_inf_count": int(np.count_nonzero(np.isinf(range_image))),
            "intensity_nan_count": int(np.count_nonzero(np.isnan(intensity_image))),
            "intensity_inf_count": int(np.count_nonzero(np.isinf(intensity_image))),
            "negative_valid_range_count": int(
                np.count_nonzero(valid_range < 0)
            ),
            "nonzero_range_outside_mask_count": int(
                np.count_nonzero(range_image[invalid_mask] != 0)
            ),
            "nonzero_intensity_outside_mask_count": int(
                np.count_nonzero(intensity_image[invalid_mask] != 0)
            ),
        },
        "valid_range_m": _finite_stats(valid_range),
        "valid_intensity": _finite_stats(valid_intensity),
    }


def inspect_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        keys = sorted(data.files)
        missing = {"range", "intensity"} - set(keys)
        if missing:
            raise KeyError(f"missing NPZ arrays: {sorted(missing)}; found={keys}")
        valid_mask = data["valid_mask"] if "valid_mask" in data else None
        report = inspect_arrays(data["range"], data["intensity"], valid_mask)
    report["input"] = str(path.resolve())
    report["npz_keys"] = keys
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_npz(args.input)
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"saved_report={args.output}")


if __name__ == "__main__":
    main()
