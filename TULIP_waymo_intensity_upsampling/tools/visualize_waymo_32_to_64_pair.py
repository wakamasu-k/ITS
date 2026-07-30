#!/usr/bin/env python3
"""Create fixed-scale visual checks for a Waymo 32-line -> 64-line pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def expand_rows(
    values_32: np.ndarray,
    mask_32: np.ndarray,
    target_rows: np.ndarray,
    target_height: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Place packed input rows at their original target row positions."""
    values_32 = np.asarray(values_32)
    mask_32 = np.asarray(mask_32, dtype=bool)
    target_rows = np.asarray(target_rows, dtype=np.int32)
    if values_32.ndim != 2 or mask_32.shape != values_32.shape:
        raise ValueError("values_32 and mask_32 must share shape [H,W]")
    if target_rows.shape != (values_32.shape[0],):
        raise ValueError("target_rows length must equal input height")
    if np.any(target_rows < 0) or np.any(target_rows >= target_height):
        raise ValueError("target_rows contains an out-of-range row")
    if np.unique(target_rows).size != target_rows.size:
        raise ValueError("target_rows must not contain duplicates")

    expanded = np.zeros((target_height, values_32.shape[1]), dtype=values_32.dtype)
    expanded_mask = np.zeros(expanded.shape, dtype=bool)
    expanded[target_rows] = values_32
    expanded_mask[target_rows] = mask_32
    return expanded, expanded_mask


def masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    result[~np.asarray(mask, dtype=bool)] = np.nan
    return result


def make_colormap(name: str):
    colormap = plt.get_cmap(name).copy()
    colormap.set_bad("black")
    colormap.set_under("black")
    return colormap


def save_scalar_map(
    *,
    values: np.ndarray,
    mask: np.ndarray,
    output: Path,
    title: str,
    colorbar_label: str,
    value_min: float,
    value_max: float,
    colormap_name: str,
) -> None:
    figure, axis = plt.subplots(figsize=(16, 3.4), constrained_layout=True)
    image = axis.imshow(
        masked_values(values, mask),
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap=make_colormap(colormap_name),
        vmin=value_min,
        vmax=value_max,
    )
    axis.set_title(title)
    axis.set_xlabel("Azimuth column (native Waymo order)")
    axis.set_ylabel("LiDAR row")
    colorbar = figure.colorbar(image, ax=axis, pad=0.01)
    colorbar.set_label(colorbar_label)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def save_mask(
    mask: np.ndarray,
    output: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(16, 3.4), constrained_layout=True)
    image = axis.imshow(
        np.asarray(mask, dtype=np.uint8),
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axis.set_title(title)
    axis.set_xlabel("Azimuth column (native Waymo order)")
    axis.set_ylabel("LiDAR row")
    colorbar = figure.colorbar(image, ax=axis, pad=0.01, ticks=[0, 1])
    colorbar.set_label("0 = invalid, 1 = valid")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def save_comparison(
    *,
    range_64: np.ndarray,
    intensity_64: np.ndarray,
    mask_64: np.ndarray,
    range_expanded: np.ndarray,
    intensity_expanded: np.ndarray,
    mask_expanded: np.ndarray,
    output: Path,
    range_max: float,
    intensity_max: float,
) -> None:
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(18, 10),
        constrained_layout=True,
    )
    specifications = (
        (
            range_64,
            mask_64,
            "64-line GT range",
            "turbo",
            0.0,
            range_max,
            "Range [m]",
        ),
        (
            range_expanded,
            mask_expanded,
            "32-line input on original 64 rows",
            "turbo",
            0.0,
            range_max,
            "Range [m]",
        ),
        (
            intensity_64,
            mask_64,
            "64-line GT raw intensity",
            "viridis",
            0.0,
            intensity_max,
            "Raw intensity (fixed display clip)",
        ),
        (
            intensity_expanded,
            mask_expanded,
            "32-line input intensity on original 64 rows",
            "viridis",
            0.0,
            intensity_max,
            "Raw intensity (fixed display clip)",
        ),
        (
            mask_64.astype(np.float32),
            np.ones_like(mask_64, dtype=bool),
            "64-line GT valid mask",
            "gray",
            0.0,
            1.0,
            "Validity",
        ),
        (
            mask_expanded.astype(np.float32),
            np.ones_like(mask_expanded, dtype=bool),
            "32-line valid mask on original 64 rows",
            "gray",
            0.0,
            1.0,
            "Validity",
        ),
    )
    for axis, specification in zip(axes.flat, specifications):
        values, mask, title, cmap, minimum, maximum, label = specification
        image = axis.imshow(
            masked_values(values, mask),
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=make_colormap(cmap),
            vmin=minimum,
            vmax=maximum,
        )
        axis.set_title(title)
        axis.set_xlabel("Azimuth column")
        axis.set_ylabel("LiDAR row")
        colorbar = figure.colorbar(image, ax=axis, pad=0.01)
        colorbar.set_label(label)
    figure.suptitle(
        "Waymo TOP LiDAR 32-line input -> 64-line GT (same frame)",
        fontsize=15,
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)


def finite_valid_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {"count": 0}
    percentiles = np.percentile(selected, [50.0, 95.0, 99.0, 99.5, 99.9])
    return {
        "count": int(selected.size),
        "min": float(np.min(selected)),
        "median": float(percentiles[0]),
        "p95": float(percentiles[1]),
        "p99": float(percentiles[2]),
        "p99_5": float(percentiles[3]),
        "p99_9": float(percentiles[4]),
        "max": float(np.max(selected)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--range-max", type=float, default=75.0)
    parser.add_argument("--intensity-max", type=float, default=0.75)
    parser.add_argument("--intensity-log-max", type=float, default=22016.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.range_max <= 0 or args.intensity_max <= 0:
        raise ValueError("range-max and intensity-max must be positive")
    if args.intensity_log_max <= 0:
        raise ValueError("intensity-log-max must be positive")

    with np.load(args.input, allow_pickle=False) as data:
        input_32 = np.asarray(data["input_32"], dtype=np.float32)
        input_mask = np.asarray(data["input_valid_mask_32"], dtype=bool)
        target_64 = np.asarray(data["target_64"], dtype=np.float32)
        target_mask = np.asarray(data["target_valid_mask_64"], dtype=bool)
        observed_rows = np.asarray(data["observed_target_rows"], dtype=np.int32)
        generated_rows = np.asarray(data["generated_target_rows"], dtype=np.int32)

    if input_32.ndim != 3 or input_32.shape[0] != 32 or input_32.shape[2] != 2:
        raise ValueError(f"input_32 must be [32,W,2], got {input_32.shape}")
    if target_64.shape != (64, input_32.shape[1], 2):
        raise ValueError(
            f"target_64 must be [64,{input_32.shape[1]},2], got {target_64.shape}"
        )

    range_32 = input_32[..., 0]
    intensity_32 = input_32[..., 1]
    range_64 = target_64[..., 0]
    intensity_64 = target_64[..., 1]
    range_expanded, expanded_mask = expand_rows(
        range_32, input_mask, observed_rows
    )
    intensity_expanded, intensity_expanded_mask = expand_rows(
        intensity_32, input_mask, observed_rows
    )
    if not np.array_equal(expanded_mask, intensity_expanded_mask):
        raise ValueError("expanded range/intensity masks differ")
    if not np.array_equal(range_expanded[observed_rows], range_64[observed_rows]):
        raise ValueError("expanded input range does not match observed GT rows")
    if not np.array_equal(
        intensity_expanded[observed_rows],
        intensity_64[observed_rows],
    ):
        raise ValueError("expanded input intensity does not match observed GT rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scalar_jobs = (
        (
            range_64,
            target_mask,
            "range_64_gt.png",
            "64-line GT range",
            "Range [m]",
            0.0,
            args.range_max,
            "turbo",
        ),
        (
            range_32,
            input_mask,
            "range_32_input.png",
            "32-line input range",
            "Range [m]",
            0.0,
            args.range_max,
            "turbo",
        ),
        (
            range_expanded,
            expanded_mask,
            "range_32_on_64_rows.png",
            "32-line range on original 64-row positions",
            "Range [m]",
            0.0,
            args.range_max,
            "turbo",
        ),
        (
            intensity_64,
            target_mask,
            "intensity_64_gt_fixed.png",
            "64-line GT raw intensity (fixed scale)",
            "Raw intensity",
            0.0,
            args.intensity_max,
            "viridis",
        ),
        (
            intensity_32,
            input_mask,
            "intensity_32_input_fixed.png",
            "32-line input raw intensity (fixed scale)",
            "Raw intensity",
            0.0,
            args.intensity_max,
            "viridis",
        ),
        (
            intensity_expanded,
            expanded_mask,
            "intensity_32_on_64_rows_fixed.png",
            "32-line intensity on original 64-row positions",
            "Raw intensity",
            0.0,
            args.intensity_max,
            "viridis",
        ),
        (
            np.log1p(np.clip(intensity_64, 0.0, None)),
            target_mask,
            "intensity_64_gt_log.png",
            "64-line GT intensity log1p view",
            "log1p(raw intensity)",
            0.0,
            float(np.log1p(args.intensity_log_max)),
            "magma",
        ),
    )
    for (
        values,
        mask,
        filename,
        title,
        label,
        minimum,
        maximum,
        colormap,
    ) in scalar_jobs:
        save_scalar_map(
            values=values,
            mask=mask,
            output=args.output_dir / filename,
            title=title,
            colorbar_label=label,
            value_min=minimum,
            value_max=maximum,
            colormap_name=colormap,
        )

    save_mask(
        target_mask,
        args.output_dir / "valid_mask_64.png",
        "64-line GT valid mask",
    )
    save_mask(
        input_mask,
        args.output_dir / "valid_mask_32.png",
        "32-line input valid mask",
    )
    save_comparison(
        range_64=range_64,
        intensity_64=intensity_64,
        mask_64=target_mask,
        range_expanded=range_expanded,
        intensity_expanded=intensity_expanded,
        mask_expanded=expanded_mask,
        output=args.output_dir / "comparison_32_to_64.png",
        range_max=args.range_max,
        intensity_max=args.intensity_max,
    )

    metadata = {
        "source_pair": str(args.input.resolve()),
        "range_display_m": [0.0, float(args.range_max)],
        "intensity_fixed_display": [0.0, float(args.intensity_max)],
        "intensity_log_display": [
            0.0,
            float(np.log1p(args.intensity_log_max)),
        ],
        "invalid_color": "black",
        "row_origin": "upper",
        "horizontal_order": "native Waymo azimuth column order; not flipped",
        "interpolation": "nearest",
        "observed_target_rows": observed_rows.tolist(),
        "generated_target_rows": generated_rows.tolist(),
        "exact_observed_range_match": True,
        "exact_observed_intensity_match": True,
        "range_64_valid_stats": finite_valid_stats(range_64, target_mask),
        "intensity_64_valid_stats": finite_valid_stats(
            intensity_64, target_mask
        ),
        "output_files": sorted(
            path.name for path in args.output_dir.glob("*.png")
        ),
    }
    metadata_path = args.output_dir / "visualization_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"saved_visualization_dir={args.output_dir}")


if __name__ == "__main__":
    main()
