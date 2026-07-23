#!/usr/bin/env python3
"""Render standalone pseudocolor maps from a verified TULIP range image.

This utility is intentionally independent from the TULIP training/evaluation
code.  It reads one NPY/NPZ/BIN array, never modifies the source data, and
writes only the requested visualization files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_array(path: Path, height: Optional[int], width: Optional[int]) -> np.ndarray:
    """Load a source array without allowing object arrays or source mutation."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError(
                    f"NPZ must contain exactly one array; found keys={archive.files}"
                )
            array = archive[archive.files[0]]
    elif suffix == ".bin":
        if height is None or width is None:
            raise ValueError("--height and --width are required for .bin input")
        values = np.fromfile(path, dtype=np.float32)
        expected = height * width * 2
        if values.size != expected:
            raise ValueError(
                f"BIN has {values.size} float32 values; expected {expected} "
                f"for [H,W,2]=[{height},{width},2]"
            )
        array = values.reshape(height, width, 2)
    else:
        raise ValueError(f"Unsupported input extension: {path.suffix}")

    if not isinstance(array, np.ndarray) or array.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {type(array).__name__} {getattr(array, 'shape', None)}")
    return np.asarray(array, dtype=np.float32)


def _to_hwc(array: np.ndarray, layout: str, range_channel: int, intensity_channel: int) -> Tuple[np.ndarray, str]:
    """Validate and convert HWC/CHW source data to HWC."""
    channels = {range_channel, intensity_channel}
    if layout == "hwc":
        if array.shape[-1] <= max(channels):
            raise ValueError(f"HWC channel index is out of bounds for shape {array.shape}")
        return array, "hwc"
    if layout == "chw":
        if array.shape[0] <= max(channels):
            raise ValueError(f"CHW channel index is out of bounds for shape {array.shape}")
        return np.moveaxis(array, 0, -1), "chw"

    hwc_possible = array.shape[-1] > max(channels)
    chw_possible = array.shape[0] > max(channels)
    if hwc_possible and not chw_possible:
        return array, "hwc"
    if chw_possible and not hwc_possible:
        return np.moveaxis(array, 0, -1), "chw"
    raise ValueError(
        "layout=auto is ambiguous; pass --layout hwc or --layout chw explicitly "
        f"for shape {array.shape}"
    )


def _percentile_bounds(values: np.ndarray) -> Tuple[float, float]:
    """Return robust 1st/99th percentile display bounds."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite valid values are available for visualization")
    low, high = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _save_map(
    values: np.ndarray,
    valid: np.ndarray,
    output_path: Path,
    title: str,
    cmap: str,
    colorbar_label: str,
    transform: Optional[str] = None,
) -> Tuple[float, float]:
    """Save one masked pseudocolor map and return its display bounds."""
    display_values = values if transform is None else np.log1p(np.maximum(values, 0.0))
    masked = np.ma.array(display_values, mask=~valid)
    vmin, vmax = _percentile_bounds(display_values[valid])

    figure, axis = plt.subplots(figsize=(16, 4), dpi=200)
    image = axis.imshow(
        masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    axis.set_xlabel("Column")
    axis.set_ylabel("Ring")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return vmin, vmax


def render(args: argparse.Namespace) -> Dict[str, Any]:
    """Render range and intensity pseudocolor maps and write metadata."""
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [
        output_dir / "tulip_range_pseudocolor.png",
        output_dir / "tulip_intensity_pseudocolor.png",
        output_dir / "visualization_metadata.json",
    ]
    if not args.overwrite:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Output exists; pass --overwrite only when replacement is intended: "
                + ", ".join(existing)
            )

    source = _load_array(input_path, args.height, args.width)
    source_shape = list(source.shape)
    hwc, detected_layout = _to_hwc(
        source, args.layout, args.range_channel, args.intensity_channel
    )
    range_values = hwc[..., args.range_channel]
    intensity_values = hwc[..., args.intensity_channel]

    range_valid = np.isfinite(range_values) & (range_values > 0)
    intensity_valid = range_valid & np.isfinite(intensity_values)
    valid = range_valid & intensity_valid
    if not np.any(valid):
        raise ValueError("No pixels are valid for both range and intensity")

    range_vmin, range_vmax = _save_map(
        range_values,
        range_valid,
        output_paths[0],
        f"{args.range_role} Range",
        "turbo",
        "Range",
    )
    intensity_vmin, intensity_vmax = _save_map(
        intensity_values,
        intensity_valid,
        output_paths[1],
        f"{args.intensity_role} Intensity",
        "magma",
        "log1p(Intensity)",
        transform="log1p",
    )

    metadata: Dict[str, Any] = {
        "sample_id": args.sample_id,
        "source_file": str(input_path),
        "source_role": args.source_role,
        "source_shape": source_shape,
        "source_dtype": str(source.dtype),
        "layout": detected_layout,
        "range_channel": args.range_channel,
        "intensity_channel": args.intensity_channel,
        "range_role": args.range_role,
        "intensity_role": args.intensity_role,
        "range_min_valid": float(np.min(range_values[range_valid])),
        "range_max_valid": float(np.max(range_values[range_valid])),
        "range_display_vmin": range_vmin,
        "range_display_vmax": range_vmax,
        "intensity_min_valid": float(np.min(intensity_values[intensity_valid])),
        "intensity_max_valid": float(np.max(intensity_values[intensity_valid])),
        "intensity_display_transform": "log1p(max(intensity, 0))",
        "intensity_display_vmin": intensity_vmin,
        "intensity_display_vmax": intensity_vmax,
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "invalid_pixel_count": int(valid.size - np.count_nonzero(valid)),
        "checkpoint_path": args.checkpoint_path,
        "generated_files": [path.name for path in output_paths],
    }
    with output_paths[2].open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input NPY, NPZ, or BIN")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--layout", choices=("auto", "hwc", "chw"), default="auto")
    parser.add_argument("--range-channel", type=int, default=0)
    parser.add_argument("--intensity-channel", type=int, default=1)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--source-role", default="unknown")
    parser.add_argument("--range-role", default="unknown")
    parser.add_argument("--intensity-role", default="unknown")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Run the standalone renderer."""
    metadata = render(build_parser().parse_args())
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
