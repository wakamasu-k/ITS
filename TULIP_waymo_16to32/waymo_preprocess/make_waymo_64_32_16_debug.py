"""Create one-frame native Waymo 64/32/16 debug artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from waymo_preprocess.waymo_range_image import (
    extract_range_intensity,
    make_16line_input,
    make_32line_gt,
)
from waymo_preprocess.waymo_tfrecord_io import get_frame, get_top_first_return


def _nonfinite_counts(array: np.ndarray) -> Dict[str, int]:
    """Count NaN and Inf values in a numeric array.

    Args:
        array: Numeric array of any shape.

    Returns:
        Counts for nan, positive infinity, negative infinity, and total
        infinity.
    """
    values = np.asarray(array)
    return {
        "nan": int(np.isnan(values).sum()),
        "pos_inf": int(np.isposinf(values).sum()),
        "neg_inf": int(np.isneginf(values).sum()),
        "total_inf": int(np.isinf(values).sum()),
    }


def _masked_stats(values: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Calculate valid-pixel statistics for one float image.

    Args:
        values: Float image [H,W].
        mask: Boolean valid-pixel mask [H,W].

    Returns:
        JSON-serializable shape, dtype, valid count/rate, min, max, and mean.
        Min/max/mean are calculated from valid pixels only.
    """
    values_array = np.asarray(values, dtype=np.float32)
    mask_array = np.asarray(mask, dtype=bool)
    if values_array.shape != mask_array.shape:
        raise ValueError(f"values shape {values_array.shape} != mask shape {mask_array.shape}.")
    valid_values = values_array[mask_array]
    result: Dict[str, Any] = {
        "shape": [int(v) for v in values_array.shape],
        "dtype": str(values_array.dtype),
        "valid_count": int(valid_values.size),
        "valid_rate": float(mask_array.mean()),
        "min": None,
        "max": None,
        "mean": None,
    }
    if valid_values.size:
        result["min"] = float(valid_values.min())
        result["max"] = float(valid_values.max())
        result["mean"] = float(valid_values.mean())
    return result


def _visualize_float_image(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a confirmation-only uint8 PNG using valid-pixel percentiles.

    Args:
        values: Float image [H,W].
        mask: Boolean valid-pixel mask [H,W].

    Returns:
        uint8 grayscale image [H,W] with values in [0,255].

    Raises:
        ValueError: If no valid finite pixels exist.
    """
    values_array = np.asarray(values, dtype=np.float32)
    mask_array = np.asarray(mask, dtype=bool)
    valid = mask_array & np.isfinite(values_array)
    if not np.any(valid):
        raise ValueError("Cannot visualize an image with no valid finite pixels.")
    low, high = np.percentile(values_array[valid], [1.0, 99.0]).astype(np.float32)
    output = np.zeros(values_array.shape, dtype=np.uint8)
    if high <= low:
        output[valid] = 255
        return output
    normalized = (values_array - low) / (high - low)
    output[valid] = np.clip(normalized[valid] * 255.0, 0.0, 255.0).astype(np.uint8)
    return output


def _write_png(path: Path, image: np.ndarray) -> None:
    """Write a 2D uint8 grayscale PNG and raise on failure.

    Args:
        path: Destination PNG path.
        image: uint8 grayscale image [H,W].

    Raises:
        ValueError: If shape or dtype is invalid.
        OSError: If OpenCV cannot write the PNG.
    """
    image_array = np.asarray(image)
    if image_array.ndim != 2 or image_array.dtype != np.uint8:
        raise ValueError(f"PNG must be 2D uint8, got {image_array.shape}/{image_array.dtype}.")
    if not cv2.imwrite(str(path), image_array):
        raise OSError(f"Failed to write PNG: {path}")


def _assert_output_is_not_read_only_dataset(path: Path) -> None:
    """Reject output paths inside the read-only /mnt/w/32line dataset.

    Args:
        path: Requested output directory.

    Raises:
        PermissionError: If path is inside /mnt/w/32line.
    """
    output_path = path.expanduser().absolute()
    protected_root = Path("/mnt/w/32line").absolute()
    try:
        output_path.relative_to(protected_root)
    except ValueError:
        return
    raise PermissionError(f"Refusing to write inside read-only dataset root: {output_path}")


def process_one_frame(
    tfrecord_path: str | Path,
    frame_index: int,
    out_dir: str | Path,
) -> Dict[str, Any]:
    """Read one frame and save native 64/32/16 debug artifacts.

    Args:
        tfrecord_path: Uncompressed Waymo TFRecord path.
        frame_index: Zero-based frame index.
        out_dir: Output directory outside /mnt/w/32line.

    Returns:
        Metadata dictionary written to metadata.json.

    Raises:
        FileNotFoundError, IndexError, KeyError, ValueError, PermissionError,
        or OSError if validation or I/O fails.
    """
    output_path = Path(out_dir).expanduser()
    _assert_output_is_not_read_only_dataset(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    frame = get_frame(tfrecord_path, frame_index)
    native = get_top_first_return(frame)
    if native.ndim != 3 or native.shape[0] != 64:
        raise ValueError(f"Expected native TOP shape [64,W,C], got {native.shape}.")
    if native.shape[2] < 2:
        raise ValueError(f"Expected range and intensity channels, got {native.shape}.")

    raw_range = native[..., 0]
    raw_intensity = native[..., 1]
    range_64, intensity_64, mask_64 = extract_range_intensity(native)
    range_32, intensity_32, mask_32, ring_ids_32 = make_32line_gt(
        range_64, intensity_64, mask_64
    )
    range_16, intensity_16, mask_16, ring_ids_16 = make_16line_input(
        range_32, intensity_32, mask_32
    )

    arrays = {
        "range_64": range_64,
        "intensity_64": intensity_64,
        "mask_64": mask_64,
        "range_32": range_32,
        "intensity_32": intensity_32,
        "mask_32": mask_32,
        "range_16": range_16,
        "intensity_16": intensity_16,
        "mask_16": mask_16,
        "ring_ids_32": ring_ids_32,
        "ring_ids_16": ring_ids_16,
    }
    npz_path = output_path / "waymo_64_32_16.npz"
    np.savez_compressed(npz_path, **arrays)

    metadata: Dict[str, Any] = {
        "tfrecord": str(Path(tfrecord_path).expanduser()),
        "frame_index": int(frame_index),
        "context_name": str(frame.context.name),
        "timestamp_micros": int(frame.timestamp_micros),
        "native_shape": [int(v) for v in native.shape],
        "return_index": 0,
        "channel_semantics": {"channel_0": "range", "channel_1": "intensity"},
        "invalid_rule": "range <= 0 or non-finite range/intensity",
        "ring_ids_32": [int(v) for v in ring_ids_32],
        "ring_ids_16": [int(v) for v in ring_ids_16],
        "stats": {
            "range_64": _masked_stats(range_64, mask_64),
            "range_32": _masked_stats(range_32, mask_32),
            "range_16": _masked_stats(range_16, mask_16),
            "intensity_64": _masked_stats(intensity_64, mask_64),
            "intensity_32": _masked_stats(intensity_32, mask_32),
            "intensity_16": _masked_stats(intensity_16, mask_16),
        },
        "nan_inf_counts": {
            "native_range": _nonfinite_counts(raw_range),
            "native_intensity": _nonfinite_counts(raw_intensity),
        },
        "npz": str(npz_path),
        "png_note": "PNG is confirmation-only percentile visualization; NPZ preserves float32 values.",
    }
    with (output_path / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    for name, values, mask in (
        ("range_64", range_64, mask_64),
        ("range_32", range_32, mask_32),
        ("range_16", range_16, mask_16),
        ("intensity_64", intensity_64, mask_64),
        ("intensity_32", intensity_32, mask_32),
        ("intensity_16", intensity_16, mask_16),
    ):
        _write_png(output_path / f"{name}.png", _visualize_float_image(values, mask))
    for name, mask in (
        ("mask_64", mask_64),
        ("mask_32", mask_32),
        ("mask_16", mask_16),
    ):
        _write_png(output_path / f"{name}.png", mask.astype(np.uint8) * 255)
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    """Build the Phase 1 command-line parser.

    Returns:
        Parser accepting --tfrecord, --frame-index, and --out-dir.
    """
    parser = argparse.ArgumentParser(
        description="Extract one Waymo TOP first-return frame into 64/32/16 debug data."
    )
    parser.add_argument("--tfrecord", required=True, help="Waymo TFRecord path.")
    parser.add_argument("--frame-index", type=int, default=0, help="Zero-based frame index.")
    parser.add_argument("--out-dir", required=True, help="Output directory outside /mnt/w/32line.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the one-frame Phase 1 conversion CLI.

    Args:
        argv: Optional argument sequence; None uses sys.argv[1:].
    """
    args = _build_parser().parse_args(argv)
    metadata = process_one_frame(args.tfrecord, args.frame_index, args.out_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
