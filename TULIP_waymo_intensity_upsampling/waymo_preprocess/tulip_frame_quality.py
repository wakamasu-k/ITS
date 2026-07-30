"""Audit exported Waymo arrays and derive TULIP 16-to-32 pairs."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from waymo_preprocess.shimizu_batch_export import ExportItem

ARRAY_NAMES = (
    "range_64", "intensity_64", "valid_mask_64",
    "range_32", "intensity_32", "valid_mask_32", "ring_ids_32",
)

def valid_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        raise ValueError("no valid pixels")
    if not np.isfinite(selected).all():
        raise ValueError("non-finite valid pixels")
    return {
        "min": float(np.min(selected)),
        "median": float(np.median(selected)),
        "p99_5": float(np.percentile(selected, 99.5)),
        "max": float(np.max(selected)),
    }

def audit_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    missing = [name for name in ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing arrays: {missing}")
    range_64 = np.asarray(arrays["range_64"])
    intensity_64 = np.asarray(arrays["intensity_64"])
    mask_64 = np.asarray(arrays["valid_mask_64"])
    range_32 = np.asarray(arrays["range_32"])
    intensity_32 = np.asarray(arrays["intensity_32"])
    mask_32 = np.asarray(arrays["valid_mask_32"])
    rings = np.asarray(arrays["ring_ids_32"])
    if range_64.ndim != 2 or range_64.shape[0] != 64:
        raise ValueError(f"invalid range_64 shape: {range_64.shape}")
    width = range_64.shape[1]
    if intensity_64.shape != (64, width) or mask_64.shape != (64, width):
        raise ValueError("64-line array shapes differ")
    if range_32.shape != (32, width):
        raise ValueError(f"invalid range_32 shape: {range_32.shape}")
    if intensity_32.shape != (32, width) or mask_32.shape != (32, width):
        raise ValueError("32-line array shapes differ")
    expected_rings = np.arange(0, 64, 2, dtype=np.int32)
    if not np.array_equal(rings, expected_rings):
        raise ValueError("ring_ids_32 is not the even-ring sequence")
    if not np.array_equal(range_32, range_64[::2]):
        raise ValueError("range_32 is not an exact even-ring subset")
    if not np.array_equal(intensity_32, intensity_64[::2]):
        raise ValueError("intensity_32 is not an exact even-ring subset")
    if not np.array_equal(mask_32, mask_64[::2]):
        raise ValueError("valid_mask_32 is not an exact even-ring subset")
    if mask_64.dtype != np.bool_ or mask_32.dtype != np.bool_:
        raise ValueError("valid masks must be boolean")
    return {
        "width": int(width),
        "valid_count_64": int(mask_64.sum()),
        "valid_count_32": int(mask_32.sum()),
        "range_32": valid_summary(range_32, mask_32),
        "intensity_32": valid_summary(intensity_32, mask_32),
    }

def audit_metadata(metadata: dict[str, Any], item: ExportItem) -> None:
    if metadata.get("segment_id") != item.segment_id:
        raise ValueError("metadata segment mismatch")
    if int(metadata.get("source_frame_index", -1)) != item.frame_index:
        raise ValueError("metadata frame index mismatch")
    if int(metadata.get("timestamp_micros", -1)) != item.timestamp_micros:
        raise ValueError("metadata timestamp mismatch")

def audit_frame(npz_path: Path, metadata_path: Path,
                item: ExportItem) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    audit_metadata(metadata, item)
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    return audit_arrays(arrays)

def build_tulip_16_32(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    audit_arrays(arrays)
    rings_32 = np.asarray(arrays["ring_ids_32"], dtype=np.int32)
    return {
        "range_16": np.asarray(arrays["range_32"])[::2].astype(np.float32),
        "intensity_16": np.asarray(arrays["intensity_32"])[::2].astype(np.float32),
        "valid_mask_16": np.asarray(arrays["valid_mask_32"])[::2].astype(bool),
        "ring_ids_16": rings_32[::2].copy(),
        "range_32": np.asarray(arrays["range_32"]).astype(np.float32),
        "intensity_32": np.asarray(arrays["intensity_32"]).astype(np.float32),
        "valid_mask_32": np.asarray(arrays["valid_mask_32"]).astype(bool),
        "ring_ids_32": rings_32.copy(),
    }
