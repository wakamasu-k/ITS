"""PyTorch Dataset for traceable Waymo 16-line -> 32-line pairs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = (
    "dataset_role", "q_subset", "q_seg", "q_frame_index",
    "timestamp_micros", "pair_npz", "input_height", "target_height",
    "width", "channels",
)


def load_index(index_csv: Path, dataset_role: str) -> list[dict[str, str]]:
    """Load rows for one explicit role.

    Input shape: CSV index.
    Output shape: list of row dictionaries.
    """
    if not dataset_role:
        raise ValueError("dataset_role must be explicit")
    with index_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"index is missing columns: {missing}")
        rows = list(reader)
    selected = [row for row in rows if row["dataset_role"] == dataset_role]
    if not selected:
        available = sorted({row["dataset_role"] for row in rows})
        raise ValueError(f"no rows for dataset_role={dataset_role!r}; available={available}")
    return selected


def transform_intensity(values: np.ndarray, mode: str, valid_mask: np.ndarray) -> np.ndarray:
    """Apply an explicit intensity transform.

    Input shape: values and valid_mask are [H,W].
    Output shape: float32 [H,W].
    """
    result = np.asarray(values, dtype=np.float32).copy()
    mask = np.asarray(valid_mask, dtype=bool)
    if result.shape != mask.shape:
        raise ValueError("intensity and valid mask shapes differ")
    result[~mask] = 0.0
    if mode == "raw":
        return result
    if mode == "log1p":
        if np.any(result[mask] < 0):
            raise ValueError("log1p intensity requires non-negative valid values")
        return np.log1p(result).astype(np.float32)
    raise ValueError(f"unsupported intensity_transform: {mode}")


def _circular_pad(array: np.ndarray, width: int) -> np.ndarray:
    """Pad the width by repeating columns from the left edge.

    Input shape: [H,W] or [H,W,C].
    Output shape: same rank with width exactly 2688.
    """
    target_width = 2688
    if array.shape[1] > target_width:
        raise ValueError(f"width {array.shape[1]} exceeds target width {target_width}")
    pad = target_width - array.shape[1]
    if pad == 0:
        return array.copy()
    source_width = array.shape[1]
    right_indices = np.arange(pad, dtype=np.int64) % source_width
    return np.concatenate((array, np.take(array, right_indices, axis=1)), axis=1)


def load_pair(path: Path, *, intensity_transform: str = "raw",
              range_scale_m: float = 1.0) -> dict[str, np.ndarray]:
    """Load one 16→32 pair and convert it to channel-first tensors.

    Input shape: NPZ input_16 [16,W,2], target_32 [32,W,2].
    Output shape: input [2,16,2688], target [2,32,2688].
    """
    if range_scale_m <= 0:
        raise ValueError("range_scale_m must be positive")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "input_16", "input_valid_mask_16", "target_32",
            "target_valid_mask_32", "observed_target_rows",
            "generated_target_rows",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"pair is missing arrays: {sorted(missing)}")
        input_16 = np.asarray(data["input_16"], dtype=np.float32).copy()
        target_32 = np.asarray(data["target_32"], dtype=np.float32).copy()
        input_mask = np.asarray(data["input_valid_mask_16"], dtype=bool)
        target_mask = np.asarray(data["target_valid_mask_32"], dtype=bool)
        observed = np.asarray(data["observed_target_rows"], dtype=np.int64)
        generated = np.asarray(data["generated_target_rows"], dtype=np.int64)
    if input_16.ndim != 3 or input_16.shape[0] != 16 or input_16.shape[2] != 2:
        raise ValueError(f"input_16 must be [16,W,2], got {input_16.shape}")
    if target_32.shape != (32, input_16.shape[1], 2):
        raise ValueError(f"target_32 must be [32,W,2], got {target_32.shape}")
    if input_mask.shape != input_16.shape[:2] or target_mask.shape != target_32.shape[:2]:
        raise ValueError("pair mask shape mismatch")
    if not np.array_equal(observed, np.arange(0, 32, 2)):
        raise ValueError("observed target rows mismatch")
    if not np.array_equal(generated, np.arange(1, 32, 2)):
        raise ValueError("generated target rows mismatch")
    if not np.isfinite(input_16).all() or not np.isfinite(target_32).all():
        raise ValueError("pair contains NaN or Inf")

    input_16[..., 0] = input_16[..., 0] / np.float32(range_scale_m)
    target_32[..., 0] = target_32[..., 0] / np.float32(range_scale_m)
    input_16[..., 1] = transform_intensity(input_16[..., 1], intensity_transform, input_mask)
    target_32[..., 1] = transform_intensity(target_32[..., 1], intensity_transform, target_mask)

    input_16 = _circular_pad(input_16, input_16.shape[1])
    target_32 = _circular_pad(target_32, target_32.shape[1])
    input_mask = _circular_pad(input_mask, input_mask.shape[1])
    target_mask = _circular_pad(target_mask, target_mask.shape[1])
    generated_mask = np.zeros_like(target_mask)
    generated_mask[1::2] = target_mask[1::2]
    return {
        "input": np.moveaxis(input_16, -1, 0).copy(),
        "target": np.moveaxis(target_32, -1, 0).copy(),
        "input_mask": input_mask.copy(),
        "target_mask": target_mask.copy(),
        "generated_mask": generated_mask,
    }


class Waymo16To32Dataset(Dataset):
    """Dataset returning padded float32 CHW input and target tensors."""

    def __init__(self, index_csv: Path, *, dataset_role: str,
                 intensity_transform: str = "raw", range_scale_m: float = 1.0) -> None:
        self.rows = load_index(Path(index_csv), dataset_role)
        self.intensity_transform = intensity_transform
        self.range_scale_m = range_scale_m

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one sample with input [2,16,2688] and target [2,32,2688]."""
        row = self.rows[index]
        pair = load_pair(Path(row["pair_npz"]),
                          intensity_transform=self.intensity_transform,
                          range_scale_m=self.range_scale_m)
        return {
            "input": torch.from_numpy(pair["input"]),
            "target": torch.from_numpy(pair["target"]),
            "input_mask": torch.from_numpy(pair["input_mask"]),
            "target_mask": torch.from_numpy(pair["target_mask"]),
            "generated_mask": torch.from_numpy(pair["generated_mask"]),
            "segment_id": row["q_seg"],
            "frame_index": int(row["q_frame_index"]),
            "timestamp_micros": int(row["timestamp_micros"]),
            "dataset_role": row["dataset_role"],
        }
