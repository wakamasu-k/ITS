"""PyTorch Dataset for traceable Waymo 32-line -> 64-line pairs."""
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
    if not dataset_role:
        raise ValueError("dataset_role must be explicit")
    with index_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS
                   if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"index is missing columns: {missing}")
        all_rows = list(reader)
    rows = [row for row in all_rows if row["dataset_role"] == dataset_role]
    if not rows:
        available = sorted({row["dataset_role"] for row in all_rows})
        raise ValueError(
            f"no rows for dataset_role={dataset_role!r}; available={available}")
    return rows

def transform_intensity(values: np.ndarray, mode: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if mode == "raw":
        return result
    if mode == "log1p":
        if np.any(result < 0):
            raise ValueError("log1p intensity transform requires non-negative values")
        return np.log1p(result).astype(np.float32)
    raise ValueError(f"unsupported intensity_transform: {mode}")

def load_pair(path: Path, intensity_transform: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "input_32", "input_valid_mask_32", "target_64",
            "target_valid_mask_64", "observed_target_rows",
            "generated_target_rows",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"pair is missing arrays: {sorted(missing)}")
        input_32 = np.asarray(data["input_32"], dtype=np.float32).copy()
        target_64 = np.asarray(data["target_64"], dtype=np.float32).copy()
        input_mask = np.asarray(data["input_valid_mask_32"], dtype=bool)
        target_mask = np.asarray(data["target_valid_mask_64"], dtype=bool)
        observed = np.asarray(data["observed_target_rows"], dtype=np.int64)
        generated = np.asarray(data["generated_target_rows"], dtype=np.int64)
    if input_32.ndim != 3 or input_32.shape[0] != 32 or input_32.shape[2] != 2:
        raise ValueError(f"input_32 must be [32,W,2], got {input_32.shape}")
    if target_64.shape != (64, input_32.shape[1], 2):
        raise ValueError(f"target_64 has invalid shape: {target_64.shape}")
    if input_mask.shape != input_32.shape[:2]:
        raise ValueError("input mask shape mismatch")
    if target_mask.shape != target_64.shape[:2]:
        raise ValueError("target mask shape mismatch")
    if not np.array_equal(observed, np.arange(0, 64, 2)):
        raise ValueError("observed target rows mismatch")
    if not np.array_equal(generated, np.arange(1, 64, 2)):
        raise ValueError("generated target rows mismatch")
    input_32[..., 1] = transform_intensity(input_32[..., 1], intensity_transform)
    target_64[..., 1] = transform_intensity(target_64[..., 1], intensity_transform)
    generated_mask = np.zeros_like(target_mask)
    generated_mask[generated] = target_mask[generated]
    return {
        "input": np.moveaxis(input_32, -1, 0).copy(),
        "target": np.moveaxis(target_64, -1, 0).copy(),
        "input_mask": input_mask.copy(),
        "target_mask": target_mask.copy(),
        "generated_mask": generated_mask,
    }

class Waymo32To64Dataset(Dataset):
    def __init__(self, index_csv: Path, *, dataset_role: str,
                 intensity_transform: str = "raw") -> None:
        self.rows = load_index(Path(index_csv), dataset_role)
        self.intensity_transform = intensity_transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        pair = load_pair(Path(row["pair_npz"]), self.intensity_transform)
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
