"""PyTorch Dataset for raw Waymo 32-line GT and derived 16-line input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .waymo_input_adapter import (
    IntensityMode,
    hwc_to_chw_tensor,
    normalize_range_intensity,
    pad_width_for_tulip,
)


class Waymo16To32Dataset(Dataset):
    """Read Phase 3 raw 32-line NPY files and derive 16-line inputs.

    Each item returns ``(low, high)`` where low has shape
    ``[2,16,target_width]`` and high has shape ``[2,32,target_width]``.
    The dataset performs no model forward, loss, or training operation.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        maximum_range: float,
        intensity_mode: IntensityMode = "none",
        native_width: int = 2650,
        target_width: int = 2688,
    ) -> None:
        """Initialize a manifest-backed Dataset.

        Input shape: one output root, split directory, or manifest JSONL path.
        Output shape per item: low ``[2,16,target_width]`` and high
        ``[2,32,target_width]`` float32 tensors.

        ``maximum_range`` is required because the Waymo normalization value
        has not been fixed by this phase.  It is not stored as a code constant.
        """
        super().__init__()
        self.root = Path(root).expanduser()
        self.manifest_path = self._resolve_manifest_path(self.root)
        self.data_root = self.manifest_path.parent.parent
        self.maximum_range = float(maximum_range)
        self.intensity_mode = intensity_mode
        self.native_width = int(native_width)
        self.target_width = int(target_width)
        if self.native_width <= 0 or self.target_width < self.native_width:
            raise ValueError("target_width must be >= positive native_width.")
        # Validate normalization parameters and mode before reading samples.
        probe = np.zeros((32, 1, 2), dtype=np.float32)
        normalize_range_intensity(probe, self.maximum_range, self.intensity_mode)
        self.records = self._read_manifest(self.manifest_path)
        if not self.records:
            raise ValueError(f"Manifest contains no samples: {self.manifest_path}")

    @staticmethod
    def _resolve_manifest_path(root: Path) -> Path:
        """Resolve a manifest path from a file, split directory, or output root."""
        if root.is_file():
            manifest_path = root
        elif (root / "manifest.jsonl").is_file():
            manifest_path = root / "manifest.jsonl"
        else:
            manifest_path = root / "train" / "manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest.jsonl does not exist: {manifest_path}")
        return manifest_path.resolve()

    @staticmethod
    def _read_manifest(path: Path) -> list[dict[str, Any]]:
        """Read and validate JSONL manifest records."""
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid manifest JSON at line {line_number}: {path}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Manifest line {line_number} is not an object.")
            relative_path = record.get("relative_npy_path")
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"Manifest line {line_number} has no relative_npy_path.")
            candidate = (path.parent.parent / relative_path).resolve()
            try:
                candidate.relative_to(path.parent.parent.resolve())
            except ValueError as error:
                raise ValueError(f"Manifest path escapes data root at line {line_number}.") from error
            record["_absolute_npy_path"] = str(candidate)
            records.append(record)
        return records

    def __len__(self) -> int:
        """Return the number of manifest samples."""
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and convert one sample.

        Input shape: raw NPY ``[32,2650,2]``.
        Output shapes: low ``[2,16,2688]`` and high ``[2,32,2688]`` by default.
        """
        record = self.records[index]
        npy_path = Path(record["_absolute_npy_path"])
        raw = np.load(npy_path, allow_pickle=False)
        if raw.shape != (32, self.native_width, 2):
            raise ValueError(
                f"Expected raw sample shape [32,{self.native_width},2], got {raw.shape}: {npy_path}"
            )
        if raw.dtype != np.float32:
            raise ValueError(f"Expected float32 raw sample, got {raw.dtype}: {npy_path}")
        if not np.isfinite(raw).all():
            raise ValueError(f"Raw sample contains NaN or Inf: {npy_path}")

        normalized_high = normalize_range_intensity(raw, self.maximum_range, self.intensity_mode)
        normalized_low = normalized_high[0::2, :, :]
        high = pad_width_for_tulip(hwc_to_chw_tensor(normalized_high), self.target_width)
        low = pad_width_for_tulip(hwc_to_chw_tensor(normalized_low), self.target_width)
        if low.shape != (2, 16, self.target_width):
            raise RuntimeError(f"Unexpected low tensor shape: {tuple(low.shape)}")
        if high.shape != (2, 32, self.target_width):
            raise RuntimeError(f"Unexpected high tensor shape: {tuple(high.shape)}")
        return low.to(torch.float32), high.to(torch.float32)
