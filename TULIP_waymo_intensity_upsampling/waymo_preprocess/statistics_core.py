"""Pure NumPy statistics helpers for Waymo 32->64 data analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


PERCENTILES = (50.0, 95.0, 99.0, 99.5, 99.9)
EVEN_ROWS = np.arange(0, 64, 2, dtype=np.int32)
ODD_ROWS = np.arange(1, 64, 2, dtype=np.int32)


def valid_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(f"value/mask shape mismatch: {values.shape} != {mask.shape}")
    return values[mask]


def describe_values(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values).reshape(-1)
    finite = values[np.isfinite(values)]
    result: dict[str, Any] = {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "nan_count": int(np.count_nonzero(np.isnan(values))),
        "inf_count": int(np.count_nonzero(np.isinf(values))),
        "negative_count": int(np.count_nonzero(finite < 0)),
    }
    if finite.size == 0:
        result.update(
            {
                "min": None,
                "median": None,
                "mean": None,
                "std": None,
                "p95": None,
                "p99": None,
                "p99_5": None,
                "p99_9": None,
                "max": None,
            }
        )
        return result
    percentiles = np.percentile(finite, PERCENTILES)
    result.update(
        {
            "min": float(np.min(finite)),
            "median": float(percentiles[0]),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "p95": float(percentiles[1]),
            "p99": float(percentiles[2]),
            "p99_5": float(percentiles[3]),
            "p99_9": float(percentiles[4]),
            "max": float(np.max(finite)),
        }
    )
    return result


def frame_statistics(
    range_64: np.ndarray,
    intensity_64: np.ndarray,
    mask_64: np.ndarray,
) -> dict[str, dict[str, dict[str, Any]]]:
    expected_shape = (64, range_64.shape[1])
    for name, array in (
        ("range_64", range_64),
        ("intensity_64", intensity_64),
        ("mask_64", mask_64),
    ):
        if np.asarray(array).shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")

    groups = {
        "all_64": np.arange(64, dtype=np.int32),
        "observed_even_32": EVEN_ROWS,
        "generated_odd_32": ODD_ROWS,
    }
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for group_name, rows in groups.items():
        group_mask = np.asarray(mask_64, dtype=bool)[rows]
        result[group_name] = {
            "range_m": describe_values(
                valid_values(np.asarray(range_64)[rows], group_mask)
            ),
            "intensity": describe_values(
                valid_values(np.asarray(intensity_64)[rows], group_mask)
            ),
        }
    return result


@dataclass
class FrameBalancedSampler:
    """Keep at most N deterministic values per frame for global plots/stats."""

    values_per_frame: int
    chunks: list[np.ndarray] = field(default_factory=list)

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size <= self.values_per_frame:
            sampled = values
        else:
            indices = np.linspace(
                0,
                values.size - 1,
                self.values_per_frame,
                dtype=np.int64,
            )
            sampled = values[indices]
        self.chunks.append(sampled.astype(np.float32, copy=False))

    def array(self) -> np.ndarray:
        if not self.chunks:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(self.chunks)


def choose_frame_indices(frame_count: int, requested: int) -> list[int]:
    if frame_count <= 0 or requested <= 0:
        raise ValueError("frame_count and requested must be positive")
    count = min(frame_count, requested)
    return np.linspace(0, frame_count - 1, count, dtype=np.int32).tolist()
