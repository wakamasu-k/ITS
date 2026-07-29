"""KITTI XYZI point clouds <-> TULIP range-image conversion utilities.

The original project implemented projection and unprojection in separate
scripts.  Keeping the geometry here makes the row angles, units, invalid-pixel
rules, and collision handling identical in data generation and diagnostics.

Range-image convention
----------------------
* channel 0: radial LiDAR range
* channel 1: reflectance / intensity
* invalid pixel: both channels are zero
* image layout: ``[H, W, 2]`` unless explicitly converted by the caller
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


KITTI_ROWS = 64
KITTI_COLS = 1024
KITTI_VERTICAL_MIN_DEG = -24.8
KITTI_VERTICAL_MAX_DEG = 2.0
KITTI_MAX_RANGE_M = 80.0
DEFAULT_LOW_ROW_INDICES = tuple(range(0, KITTI_ROWS, 4))


@dataclass(frozen=True)
class ProjectionStats:
    """Counts describing which raw points survived range-image projection."""

    raw_point_count: int
    finite_point_count: int
    in_range_point_count: int
    in_fov_point_count: int
    valid_pixel_count: int
    collision_count: int
    selected_source_indices: np.ndarray

    def as_json_dict(self) -> dict[str, int]:
        """Return JSON-safe scalar statistics (source indices stay in memory)."""

        return {
            "raw_point_count": self.raw_point_count,
            "finite_point_count": self.finite_point_count,
            "in_range_point_count": self.in_range_point_count,
            "in_fov_point_count": self.in_fov_point_count,
            "valid_pixel_count": self.valid_pixel_count,
            "collision_count": self.collision_count,
        }


def load_kitti_xyzi(path: str | Path) -> np.ndarray:
    """Load a KITTI ``.bin`` file as float32 ``[N, 4]`` XYZI."""

    path = Path(path)
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4:
        raise ValueError(
            f"KITTI point file must contain float32 XYZI tuples: {path}"
        )
    return raw.reshape(-1, 4)


def _vertical_angles_deg(
    row_indices: np.ndarray,
    full_rows: int = KITTI_ROWS,
) -> np.ndarray:
    """Map original full-resolution row indices to beam-center angles."""

    if full_rows < 2:
        raise ValueError(f"full_rows must be >= 2, got {full_rows}")
    resolution = (
        KITTI_VERTICAL_MAX_DEG - KITTI_VERTICAL_MIN_DEG
    ) / (full_rows - 1)
    return KITTI_VERTICAL_MIN_DEG + row_indices * resolution


def project_xyzi_to_range_image(
    points_xyzi: np.ndarray,
    *,
    rows: int = KITTI_ROWS,
    cols: int = KITTI_COLS,
    max_range_m: float = KITTI_MAX_RANGE_M,
) -> tuple[np.ndarray, np.ndarray, ProjectionStats]:
    """Project XYZI points into a range image using nearest-point z-buffering.

    Returns ``(range_image, source_index_image, stats)``.  The source-index
    image records which raw point produced each valid pixel and uses ``-1`` for
    invalid pixels.  This trace is essential for quantitative round-trip tests.
    """

    points = np.asarray(points_xyzi, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"Expected [N,4+] XYZI array, got {points.shape}")
    if rows < 2 or cols < 1 or max_range_m <= 0:
        raise ValueError(
            f"Invalid projection geometry: rows={rows}, cols={cols}, "
            f"max_range_m={max_range_m}"
        )

    xyz = points[:, :3].astype(np.float64)
    intensity = points[:, 3].astype(np.float64)
    radial_range = np.linalg.norm(xyz, axis=1)

    finite = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(intensity)
        & np.isfinite(radial_range)
    )
    in_range = finite & (radial_range > 0.0) & (radial_range <= max_range_m)

    vertical_deg = np.degrees(
        np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))
    )
    vertical_res_deg = (
        KITTI_VERTICAL_MAX_DEG - KITTI_VERTICAL_MIN_DEG
    ) / (rows - 1)
    row = np.rint(
        (vertical_deg - KITTI_VERTICAL_MIN_DEG) / vertical_res_deg
    ).astype(np.int64)

    # Keep this azimuth convention compatible with the existing TULIP KITTI
    # preprocessing: atan2(x, y), 90 degrees at the center column.
    horizontal_deg = np.degrees(np.arctan2(xyz[:, 0], xyz[:, 1]))
    horizontal_res_deg = 360.0 / cols
    col = (
        -np.trunc((horizontal_deg - 90.0) / horizontal_res_deg).astype(np.int64)
        + cols // 2
    ) % cols

    in_fov = in_range & (row >= 0) & (row < rows)
    candidates = np.flatnonzero(in_fov)

    range_image = np.zeros((rows, cols, 2), dtype=np.float32)
    source_index_image = np.full((rows, cols), -1, dtype=np.int64)
    if candidates.size:
        linear = row[candidates] * cols + col[candidates]

        # Sort by pixel first and range second.  The first entry for every
        # pixel is therefore the physically closest return.
        order = np.lexsort((radial_range[candidates], linear))
        sorted_candidates = candidates[order]
        sorted_linear = linear[order]
        keep = np.ones(sorted_linear.size, dtype=bool)
        keep[1:] = sorted_linear[1:] != sorted_linear[:-1]
        selected = sorted_candidates[keep]

        selected_row = row[selected]
        selected_col = col[selected]
        range_image[selected_row, selected_col, 0] = radial_range[selected]
        range_image[selected_row, selected_col, 1] = intensity[selected]
        source_index_image[selected_row, selected_col] = selected
    else:
        selected = np.empty(0, dtype=np.int64)

    stats = ProjectionStats(
        raw_point_count=int(points.shape[0]),
        finite_point_count=int(finite.sum()),
        in_range_point_count=int(in_range.sum()),
        in_fov_point_count=int(in_fov.sum()),
        valid_pixel_count=int(selected.size),
        collision_count=int(candidates.size - selected.size),
        selected_source_indices=selected,
    )
    return range_image, source_index_image, stats


def split_range_intensity(
    image: np.ndarray,
    *,
    expected_rows: int,
    expected_cols: int = KITTI_COLS,
) -> tuple[np.ndarray, np.ndarray]:
    """Strictly extract range and intensity from HWC or CHW two-channel data."""

    array = np.asarray(image)
    if array.shape == (expected_rows, expected_cols, 2):
        return array[..., 0], array[..., 1]
    if array.shape == (2, expected_rows, expected_cols):
        return array[0], array[1]
    raise ValueError(
        "Expected range image shape "
        f"({expected_rows},{expected_cols},2) or "
        f"(2,{expected_rows},{expected_cols}), got {array.shape}"
    )


def unproject_range_image_to_xyzi(
    image: np.ndarray,
    *,
    row_indices: Sequence[int] | None = None,
    full_rows: int = KITTI_ROWS,
    cols: int = KITTI_COLS,
    range_unit: str = "meter",
    max_range_m: float = KITTI_MAX_RANGE_M,
) -> np.ndarray:
    """Convert a two-channel range image back to a row-major XYZI point cloud.

    ``row_indices`` maps each input row to its original full-resolution row.
    Passing ``[0,4,...,60]`` for a 16-row input prevents the former error where
    those rows were incorrectly spread over the entire vertical field of view.
    """

    array = np.asarray(image)
    input_rows = array.shape[0] if array.ndim == 3 and array.shape[-1] == 2 else (
        array.shape[1] if array.ndim == 3 and array.shape[0] == 2 else -1
    )
    if input_rows < 1:
        raise ValueError(f"Cannot determine range-image rows from {array.shape}")

    range_values, intensity = split_range_intensity(
        array, expected_rows=input_rows, expected_cols=cols
    )
    range_values = range_values.astype(np.float64)
    intensity = intensity.astype(np.float64)

    if range_unit == "normalized":
        radial_range = range_values * max_range_m
    elif range_unit == "meter":
        radial_range = range_values
    else:
        raise ValueError(
            f"range_unit must be 'meter' or 'normalized', got {range_unit!r}"
        )

    if row_indices is None:
        if input_rows != full_rows:
            raise ValueError(
                "row_indices are required when input rows differ from "
                f"full_rows ({input_rows} != {full_rows})"
            )
        original_rows = np.arange(full_rows, dtype=np.float64)
    else:
        original_rows = np.asarray(row_indices, dtype=np.float64)
        if original_rows.shape != (input_rows,):
            raise ValueError(
                f"row_indices must contain {input_rows} entries, "
                f"got {original_rows.shape}"
            )
        if np.any(original_rows < 0) or np.any(original_rows >= full_rows):
            raise ValueError(
                f"row_indices must lie in [0,{full_rows - 1}]: {original_rows}"
            )

    row_grid, col_grid = np.indices((input_rows, cols))
    vertical_deg = _vertical_angles_deg(original_rows[row_grid], full_rows)
    horizontal_deg = 90.0 - (col_grid - cols / 2.0) * (360.0 / cols)

    vertical = np.deg2rad(vertical_deg)
    horizontal = np.deg2rad(horizontal_deg)
    valid = (
        np.isfinite(radial_range)
        & np.isfinite(intensity)
        & (radial_range > 0.0)
        & (radial_range <= max_range_m)
    )

    x = np.sin(horizontal) * np.cos(vertical) * radial_range
    y = np.cos(horizontal) * np.cos(vertical) * radial_range
    z = np.sin(vertical) * radial_range
    return np.column_stack(
        (x[valid], y[valid], z[valid], intensity[valid])
    ).astype(np.float32)


def select_low_rows(
    full_resolution_image: np.ndarray,
    row_indices: Sequence[int] = DEFAULT_LOW_ROW_INDICES,
) -> np.ndarray:
    """Select low-resolution scan rows from an HWC full-resolution image."""

    image = np.asarray(full_resolution_image)
    if image.shape != (KITTI_ROWS, KITTI_COLS, 2):
        raise ValueError(
            f"Expected ({KITTI_ROWS},{KITTI_COLS},2), got {image.shape}"
        )
    indices = np.asarray(row_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= KITTI_ROWS):
        raise ValueError(f"Invalid low-resolution row indices: {indices}")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"Duplicate low-resolution row indices: {indices}")
    return image[indices].copy()


def restore_low_rows(
    low_resolution_image: np.ndarray,
    row_indices: Sequence[int] = DEFAULT_LOW_ROW_INDICES,
    *,
    full_rows: int = KITTI_ROWS,
) -> np.ndarray:
    """Restore low rows to their original positions in an otherwise-zero RI."""

    low = np.asarray(low_resolution_image)
    indices = np.asarray(row_indices, dtype=np.int64)
    if low.shape != (indices.size, KITTI_COLS, 2):
        raise ValueError(
            f"Expected ({indices.size},{KITTI_COLS},2), got {low.shape}"
        )
    restored = np.zeros((full_rows, KITTI_COLS, 2), dtype=low.dtype)
    restored[indices] = low
    return restored


def validate_range_image(
    image: np.ndarray,
    *,
    max_range_m: float = KITTI_MAX_RANGE_M,
) -> dict[str, float | int | list[int]]:
    """Return invariant and distribution statistics for an HWC range image."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"Expected HWC two-channel range image, got {array.shape}")
    radial_range = array[..., 0]
    intensity = array[..., 1]
    finite = np.isfinite(array)
    range_valid = radial_range > 0.0
    intensity_without_range = (~range_valid) & (intensity != 0.0)

    valid_range_values = radial_range[range_valid]
    valid_intensity_values = intensity[range_valid]
    return {
        "shape": list(array.shape),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "valid_pixel_count": int(range_valid.sum()),
        "intensity_without_range_count": int(intensity_without_range.sum()),
        "range_over_max_count": int((radial_range > max_range_m).sum()),
        "range_min_m": float(valid_range_values.min())
        if valid_range_values.size
        else 0.0,
        "range_max_m": float(valid_range_values.max())
        if valid_range_values.size
        else 0.0,
        "intensity_min": float(valid_intensity_values.min())
        if valid_intensity_values.size
        else 0.0,
        "intensity_max": float(valid_intensity_values.max())
        if valid_intensity_values.size
        else 0.0,
        "finite_value_count": int(finite.sum()),
    }
