"""Collect streaming range/intensity statistics for Waymo 32-line GT images.

The module stops before normalization, Dataset, and model code.  It reuses the
Phase 1 TFRecord and native Range Image readers and keeps only running moments
plus bounded per-frame samples for percentile estimation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

if __package__ in (None, ""):
    from waymo_range_image import extract_range_intensity, make_32line_gt
    from waymo_tfrecord_io import get_top_first_return, iter_frames
else:
    from waymo_preprocess.waymo_range_image import extract_range_intensity, make_32line_gt
    from waymo_preprocess.waymo_tfrecord_io import get_top_first_return, iter_frames


SCRIPT_VERSION = "phase2-range-intensity-statistics-v1"
LIDAR_NAME = "TOP"
RETURN_INDEX = 0
RING_IDS_32 = np.arange(0, 64, 2, dtype=np.int32)
PERCENTILES = (1, 5, 50, 90, 95, 99, 99.5, 99.9, 99.99)
PERCENTILE_KEYS = ("p01", "p05", "p50", "p90", "p95", "p99", "p99_5", "p99_9", "p99_99")


def _shape_key(shape: Sequence[int]) -> str:
    """Return a stable JSON key for an array shape.

    Args:
        shape: Shape sequence, for example ``(64, 2650, 4)``.

    Returns:
        Shape formatted as a compact JSON list string.
    """
    return json.dumps([int(value) for value in shape], separators=(",", ":"))


@dataclass
class StreamingMoments:
    """Streaming count, extrema, sum, and sum-of-squares accumulator.

    Input shape: any flattenable finite numeric array.
    Output shape: scalar statistics returned by :meth:`to_dict`.
    """

    count: int = 0
    minimum: float = np.inf
    maximum: float = -np.inf
    total: float = 0.0
    sum_of_squares: float = 0.0

    def update(self, values: np.ndarray) -> None:
        """Update moments without retaining the input values.

        Args:
            values: Numeric array containing values to aggregate.

        Raises:
            ValueError: If any supplied value is NaN or Inf.
        """
        array = np.asarray(values)
        if not np.isfinite(array).all():
            raise ValueError("StreamingMoments received NaN or Inf.")
        flat = array.reshape(-1)
        if flat.size == 0:
            return
        flat64 = flat.astype(np.float64, copy=False)
        self.count += int(flat64.size)
        self.minimum = min(self.minimum, float(np.min(flat64)))
        self.maximum = max(self.maximum, float(np.max(flat64)))
        self.total += float(np.sum(flat64, dtype=np.float64))
        self.sum_of_squares += float(np.sum(flat64 * flat64, dtype=np.float64))

    def to_dict(self, samples: np.ndarray) -> dict[str, float | int]:
        """Return moments and percentiles calculated from bounded samples.

        Args:
            samples: Sampled values used only for percentile estimation.

        Returns:
            JSON-serializable scalar statistics.

        Raises:
            ValueError: If no values or samples were accumulated.
        """
        if self.count == 0:
            raise ValueError("Cannot summarize empty statistics.")
        sample_array = np.asarray(samples, dtype=np.float64).reshape(-1)
        if sample_array.size == 0:
            raise ValueError("Cannot calculate percentiles without samples.")
        if not np.isfinite(sample_array).all():
            raise ValueError("Percentile samples contain NaN or Inf.")
        mean = self.total / self.count
        variance = max(self.sum_of_squares / self.count - mean * mean, 0.0)
        result: dict[str, float | int] = {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": float(np.sqrt(variance)),
        }
        for key, value in zip(PERCENTILE_KEYS, np.percentile(sample_array, PERCENTILES)):
            result[key] = float(value)
        return result


@dataclass
class StatisticsAccumulator:
    """Accumulate statistics for native and derived 32-line frames.

    Input shape per frame: native ``[64,W,C]`` and derived arrays ``[32,W]``.
    Output shape: JSON-compatible run, count, statistic, and shape sections.
    """

    sample_per_frame: int
    rng: np.random.Generator
    range_moments: StreamingMoments = field(default_factory=StreamingMoments)
    intensity_moments: StreamingMoments = field(default_factory=StreamingMoments)
    range_samples: list[np.ndarray] = field(default_factory=list)
    intensity_samples: list[np.ndarray] = field(default_factory=list)
    sampled_value_count: int = 0
    total_pixel_count: int = 0
    valid_pixel_count: int = 0
    processed_frames: int = 0
    native_shapes: Counter[str] = field(default_factory=Counter)
    gt_shapes: Counter[str] = field(default_factory=Counter)

    def add_frame(
        self,
        native_shape: Sequence[int],
        range_32: np.ndarray,
        intensity_32: np.ndarray,
        mask_32: np.ndarray,
    ) -> None:
        """Add one frame using the same sampled valid positions for both channels.

        Args:
            native_shape: Original native Range Image shape ``[64,W,C]``.
            range_32: Float32 range image ``[32,W]``.
            intensity_32: Float32 intensity image ``[32,W]``.
            mask_32: Bool validity mask ``[32,W]``.

        Raises:
            ValueError: If shapes or finite-value constraints are invalid.
        """
        range_array = np.asarray(range_32, dtype=np.float32)
        intensity_array = np.asarray(intensity_32, dtype=np.float32)
        mask_array = np.asarray(mask_32, dtype=bool)
        if range_array.ndim != 2 or range_array.shape[0] != 32:
            raise ValueError("range_32 must have shape [32,W].")
        if intensity_array.shape != range_array.shape or mask_array.shape != range_array.shape:
            raise ValueError("32-line range, intensity, and mask shapes must match.")
        if not np.isfinite(range_array).all() or not np.isfinite(intensity_array).all():
            raise ValueError("32-line arrays contain NaN or Inf.")

        valid_indices = np.flatnonzero(mask_array.reshape(-1))
        self.total_pixel_count += int(mask_array.size)
        self.valid_pixel_count += int(valid_indices.size)
        self.processed_frames += 1
        self.native_shapes[_shape_key(native_shape)] += 1
        self.gt_shapes[_shape_key(range_array.shape)] += 1

        flat_range = range_array.reshape(-1)
        flat_intensity = intensity_array.reshape(-1)
        self.range_moments.update(flat_range[valid_indices])
        self.intensity_moments.update(flat_intensity[valid_indices])

        sample_count = min(self.sample_per_frame, int(valid_indices.size))
        if sample_count == 0:
            return
        selected = self.rng.choice(valid_indices, size=sample_count, replace=False)
        self.range_samples.append(flat_range[selected].copy())
        self.intensity_samples.append(flat_intensity[selected].copy())
        self.sampled_value_count += sample_count

    def finalize_statistics(self) -> tuple[dict[str, float | int], dict[str, float | int]]:
        """Finalize range and intensity statistics from streaming state.

        Returns:
            Range and intensity statistic dictionaries.

        Raises:
            ValueError: If no valid pixels or percentile samples exist.
        """
        if self.valid_pixel_count == 0:
            raise ValueError("No valid 32-line pixels were processed.")
        if self.sampled_value_count == 0:
            raise ValueError("No percentile samples were collected.")
        range_samples = np.concatenate(self.range_samples)
        intensity_samples = np.concatenate(self.intensity_samples)
        return self.range_moments.to_dict(range_samples), self.intensity_moments.to_dict(intensity_samples)


def discover_tfrecords(
    input_root: str | Path | None,
    tfrecord: str | Path | None,
    max_tfrecords: int,
) -> tuple[list[Path], int]:
    """Discover sorted ``.tfrecord`` inputs.

    Input shape: one directory or one file path.
    Output shape: ``(selected_paths, discovered_count)``.

    Raises:
        ValueError: If both/neither source is given, suffix is invalid, or no
            matching files exist.
        FileNotFoundError: If the selected path does not exist.
    """
    if (input_root is None) == (tfrecord is None):
        raise ValueError("Specify exactly one of --input-root or --tfrecord.")
    if max_tfrecords <= 0:
        raise ValueError("max_tfrecords must be positive.")
    if tfrecord is not None:
        path = Path(tfrecord).expanduser()
        if path.suffix != ".tfrecord":
            raise ValueError(f"Input must have .tfrecord suffix: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"TFRecord does not exist: {path}")
        return [path], 1

    root = Path(input_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"TFRecord directory does not exist: {root}")
    discovered = sorted(path for path in root.iterdir() if path.is_file() and path.suffix == ".tfrecord")
    if not discovered:
        raise ValueError(f"No .tfrecord files found under: {root}")
    return discovered[:max_tfrecords], len(discovered)


def _failure_record(path: Path, frame_index: int, error: Exception) -> dict[str, object]:
    """Create a JSON-safe failure record."""
    return {
        "tfrecord": str(path),
        "frame_index": int(frame_index),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def analyze_files(
    paths: Sequence[Path],
    *,
    max_frames_per_tfrecord: int,
    sample_per_frame: int,
    seed: int,
    discovered_tfrecords: int,
    input_root: str | Path | None,
    tfrecord: str | Path | None,
    max_tfrecords: int,
) -> dict[str, object]:
    """Analyze selected TFRecords and record recoverable failures.

    Input shape: selected TFRecord paths.
    Output shape: JSON-compatible report with ``run``, ``counts``, statistics,
        ``shapes``, and ``failures`` sections.

    Raises:
        ValueError: If limits are invalid or no valid frame data is produced.
    """
    if max_frames_per_tfrecord <= 0:
        raise ValueError("max_frames_per_tfrecord must be positive.")
    if sample_per_frame <= 0:
        raise ValueError("sample_per_frame must be positive.")
    started = time.perf_counter()
    accumulator = StatisticsAccumulator(sample_per_frame, np.random.default_rng(seed))
    failures: list[dict[str, object]] = []
    processed_tfrecords = 0
    failed_tfrecords = 0

    for path in paths:
        tfrecord_processed = False
        tfrecord_failed = False
        frame_index = -1
        try:
            frame_iterator: Iterator[object] = iter_frames(path)
            for frame_index, frame in enumerate(itertools.islice(frame_iterator, max_frames_per_tfrecord)):
                try:
                    native = get_top_first_return(frame)
                    range_64, intensity_64, mask_64 = extract_range_intensity(native)
                    range_32, intensity_32, mask_32, _ = make_32line_gt(range_64, intensity_64, mask_64)
                    accumulator.add_frame(native.shape, range_32, intensity_32, mask_32)
                    tfrecord_processed = True
                except Exception as error:
                    failures.append(_failure_record(path, frame_index, error))
        except Exception as error:
            tfrecord_failed = True
            failures.append(_failure_record(path, frame_index, error))
        if tfrecord_processed:
            processed_tfrecords += 1
        if tfrecord_failed:
            failed_tfrecords += 1

    range_statistics, intensity_statistics = accumulator.finalize_statistics()
    invalid_pixel_count = accumulator.total_pixel_count - accumulator.valid_pixel_count
    report = {
        "run": {
            "input_root": str(input_root) if input_root is not None else None,
            "tfrecord": str(tfrecord) if tfrecord is not None else None,
            "max_tfrecords": max_tfrecords,
            "max_frames_per_tfrecord": max_frames_per_tfrecord,
            "sample_per_frame": sample_per_frame,
            "seed": seed,
            "lidar_name": LIDAR_NAME,
            "return_index": RETURN_INDEX,
            "ring_ids_32": RING_IDS_32.tolist(),
            "execution_time": time.perf_counter() - started,
            "script_version": SCRIPT_VERSION,
        },
        "counts": {
            "discovered_tfrecords": discovered_tfrecords,
            "selected_tfrecords": len(paths),
            "processed_tfrecords": processed_tfrecords,
            "failed_tfrecords": failed_tfrecords,
            "processed_frames": accumulator.processed_frames,
            "failed_frames": sum(1 for item in failures if item["frame_index"] >= 0),
            "total_pixel_count": accumulator.total_pixel_count,
            "valid_pixel_count": accumulator.valid_pixel_count,
            "invalid_pixel_count": invalid_pixel_count,
            "valid_rate": accumulator.valid_pixel_count / accumulator.total_pixel_count,
            "sampled_value_count": accumulator.sampled_value_count,
        },
        "range": range_statistics,
        "intensity": intensity_statistics,
        "shapes": {
            "native": dict(sorted(accumulator.native_shapes.items())),
            "32line": dict(sorted(accumulator.gt_shapes.items())),
        },
        "failures": failures,
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    """Build the Phase 2 CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-root", type=Path)
    source.add_argument("--tfrecord", type=Path)
    parser.add_argument("--max-tfrecords", type=int, default=5)
    parser.add_argument("--max-frames-per-tfrecord", type=int, default=20)
    parser.add_argument("--sample-per-frame", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _print_summary(report: dict[str, object], output_json: Path) -> None:
    """Print the requested stdout summary."""
    counts = report["counts"]
    print(f"processed TFRecords: {counts['processed_tfrecords']}")
    print(f"processed frames: {counts['processed_frames']}")
    print(f"valid pixels/rate: {counts['valid_pixel_count']} / {counts['valid_rate']:.12f}")
    print("range statistics: " + json.dumps(report["range"], sort_keys=True))
    print("intensity statistics: " + json.dumps(report["intensity"], sort_keys=True))
    print("detected shapes: " + json.dumps(report["shapes"], sort_keys=True))
    print(f"failure count: {len(report['failures'])}")
    print(f"output JSON: {output_json}")


def main(argv: Iterable[str] | None = None) -> int:
    """Run the Phase 2 statistics CLI.

    Args:
        argv: Optional argument sequence; None uses command-line arguments.

    Returns:
        Zero on success.

    Raises:
        ValueError: If no valid data is available after processing.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        selected, discovered = discover_tfrecords(args.input_root, args.tfrecord, args.max_tfrecords)
        report = analyze_files(
            selected,
            max_frames_per_tfrecord=args.max_frames_per_tfrecord,
            sample_per_frame=args.sample_per_frame,
            seed=args.seed,
            discovered_tfrecords=discovered,
            input_root=args.input_root,
            tfrecord=args.tfrecord,
            max_tfrecords=args.max_tfrecords,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    output_json = args.output_json.expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _print_summary(report, output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
