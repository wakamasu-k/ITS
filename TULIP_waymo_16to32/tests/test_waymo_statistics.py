"""Tests for Phase 2 streaming Waymo statistics."""

import json
from pathlib import Path

import numpy as np
import pytest

import waymo_preprocess.analyze_waymo_range_intensity_stats as stats
from waymo_preprocess.waymo_range_image import extract_range_intensity, make_32line_gt


def _components(width: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic 32-line float32 components and a full mask."""
    range_image = np.arange(1, 32 * width + 1, dtype=np.float32).reshape(32, width)
    intensity_image = (range_image * 10.0).astype(np.float32)
    return range_image, intensity_image, np.ones((32, width), dtype=bool)


def _accumulator(seed: int = 42, sample_per_frame: int = 10000) -> stats.StatisticsAccumulator:
    """Build an accumulator with one deterministic frame."""
    accumulator = stats.StatisticsAccumulator(sample_per_frame, np.random.default_rng(seed))
    accumulator.add_frame((64, 4, 4), *_components())
    return accumulator


def test_invalid_zero_nan_inf_are_excluded() -> None:
    """Phase 1 masking excludes zero, negative, NaN, and Inf pixels."""
    native = np.ones((64, 2, 2), dtype=np.float32)
    native[..., 0] = 2.0
    native[..., 1] = 5.0
    native[0, 0, 0] = 0.0
    native[2, 0, 0] = -1.0
    native[4, 0, 0] = np.nan
    native[6, 0, 1] = np.inf
    range_64, intensity_64, mask_64 = extract_range_intensity(native)
    range_32, intensity_32, mask_32, _ = make_32line_gt(range_64, intensity_64, mask_64)
    accumulator = stats.StatisticsAccumulator(100, np.random.default_rng(42))
    accumulator.add_frame(native.shape, range_32, intensity_32, mask_32)
    result_range, result_intensity = accumulator.finalize_statistics()
    assert result_range["min"] == 2.0
    assert result_range["max"] == 2.0
    assert result_intensity["min"] == 5.0
    assert result_intensity["max"] == 5.0
    assert accumulator.valid_pixel_count == 32 * 2 - 4


def test_known_moments_and_percentiles() -> None:
    """Mean, population std, and percentile keys match NumPy."""
    accumulator = stats.StatisticsAccumulator(100, np.random.default_rng(1))
    range_image = np.arange(1, 33, dtype=np.float32).reshape(32, 1)
    intensity_image = (range_image * 2).astype(np.float32)
    accumulator.add_frame((64, 1, 2), range_image, intensity_image, np.ones_like(range_image, dtype=bool))
    result_range, result_intensity = accumulator.finalize_statistics()
    expected = np.arange(1, 33, dtype=np.float64)
    assert np.isclose(result_range["mean"], expected.mean())
    assert np.isclose(result_range["std"], expected.std())
    for key, percentile in zip(stats.PERCENTILE_KEYS, stats.PERCENTILES):
        assert np.isclose(result_range[key], np.percentile(expected, percentile))
    assert np.isclose(result_intensity["mean"], (expected * 2).mean())


def test_sample_per_frame_is_bounded() -> None:
    """Only at most sample_per_frame positions are retained per frame."""
    accumulator = _accumulator(sample_per_frame=3)
    assert accumulator.sampled_value_count == 3
    assert sum(array.size for array in accumulator.range_samples) == 3
    assert sum(array.size for array in accumulator.intensity_samples) == 3


def test_same_seed_reproduces_and_different_seed_is_accepted() -> None:
    """Sampling is deterministic for equal seeds and configurable otherwise."""
    first = _accumulator(seed=42, sample_per_frame=5)
    second = _accumulator(seed=42, sample_per_frame=5)
    third = _accumulator(seed=7, sample_per_frame=5)
    np.testing.assert_array_equal(first.range_samples[0], second.range_samples[0])
    np.testing.assert_array_equal(first.intensity_samples[0], second.intensity_samples[0])
    assert third.sampled_value_count == 5


def test_shape_counts_are_serializable() -> None:
    """Native and 32-line shapes are counted and JSON serializable."""
    accumulator = _accumulator()
    accumulator.add_frame((64, 4, 4), *_components())
    assert accumulator.native_shapes["[64,4,4]"] == 2
    assert accumulator.gt_shapes["[32,4]"] == 2
    report = {
        "native": dict(accumulator.native_shapes),
        "32line": dict(accumulator.gt_shapes),
    }
    json.dumps(report)


def test_empty_data_is_rejected() -> None:
    """An accumulator with no valid values fails clearly."""
    accumulator = stats.StatisticsAccumulator(10, np.random.default_rng(42))
    with pytest.raises(ValueError, match="No valid"):
        accumulator.finalize_statistics()


def test_failure_record_is_retained_and_other_frame_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame error is recorded while a later valid frame is processed."""
    valid_native = np.ones((64, 2, 2), dtype=np.float32)
    valid_native[..., 0] = 3.0
    valid_native[..., 1] = 4.0

    def fake_iter_frames(_path: Path):
        yield "bad"
        yield "good"

    def fake_get_top_first_return(frame: object) -> np.ndarray:
        if frame == "bad":
            raise RuntimeError("synthetic frame failure")
        return valid_native

    monkeypatch.setattr(stats, "iter_frames", fake_iter_frames)
    monkeypatch.setattr(stats, "get_top_first_return", fake_get_top_first_return)
    report = stats.analyze_files(
        [Path("synthetic.tfrecord")],
        max_frames_per_tfrecord=2,
        sample_per_frame=10,
        seed=42,
        discovered_tfrecords=1,
        input_root=None,
        tfrecord=Path("synthetic.tfrecord"),
        max_tfrecords=1,
    )
    assert report["counts"]["processed_frames"] == 1
    assert report["counts"]["failed_frames"] == 1
    assert len(report["failures"]) == 1
    assert report["failures"][0]["frame_index"] == 0


def test_discovery_is_sorted_and_only_tfrecord(tmp_path: Path) -> None:
    """Directory discovery sorts paths and excludes other extensions."""
    (tmp_path / "b.tfrecord").touch()
    (tmp_path / "a.tfrecord").touch()
    (tmp_path / "ignore.txt").touch()
    selected, discovered = stats.discover_tfrecords(tmp_path, None, 5)
    assert discovered == 2
    assert [path.name for path in selected] == ["a.tfrecord", "b.tfrecord"]


def test_input_root_and_tfrecord_are_mutually_exclusive(tmp_path: Path) -> None:
    """Both source options are rejected explicitly."""
    with pytest.raises(ValueError, match="exactly one"):
        stats.discover_tfrecords(tmp_path, tmp_path / "a.tfrecord", 5)


def test_report_json_round_trip() -> None:
    """The complete analysis report can be serialized to JSON."""
    accumulator = _accumulator(sample_per_frame=3)
    range_statistics, intensity_statistics = accumulator.finalize_statistics()
    report = {
        "run": {"ring_ids_32": stats.RING_IDS_32.tolist()},
        "counts": {"sampled_value_count": accumulator.sampled_value_count},
        "range": range_statistics,
        "intensity": intensity_statistics,
        "shapes": {"native": dict(accumulator.native_shapes)},
        "failures": [],
    }
    json.dumps(report)
