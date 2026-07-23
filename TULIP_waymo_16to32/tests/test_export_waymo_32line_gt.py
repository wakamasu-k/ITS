"""Tests for Phase 3 raw Waymo 32-line export."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import waymo_preprocess.export_waymo_32line_gt as exporter


def _native(valid: bool = True) -> np.ndarray:
    """Create a deterministic native [64,2650,4] float32 Range Image."""
    image = np.zeros((64, 2650, 4), dtype=np.float32)
    image[..., 0] = np.arange(1, 65, dtype=np.float32)[:, None]
    image[..., 1] = np.arange(10, 2660, dtype=np.float32)[None, :]
    if not valid:
        image[0, 0, 0] = 0.0
        image[2, 1, 0] = -1.0
        image[4, 2, 0] = np.nan
        image[6, 3, 1] = np.inf
    return image


def _frame(with_geometry: bool = True) -> SimpleNamespace:
    if with_geometry:
        calibration = SimpleNamespace(
            name=1,
            extrinsic=SimpleNamespace(transform=list(np.eye(4, dtype=np.float32).reshape(-1))),
            beam_inclinations=list(np.linspace(-0.3, 0.04, 64, dtype=np.float32)),
        )
        pose = SimpleNamespace(transform=list(np.eye(4, dtype=np.float32).reshape(-1)))
    else:
        calibration = SimpleNamespace(name=2, extrinsic=SimpleNamespace(transform=[]), beam_inclinations=[])
        pose = SimpleNamespace(transform=[])
    return SimpleNamespace(
        context=SimpleNamespace(name="synthetic_context", laser_calibrations=[calibration]),
        timestamp_micros=123456789,
        pose=pose,
    )


def _patch_reader(monkeypatch: pytest.MonkeyPatch, frame: object, native: np.ndarray) -> None:
    monkeypatch.setattr(exporter, "iter_frames", lambda _path: iter([frame]))
    monkeypatch.setattr(exporter, "get_top_first_return", lambda _frame: native)


def test_build_raw_32line_contract() -> None:
    """Packed output is raw range/intensity from native even rows."""
    packed, mask, ring_ids = exporter.build_raw_32line(_native(valid=False))
    assert packed.shape == (32, 2650, 2)
    assert packed.dtype == np.float32
    assert mask.shape == (32, 2650)
    assert mask.dtype == np.bool_
    expected = _native()
    np.testing.assert_array_equal(
        packed[:, :, 0], np.where(mask, expected[0::2, :, 0], 0.0)
    )
    np.testing.assert_array_equal(
        packed[:, :, 1], np.where(mask, expected[0::2, :, 1], 0.0)
    )
    np.testing.assert_array_equal(ring_ids, np.arange(0, 64, 2, dtype=np.int32))
    assert packed[0, 0].tolist() == [0.0, 0.0]
    assert packed[2, 2].tolist() == [0.0, 0.0]
    assert np.isfinite(packed).all()


def test_export_writes_npy_sidecar_manifest_and_geometry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One frame produces all required reloadable artifacts."""
    source = tmp_path / "sample.tfrecord"
    source.touch()
    _patch_reader(monkeypatch, _frame(with_geometry=True), _native())
    summary = exporter.export_tfrecords(
        [source], output_root=tmp_path / "out", split="train", frame_limit=1
    )
    assert summary["saved_samples"] == 1
    output_root = tmp_path / "out"
    npy_path = output_root / "train" / "synthetic_context" / "000000.npy"
    json_path = output_root / "train" / "synthetic_context" / "000000.json"
    array = np.load(npy_path, allow_pickle=False)
    metadata = json.loads(json_path.read_text())
    manifest = [json.loads(line) for line in (output_root / "train" / "manifest.jsonl").read_text().splitlines()]
    failures = json.loads((output_root / "train" / "failures.json").read_text())
    assert array.shape == (32, 2650, 2)
    assert array.dtype == np.float32
    assert metadata["native_shape"] == [64, 2650, 4]
    assert metadata["output_shape"] == [32, 2650, 2]
    assert metadata["normalization"] == "none"
    assert metadata["top_lidar_extrinsic"] is not None
    assert len(metadata["beam_inclinations_64"]) == 64
    assert len(metadata["beam_inclinations_32"]) == 32
    assert metadata["vehicle_pose"] is not None
    assert metadata["relative_npy_path"] == "train/synthetic_context/000000.npy"
    assert manifest[0]["relative_npy_path"] == metadata["relative_npy_path"]
    assert manifest[0]["shape"] == [32, 2650, 2]
    assert manifest[0]["valid_rate"] == metadata["valid_rate"]
    assert failures == []


def test_missing_optional_geometry_is_null_with_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable optional geometry is null and explicitly explained."""
    source = tmp_path / "sample.tfrecord"
    source.touch()
    _patch_reader(monkeypatch, _frame(with_geometry=False), _native())
    exporter.export_tfrecords([source], output_root=tmp_path / "out", split="train", frame_limit=1)
    metadata = json.loads((tmp_path / "out/train/synthetic_context/000000.json").read_text())
    assert metadata["top_lidar_extrinsic"] is None
    assert metadata["beam_inclinations_64"] is None
    assert metadata["beam_inclinations_32"] is None
    assert metadata["vehicle_pose"] is None
    assert metadata["metadata_unavailable_reasons"]


def test_failure_is_recorded_and_later_frame_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame error is retained while a later frame can still be saved."""
    source = tmp_path / "sample.tfrecord"
    source.touch()
    frames = [_frame(), _frame()]
    monkeypatch.setattr(exporter, "iter_frames", lambda _path: iter(frames))
    calls = {"count": 0}

    def fake_get(_frame: object) -> np.ndarray:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("synthetic failure")
        return _native()

    monkeypatch.setattr(exporter, "get_top_first_return", fake_get)
    summary = exporter.export_tfrecords(
        [source], output_root=tmp_path / "out", split="train", frame_limit=2
    )
    assert summary["saved_samples"] == 1
    assert summary["failed_frames"] == 1
    failures = json.loads((tmp_path / "out/train/failures.json").read_text())
    assert failures[0]["frame_index"] == 0
    assert failures[0]["error_type"] == "RuntimeError"
    assert (tmp_path / "out/train/synthetic_context/000001.npy").exists()


def test_existing_protection_and_skip_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing sample is protected and can be explicitly skipped."""
    source = tmp_path / "sample.tfrecord"
    source.touch()
    _patch_reader(monkeypatch, _frame(), _native())
    output = tmp_path / "out"
    exporter.export_tfrecords([source], output_root=output, split="train", frame_limit=1)
    npy_path = output / "train/synthetic_context/000000.npy"
    original = np.load(npy_path, allow_pickle=False).copy()
    with pytest.raises(exporter.AllFramesFailedError):
        exporter.export_tfrecords([source], output_root=output, split="train", frame_limit=1)
    summary = exporter.export_tfrecords(
        [source], output_root=output, split="train", frame_limit=1, skip_existing=True
    )
    assert summary["skipped_samples"] == 1
    np.testing.assert_array_equal(np.load(npy_path, allow_pickle=False), original)


def test_overwrite_replaces_existing_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Overwrite explicitly replaces the NPY and sidecar atomically."""
    source = tmp_path / "sample.tfrecord"
    source.touch()
    _patch_reader(monkeypatch, _frame(), _native())
    output = tmp_path / "out"
    exporter.export_tfrecords([source], output_root=output, split="train", frame_limit=1)
    changed = _native()
    changed[..., 0] += 1000.0
    monkeypatch.setattr(exporter, "get_top_first_return", lambda _frame: changed)
    exporter.export_tfrecords([source], output_root=output, split="train", frame_limit=1, overwrite=True)
    array = np.load(output / "train/synthetic_context/000000.npy", allow_pickle=False)
    assert float(array[0, 0, 0]) == 1001.0


def test_atomic_npy_has_no_temporary_file(tmp_path: Path) -> None:
    """Atomic save leaves a readable NPY and no exporter temp file."""
    path = tmp_path / "sample.npy"
    exporter._atomic_write_npy(path, np.ones((2, 3, 2), dtype=np.float32))
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), np.ones((2, 3, 2), dtype=np.float32))
    assert list(tmp_path.glob(".*.tmp")) == []


def test_input_source_exclusivity_and_sorted_discovery(tmp_path: Path) -> None:
    """Directory and list discovery are sorted and source arguments are exclusive."""
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "b.tfrecord").touch()
    (root / "a.tfrecord").touch()
    (root / "ignore.txt").touch()
    selected = exporter.discover_tfrecords(input_root=root)
    assert [path.name for path in selected] == ["a.tfrecord", "b.tfrecord"]
    list_path = tmp_path / "inputs.txt"
    list_path.write_text("inputs/b.tfrecord\ninputs/a.tfrecord\n")
    listed = exporter.discover_tfrecords(tfrecord_list=list_path)
    assert [path.name for path in listed] == ["a.tfrecord", "b.tfrecord"]
    with pytest.raises(ValueError, match="exactly one"):
        exporter.discover_tfrecords(input_root=root, tfrecord=root / "a.tfrecord")


def test_output_safety_empty_input_and_parser_controls(tmp_path: Path) -> None:
    """Forbidden output, empty input, and mutually exclusive flags fail clearly."""
    with pytest.raises(ValueError, match="forbidden"):
        exporter.validate_output_root("/mnt/w/32line/blocked")
    with pytest.raises(ValueError, match=r"No \.tfrecord"):
        exporter.discover_tfrecords(input_root=tmp_path)
    with pytest.raises(SystemExit):
        exporter._build_parser().parse_args([
            "--tfrecord", "a.tfrecord", "--split", "train", "--output-root", "out",
            "--overwrite", "--skip-existing",
        ])


def test_all_failure_is_non_success_condition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """All failed frames raise and still write failures.json."""
    source = tmp_path / "sample.tfrecord"
    source.touch()
    monkeypatch.setattr(exporter, "iter_frames", lambda _path: iter([_frame()]))
    monkeypatch.setattr(exporter, "get_top_first_return", lambda _frame: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(exporter.AllFramesFailedError):
        exporter.export_tfrecords([source], output_root=tmp_path / "out", split="train", frame_limit=1)
    failures = json.loads((tmp_path / "out/train/failures.json").read_text())
    assert len(failures) == 1
