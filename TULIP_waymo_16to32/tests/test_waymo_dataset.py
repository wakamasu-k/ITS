"""Tests for the Phase 4A 16-line/32-line Dataset."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tulip_adapter.waymo_16to32_dataset import Waymo16To32Dataset


def _make_manifest_dataset(tmp_path: Path, count: int = 2) -> Path:
    """Create a minimal Phase 3-compatible output root."""
    output_root = tmp_path / "data"
    sample_dir = output_root / "train" / "context"
    sample_dir.mkdir(parents=True)
    records = []
    for index in range(count):
        raw = np.zeros((32, 5, 2), dtype=np.float32)
        raw[..., 0] = np.arange(1, 33, dtype=np.float32)[:, None] + index
        raw[..., 1] = np.arange(5, dtype=np.float32)[None, :] + index
        if index == 0:
            raw[0, 0] = 0.0
        npy_path = sample_dir / f"{index:06d}.npy"
        np.save(npy_path, raw, allow_pickle=False)
        records.append({
            "sample_id": f"context/{index:06d}",
            "relative_npy_path": f"train/context/{index:06d}.npy",
            "shape": [32, 5, 2],
            "valid_rate": 1.0,
        })
    manifest = output_root / "train" / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    return output_root


def test_dataset_shapes_dtype_and_even_rows(tmp_path: Path) -> None:
    """Dataset returns low/high CHW float32 tensors with circular padding."""
    root = _make_manifest_dataset(tmp_path)
    dataset = Waymo16To32Dataset(
        root / "train" / "manifest.jsonl",
        maximum_range=40.0,
        intensity_mode="none",
        native_width=5,
        target_width=8,
    )
    low, high = dataset[0]
    assert len(dataset) == 2
    assert low.shape == (2, 16, 8)
    assert high.shape == (2, 32, 8)
    assert low.dtype == torch.float32
    assert high.dtype == torch.float32
    torch.testing.assert_close(low[:, :, :5], high[:, 0::2, :5])
    torch.testing.assert_close(low[:, :, 5:], high[:, 0::2, :3])
    assert high[:, 0, 0].tolist() == [0.0, 0.0]


def test_dataset_accepts_output_root_and_dataloader_batches(tmp_path: Path) -> None:
    """Output-root discovery and DataLoader batching work."""
    root = _make_manifest_dataset(tmp_path, count=2)
    dataset = Waymo16To32Dataset(root, maximum_range=40.0, native_width=5, target_width=8)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    low_batch, high_batch = next(iter(loader))
    assert low_batch.shape == (2, 2, 16, 8)
    assert high_batch.shape == (2, 2, 32, 8)
    assert low_batch.dtype == torch.float32
    assert high_batch.dtype == torch.float32


@pytest.mark.parametrize("mode", ["none", "clip_1", "log1p"])
def test_dataset_intensity_modes(tmp_path: Path, mode: str) -> None:
    """Dataset exposes all requested intensity transform choices."""
    root = _make_manifest_dataset(tmp_path, count=1)
    low, high = Waymo16To32Dataset(
        root, maximum_range=40.0, intensity_mode=mode, native_width=5, target_width=8  # type: ignore[arg-type]
    )[0]
    assert torch.isfinite(low).all()
    assert torch.isfinite(high).all()
    if mode == "clip_1":
        assert float(high[1].max()) <= 1.0
    if mode == "log1p":
        assert float(high[1, 1, 1]) == pytest.approx(float(np.log1p(1.0)))


def test_dataset_rejects_empty_or_bad_raw_sample(tmp_path: Path) -> None:
    """Empty manifest and wrong NPY shape fail clearly."""
    empty_root = tmp_path / "empty"
    (empty_root / "train").mkdir(parents=True)
    (empty_root / "train" / "manifest.jsonl").write_text("")
    with pytest.raises(ValueError, match="no samples"):
        Waymo16To32Dataset(empty_root, maximum_range=40.0)

    root = _make_manifest_dataset(tmp_path / "bad", count=1)
    bad_path = root / "train/context/000000.npy"
    np.save(bad_path, np.zeros((16, 5, 2), dtype=np.float32), allow_pickle=False)
    dataset = Waymo16To32Dataset(root, maximum_range=40.0, native_width=5, target_width=8)
    with pytest.raises(ValueError, match="Expected raw sample shape"):
        _ = dataset[0]


def test_manifest_path_escape_is_rejected(tmp_path: Path) -> None:
    """Manifest paths cannot escape the output root."""
    root = tmp_path / "escape"
    (root / "train").mkdir(parents=True)
    (root / "train/manifest.jsonl").write_text(
        json.dumps({"relative_npy_path": "../outside.npy"}) + "\n"
    )
    with pytest.raises(ValueError, match="escapes"):
        Waymo16To32Dataset(root, maximum_range=40.0)
