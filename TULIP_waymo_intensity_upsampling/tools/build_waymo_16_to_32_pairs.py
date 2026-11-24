#!/usr/bin/env python3
"""Build traceable Waymo 16-line input -> 32-line target pairs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


FRAME_COLUMNS = (
    "dataset_role",
    "q_subset",
    "q_seg",
    "q_frame_index",
    "frame_timestamp_micros",
    "source_tfrecord",
    "frame_npz",
)
PAIR_COLUMNS = (
    "dataset_role",
    "q_subset",
    "q_seg",
    "q_frame_index",
    "timestamp_micros",
    "pair_npz",
    "pair_json",
    "input_height",
    "target_height",
    "width",
    "channels",
    "observed_target_rows",
    "generated_target_rows",
)
ROLES = ("train", "validation", "test")
RING_IDS_32 = np.arange(0, 64, 2, dtype=np.int32)
RING_IDS_16 = np.arange(0, 64, 4, dtype=np.int32)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ through a temporary file and atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        np.savez_compressed(temporary_path, **arrays)
        generated_path = temporary_path.with_suffix(temporary_path.suffix + ".npz")
        os.replace(generated_path, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
        temporary_path.with_suffix(temporary_path.suffix + ".npz").unlink(
            missing_ok=True
        )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Write JSON through a temporary file and atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    """Resolve an index path relative to the repository when necessary."""
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    """Prefer repository-relative paths in generated indexes."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_frame_rows(frame_index: Path, max_frames: int | None) -> list[dict[str, str]]:
    """Load and validate frame-index rows in deterministic order.

    Input shape: CSV rows containing one extracted [64,W] frame NPZ path.
    Output shape: list of validated frame identity dictionaries.
    """
    with frame_index.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in FRAME_COLUMNS if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"frame index is missing columns: {missing}")
        rows = list(reader)
    for row in rows:
        if row["dataset_role"] not in ROLES:
            raise ValueError(f"unsupported dataset role: {row['dataset_role']!r}")
    rows.sort(key=lambda row: (
        ROLES.index(row["dataset_role"]),
        row["q_seg"],
        int(row["q_frame_index"]),
    ))
    if max_frames is not None:
        if max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        rows = rows[:max_frames]
    return rows


def build_pair(data: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create a 16-line input and 32-line target from one extracted frame.

    Input shape: frame arrays range_32/intensity_32/mask_32 are [32,W].
    Output shape: input_16 and target_32 are [H,W,2], H=16 and H=32.
    """
    required = {
        "range_32",
        "intensity_32",
        "valid_mask_32",
        "ring_ids_32",
    }
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"frame is missing arrays: {sorted(missing)}")

    range_32 = np.asarray(data["range_32"], dtype=np.float32)
    intensity_32 = np.asarray(data["intensity_32"], dtype=np.float32)
    mask_32 = np.asarray(data["valid_mask_32"], dtype=bool)
    ring_ids_32 = np.asarray(data["ring_ids_32"], dtype=np.int32)
    if range_32.shape != (32, range_32.shape[1]):
        raise ValueError(f"range_32 must be [32,W], got {range_32.shape}")
    expected = range_32.shape
    for name, array in (("intensity_32", intensity_32), ("valid_mask_32", mask_32)):
        if array.shape != expected:
            raise ValueError(f"{name} must be {expected}, got {array.shape}")
    if not np.array_equal(ring_ids_32, RING_IDS_32):
        raise ValueError(f"ring_ids_32 must be {RING_IDS_32.tolist()}")
    if not np.isfinite(range_32).all() or not np.isfinite(intensity_32).all():
        raise ValueError("range_32 and intensity_32 must be finite")

    observed_rows = np.arange(0, 32, 2, dtype=np.int32)
    generated_rows = np.arange(1, 32, 2, dtype=np.int32)
    input_range = range_32[observed_rows].copy()
    input_intensity = intensity_32[observed_rows].copy()
    input_mask = mask_32[observed_rows].copy()
    input_16 = np.stack((input_range, input_intensity), axis=-1).astype(np.float32)
    target_32 = np.stack((range_32, intensity_32), axis=-1).astype(np.float32)
    arrays = {
        "input_16": input_16,
        "input_valid_mask_16": input_mask,
        "target_32": target_32,
        "target_valid_mask_32": mask_32.copy(),
        "ring_ids_16": RING_IDS_16.copy(),
        "ring_ids_32": RING_IDS_32.copy(),
        "observed_target_rows": observed_rows,
        "generated_target_rows": generated_rows,
    }
    metadata = {
        "task": "Waymo TOP LiDAR 16-to-32 range and intensity upsampling",
        "input_shape": [16, int(expected[1]), 2],
        "target_shape": [32, int(expected[1]), 2],
        "channel_semantics": {"0": "range_m", "1": "raw_intensity"},
        "ring_ids_16": RING_IDS_16.tolist(),
        "ring_ids_32": RING_IDS_32.tolist(),
        "observed_target_rows": observed_rows.tolist(),
        "generated_target_rows": generated_rows.tolist(),
        "normalization": "none",
        "exact_subset_verified": True,
    }
    return arrays, metadata


def _write_index(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    """Write a CSV index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-index", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> None:
    """Build 16→32 pairs and a traceable CSV index."""
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    rows = load_frame_rows(args.frame_index, args.max_frames)
    pair_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for row in rows:
        frame_path = _resolve_repo_path(repo_root, row["frame_npz"])
        pair_dir = output_root / row["dataset_role"] / row["q_seg"] / f"{int(row['q_frame_index']):06d}"
        pair_npz = pair_dir / "tulip_pair_16_to_32.npz"
        pair_json = pair_dir / "tulip_pair_16_to_32.json"
        try:
            with np.load(frame_path, allow_pickle=False) as data:
                arrays, metadata = build_pair(data)
            metadata.update({
                "dataset_role": row["dataset_role"],
                "subset": row["q_subset"],
                "segment_id": row["q_seg"],
                "frame_index": int(row["q_frame_index"]),
                "timestamp_micros": int(row["frame_timestamp_micros"]),
                "source_tfrecord": row["source_tfrecord"],
                "source_frame_npz": row["frame_npz"],
            })
            _atomic_npz(pair_npz, arrays)
            _atomic_json(pair_json, metadata)
            pair_rows.append({
                "dataset_role": row["dataset_role"],
                "q_subset": row["q_subset"],
                "q_seg": row["q_seg"],
                "q_frame_index": row["q_frame_index"],
                "timestamp_micros": row["frame_timestamp_micros"],
                "pair_npz": _relative_or_absolute(pair_npz, repo_root),
                "pair_json": _relative_or_absolute(pair_json, repo_root),
                "input_height": "16",
                "target_height": "32",
                "width": str(metadata["input_shape"][1]),
                "channels": "2",
                "observed_target_rows": "0:32:2",
                "generated_target_rows": "1:32:2",
            })
        except Exception as exc:
            failures.append({
                "dataset_role": row["dataset_role"],
                "q_subset": row["q_subset"],
                "q_seg": row["q_seg"],
                "q_frame_index": row["q_frame_index"],
                "error": f"{type(exc).__name__}: {exc}",
            })

    pair_rows.sort(key=lambda row: (
        ROLES.index(row["dataset_role"]), row["q_seg"], int(row["q_frame_index"])
    ))
    _write_index(output_root / "index.csv", pair_rows, PAIR_COLUMNS)
    _write_index(
        output_root / "failures.csv",
        failures,
        ("dataset_role", "q_subset", "q_seg", "q_frame_index", "error"),
    )
    print(json.dumps({
        "selected_frames": len(rows),
        "indexed_pairs": len(pair_rows),
        "failed_pairs": len(failures),
        "pair_index": str(output_root / "index.csv"),
        "failures": str(output_root / "failures.csv"),
    }, ensure_ascii=False, indent=2))
    if not pair_rows:
        raise SystemExit("no 16-to-32 pairs were created")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
