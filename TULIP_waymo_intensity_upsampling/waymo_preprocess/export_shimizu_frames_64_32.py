#!/usr/bin/env python3
"""Export native 64-line and even-ring 32-line arrays from a frame manifest."""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from waymo_preprocess.shimizu_batch_export import (
    ExportItem, group_by_tfrecord, load_export_items, output_paths,
    select_pending, validate_frame_identity,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--failures", type=Path)
    return parser.parse_args()

def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def write_failure_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("q_subset", "q_seg", "q_frame_index", "source_tfrecord", "error")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def export_selected_frame(item: ExportItem, frame: Any, output_root: Path, *,
                          dataset_role: str = "localization_evaluation",
                          source_subset: str | None = None) -> None:
    source_subset = source_subset or item.subset
    from waymo_preprocess.export_one_frame_64_32 import (
        PREPROCESSING_VERSION, extract_range_intensity, select_even_rings,
        top_first_return, valid_stats,
    )
    context_name = str(frame.context.name)
    timestamp = int(frame.timestamp_micros)
    validate_frame_identity(item, context_name, timestamp)
    native = top_first_return(frame)
    range_64, intensity_64, mask_64 = extract_range_intensity(native)
    range_32, intensity_32, mask_32, ring_ids = select_even_rings(
        range_64, intensity_64, mask_64)
    npz_path, metadata_path = output_paths(output_root, item)
    atomic_npz(npz_path, {
        "range_64": range_64.astype(np.float32),
        "intensity_64": intensity_64.astype(np.float32),
        "valid_mask_64": mask_64,
        "range_32": range_32.astype(np.float32),
        "intensity_32": intensity_32.astype(np.float32),
        "valid_mask_32": mask_32,
        "ring_ids_32": ring_ids,
    })
    atomic_json(metadata_path, {
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset_role": dataset_role,
        "segment_id": item.segment_id,
        "source_subset": source_subset,
        "source_tfrecord": str(item.tfrecord),
        "source_frame_index": item.frame_index,
        "context_name": context_name,
        "timestamp_micros": timestamp,
        "native_range_image_shape": list(native.shape),
        "range_shape_64": list(range_64.shape),
        "range_shape_32": list(range_32.shape),
        "ring_keep_rule": "even",
        "ring_ids_32": ring_ids.tolist(),
        "valid_count_64": int(mask_64.sum()),
        "valid_count_32": int(mask_32.sum()),
        "range_64_valid_m": valid_stats(range_64, mask_64),
        "intensity_64_valid": valid_stats(intensity_64, mask_64),
        "range_32_valid_m": valid_stats(range_32, mask_32),
        "intensity_32_valid": valid_stats(intensity_32, mask_32),
        "normalization": "none",
        "histogram_equalization": False,
        "clahe": False,
        "temporal_accumulation": False,
        "dynamic_object_removal": False,
    })

def failure(item: ExportItem, exc: Exception) -> dict[str, str]:
    return {"q_subset": item.subset, "q_seg": item.segment_id,
            "q_frame_index": str(item.frame_index),
            "source_tfrecord": str(item.tfrecord),
            "error": f"{type(exc).__name__}: {exc}"}

def main() -> None:
    args = parse_args()
    items = load_export_items(args.manifest, args.max_frames)
    pending, skipped = select_pending(items, args.output_root, args.resume)
    groups = group_by_tfrecord(pending)
    print(json.dumps({
        "selected_frames": len(items), "pending_frames": len(pending),
        "resume_skipped": skipped, "tfrecords": len(groups),
        "dry_run": args.dry_run,
    }, ensure_ascii=False), flush=True)
    if args.dry_run:
        for item in pending[:10]:
            npz_path, _ = output_paths(args.output_root, item)
            print(f"would_export={item.tfrecord}#{item.frame_index} -> {npz_path}")
        return
    failure_path = args.failures or args.output_root / "failures.csv"
    if not pending:
        write_failure_rows(failure_path, [])
        print(json.dumps({
            "exported": 0, "resume_skipped": skipped,
            "failed": 0, "failures": str(failure_path),
        }, ensure_ascii=False, indent=2))
        return

    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset
    failures = []
    exported = 0
    for group_number, (tfrecord, group) in enumerate(groups.items(), 1):
        expected = {item.frame_index: item for item in group}
        remaining = set(expected)
        last_index = max(remaining)
        print(f"tfrecord {group_number}/{len(groups)} frames={len(group)} "
              f"path={tfrecord}", flush=True)
        try:
            dataset = tf.data.TFRecordDataset(str(tfrecord), compression_type="")
            for index, record in enumerate(dataset):
                if index in expected:
                    item = expected[index]
                    try:
                        frame = open_dataset.Frame()
                        frame.ParseFromString(record.numpy())
                        export_selected_frame(item, frame, args.output_root)
                        exported += 1
                        remaining.remove(index)
                        print(f"exported {exported}/{len(pending)} "
                              f"{item.segment_id}#{index}", flush=True)
                    except Exception as exc:
                        failures.append(failure(item, exc))
                        remaining.discard(index)
                        if args.fail_fast:
                            raise
                if index >= last_index:
                    break
            for index in sorted(remaining):
                failures.append(failure(
                    expected[index], IndexError(f"frame {index} not found")))
        except Exception:
            if args.fail_fast:
                raise
            for index in sorted(remaining):
                failures.append(failure(
                    expected[index], RuntimeError("TFRecord scan failed")))
    write_failure_rows(failure_path, failures)
    print(json.dumps({
        "exported": exported, "resume_skipped": skipped,
        "failed": len(failures), "failures": str(failure_path),
    }, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
