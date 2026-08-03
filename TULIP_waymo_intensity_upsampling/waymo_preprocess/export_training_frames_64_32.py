#!/usr/bin/env python3
"""Extract a bounded training sample of native 64/even-ring 32 Waymo frames."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waymo_preprocess.export_shimizu_frames_64_32 import (
    export_selected_frame, write_failure_rows,
)
from waymo_preprocess.shimizu_batch_export import ExportItem
from waymo_preprocess.training_frame_export import (
    ROLES, frame_manifest_row, load_training_segments, pending_frame_indices,
    selected_frame_indices, training_output_paths, write_frame_manifest,
)
from waymo_preprocess.training_segment_manifest import read_excluded_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-manifest", required=True, type=Path)
    parser.add_argument("--evaluation-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--frame-manifest", required=True, type=Path)
    parser.add_argument("--failures", required=True, type=Path)
    parser.add_argument("--train-segments", type=int, default=8)
    parser.add_argument("--validation-segments", type=int, default=2)
    parser.add_argument("--test-segments", type=int, default=2)
    parser.add_argument("--frames-per-segment", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_existing_rows(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["dataset_role"], row["q_seg"], int(row["q_frame_index"])): row
        for row in rows
    }


def main() -> None:
    args = parse_args()
    limits = {
        "train": args.train_segments,
        "validation": args.validation_segments,
        "test": args.test_segments,
    }
    excluded = read_excluded_segments(args.evaluation_manifest)
    segments = load_training_segments(
        args.segment_manifest, limits, excluded)
    indices = selected_frame_indices(
        args.frames_per_segment, args.frame_stride, args.start_frame)

    plans = []
    skipped = 0
    for segment in segments:
        pending, count = pending_frame_indices(
            args.output_root, segment, indices, args.resume)
        plans.append((segment, pending))
        skipped += count
    pending_count = sum(len(pending) for _, pending in plans)
    summary = {
        "selected_segments": len(segments),
        "segments_by_role": {
            role: sum(segment.dataset_role == role for segment in segments)
            for role in ROLES
        },
        "frames_per_segment": len(indices),
        "planned_frames": len(segments) * len(indices),
        "pending_frames": pending_count,
        "resume_skipped": skipped,
        "frame_stride": args.frame_stride,
        "start_frame": args.start_frame,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.dry_run:
        for segment, pending in plans[:10]:
            preview = ",".join(str(value) for value in pending[:5])
            print(
                f"would_scan={segment.dataset_role}:{segment.tfrecord} "
                f"frame_indices={preview}", flush=True)
        return

    existing = load_existing_rows(args.frame_manifest) if args.resume else {}
    rows = dict(existing)
    failures = []
    exported = 0
    if pending_count:
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2 as open_dataset

        for number, (segment, pending) in enumerate(plans, 1):
            if not pending:
                continue
            wanted = set(pending)
            last_index = max(wanted)
            print(
                f"tfrecord {number}/{len(plans)} role={segment.dataset_role} "
                f"frames={len(wanted)} path={segment.tfrecord}", flush=True)
            try:
                dataset = tf.data.TFRecordDataset(
                    str(segment.tfrecord), compression_type="")
                for index, record in enumerate(dataset):
                    if index in wanted:
                        try:
                            frame = open_dataset.Frame()
                            frame.ParseFromString(record.numpy())
                            timestamp = int(frame.timestamp_micros)
                            item = ExportItem(
                                segment.dataset_role, segment.segment_id,
                                index, timestamp, segment.tfrecord)
                            export_selected_frame(
                                item, frame, args.output_root,
                                dataset_role=segment.dataset_role,
                                source_subset=segment.source_split)
                            npz_path, metadata_path = training_output_paths(
                                args.output_root, segment, index)
                            rows[(segment.dataset_role, segment.segment_id, index)] = (
                                frame_manifest_row(
                                    segment, index, timestamp,
                                    npz_path, metadata_path))
                            exported += 1
                            wanted.remove(index)
                            print(
                                f"exported {exported}/{pending_count} "
                                f"{segment.dataset_role}:{segment.segment_id}#{index}",
                                flush=True)
                        except Exception as exc:
                            failures.append({
                                "q_subset": segment.source_split,
                                "q_seg": segment.segment_id,
                                "q_frame_index": str(index),
                                "source_tfrecord": str(segment.tfrecord),
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                            wanted.discard(index)
                            if args.fail_fast:
                                raise
                    if index >= last_index:
                        break
                for index in sorted(wanted):
                    failures.append({
                        "q_subset": segment.source_split,
                        "q_seg": segment.segment_id,
                        "q_frame_index": str(index),
                        "source_tfrecord": str(segment.tfrecord),
                        "error": f"IndexError: frame {index} not found",
                    })
            except Exception:
                if args.fail_fast:
                    raise
                for index in sorted(wanted):
                    failures.append({
                        "q_subset": segment.source_split,
                        "q_seg": segment.segment_id,
                        "q_frame_index": str(index),
                        "source_tfrecord": str(segment.tfrecord),
                        "error": "RuntimeError: TFRecord scan failed",
                    })

    ordered_rows = [
        rows[key] for key in sorted(rows, key=lambda key: (
            ROLES.index(key[0]), key[1], key[2]))
    ]
    write_frame_manifest(ordered_rows, args.frame_manifest)
    write_failure_rows(args.failures, failures)
    print(json.dumps({
        "exported": exported,
        "resume_skipped": skipped,
        "indexed_frames": len(ordered_rows),
        "failed": len(failures),
        "frame_manifest": str(args.frame_manifest),
        "failures": str(args.failures),
    }, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
