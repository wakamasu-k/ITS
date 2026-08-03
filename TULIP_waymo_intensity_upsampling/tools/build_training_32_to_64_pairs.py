#!/usr/bin/env python3
"""Build role-preserving Waymo training pairs from extracted 64/32 frames."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.prepare_32_to_64_pair import build_pair
from waymo_preprocess.export_shimizu_frames_64_32 import atomic_json, atomic_npz
from waymo_preprocess.training_pair_build import (
    INDEX_COLUMNS, ROLES, completed_pair, index_row, load_training_frames,
    pair_paths, write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-index", required=True, type=Path)
    parser.add_argument("--pair-index", required=True, type=Path)
    parser.add_argument("--failures", required=True, type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = load_training_frames(args.frame_index, args.max_frames)
    complete = sum(completed_pair(frame) for frame in frames)
    summary = {
        "selected_frames": len(frames),
        "frames_by_role": {
            role: sum(frame.dataset_role == role for frame in frames)
            for role in ROLES
        },
        "complete_pairs": complete,
        "pending_pairs": len(frames) - complete if args.resume else len(frames),
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.dry_run:
        for frame in frames[:10]:
            pair_npz, _ = pair_paths(frame)
            print(
                f"would_pair={frame.dataset_role}:{frame.frame_npz} "
                f"-> {pair_npz}", flush=True)
        return

    rows = []
    failures = []
    built = 0
    skipped = 0
    for position, frame in enumerate(frames, 1):
        pair_npz, pair_json = pair_paths(frame)
        try:
            if args.resume and completed_pair(frame):
                metadata = json.loads(pair_json.read_text(encoding="utf-8"))
                skipped += 1
            else:
                with np.load(frame.frame_npz, allow_pickle=False) as data:
                    arrays, metadata = build_pair(data)
                metadata.update({
                    "dataset_role": frame.dataset_role,
                    "subset": frame.source_subset,
                    "segment_id": frame.segment_id,
                    "frame_index": frame.frame_index,
                    "timestamp_micros": frame.timestamp_micros,
                    "source_tfrecord": str(frame.source_tfrecord),
                    "source_frame_npz": str(frame.frame_npz),
                })
                atomic_npz(pair_npz, arrays)
                atomic_json(pair_json, metadata)
                built += 1
            rows.append(index_row(frame, metadata))
            print(
                f"paired {position}/{len(frames)} "
                f"{frame.dataset_role}:{frame.segment_id}#{frame.frame_index}",
                flush=True)
        except Exception as exc:
            failures.append({
                "dataset_role": frame.dataset_role,
                "q_subset": frame.source_subset,
                "q_seg": frame.segment_id,
                "q_frame_index": str(frame.frame_index),
                "error": f"{type(exc).__name__}: {exc}",
            })

    rows.sort(key=lambda row: (
        ROLES.index(row["dataset_role"]), row["q_seg"],
        int(row["q_frame_index"])))
    write_rows(rows, args.pair_index, INDEX_COLUMNS)
    write_rows(
        failures, args.failures,
        ("dataset_role", "q_subset", "q_seg", "q_frame_index", "error"))
    print(json.dumps({
        "selected_frames": len(frames),
        "indexed_pairs": len(rows),
        "built_pairs": built,
        "resume_skipped": skipped,
        "failed_pairs": len(failures),
        "pair_index": str(args.pair_index),
        "failures": str(args.failures),
    }, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
