#!/usr/bin/env python3
"""Create segment-wise train/validation/test splits without reading TFRecords."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waymo_preprocess.training_segment_manifest import (
    discover_tfrecords, read_excluded_segments, split_segments,
    write_segment_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waymo-root", required=True, type=Path)
    parser.add_argument("--evaluation-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-split", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_splits = tuple(args.source_split or ("training",))
    excluded = read_excluded_segments(args.evaluation_manifest)
    records = discover_tfrecords(args.waymo_root, source_splits)
    rows, summary = split_segments(
        records, excluded_segments=excluded, seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction, max_segments=args.max_segments)
    summary.update({
        "waymo_root": str(args.waymo_root),
        "evaluation_manifest": str(args.evaluation_manifest),
        "output": str(args.output),
        "seed": args.seed,
        "dry_run": args.dry_run,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.dry_run:
        write_segment_rows(rows, args.output)


if __name__ == "__main__":
    main()
