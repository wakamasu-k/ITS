#!/usr/bin/env python3
"""Build Shimizu query-frame correspondence CSV."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from waymo_preprocess.shimizu_frame_manifest import (
    build_frame_rows, parse_path_maps, write_frame_rows,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--line32-root", required=True, type=Path)
    parser.add_argument("--waymo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--allow-missing-tfrecord", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    rows = build_frame_rows(
        args.input_manifest, args.line32_root, args.waymo_root,
        path_maps=parse_path_maps(args.path_map))
    missing = [row for row in rows if row["tfrecord_exists"] != "true"]
    if missing and not args.allow_missing_tfrecord:
        examples = ", ".join(
            f"{row['q_subset']}/{row['q_seg']}" for row in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} frame(s) have no source TFRecord; examples: {examples}. "
            "Use --allow-missing-tfrecord to retain them.")
    write_frame_rows(rows, args.output)
    print(json.dumps({
        "input_manifest": str(args.input_manifest),
        "output": str(args.output),
        "unique_frames": len(rows),
        "source_manifest_rows": sum(int(row["source_manifest_rows"]) for row in rows),
        "missing_tfrecord_frames": len(missing),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
