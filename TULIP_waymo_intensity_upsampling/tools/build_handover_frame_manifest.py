#!/usr/bin/env python3
"""Convert a handover pair manifest to one row per Waymo frame."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from waymo_preprocess.handover_frame_manifest import (
    build_rows, iter_waymo_frame_identities, parse_path_maps, verify_rows, write_csv,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--waymo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--line32-root", type=Path)
    parser.add_argument("--metadata-column", default="cam_json")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--verify-tfrecord", action="store_true",
                        help="read TFRecords and verify context/index/timestamp")
    parser.add_argument("--allow-missing-tfrecord", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    rows = build_rows(args.input_manifest, args.waymo_root,
                      metadata_column=args.metadata_column,
                      path_maps=parse_path_maps(args.path_map),
                      line32_root=args.line32_root)
    missing = [row for row in rows if not row["source_tfrecord"]]
    if missing and not args.allow_missing_tfrecord:
        examples = ", ".join(row["segment_id"] for row in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} frame(s) have no source TFRecord; examples: {examples}. "
            "Use --allow-missing-tfrecord to retain them.")
    if args.verify_tfrecord:
        verify_rows(rows, iter_waymo_frame_identities)
    write_csv(rows, args.output)
    print(json.dumps({"input_manifest": str(args.input_manifest),
                      "output": str(args.output), "frame_rows": len(rows),
                      "missing_tfrecord_rows": len(missing),
                      "tfrecord_verified": bool(args.verify_tfrecord)},
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
