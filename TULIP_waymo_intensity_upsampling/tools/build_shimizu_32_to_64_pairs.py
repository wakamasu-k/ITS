#!/usr/bin/env python3
"""Build traceable 32-line input -> 64-line target pairs in batches."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_32_to_64_pair import build_pair
from waymo_preprocess.export_shimizu_frames_64_32 import atomic_json, atomic_npz
from waymo_preprocess.shimizu_batch_export import load_export_items, output_paths

INDEX_FIELDS = (
    "dataset_role", "q_subset", "q_seg", "q_frame_index",
    "timestamp_micros", "pair_npz", "pair_json", "input_height",
    "target_height", "width", "channels", "observed_target_rows",
    "generated_target_rows",
)

def pair_paths(frame_npz: Path) -> tuple[Path, Path]:
    return frame_npz.with_name("tulip_pair_32_to_64.npz"), frame_npz.with_name(
        "tulip_pair_32_to_64.json")

def pair_complete(npz_path: Path, json_path: Path, item) -> bool:
    if not npz_path.is_file() or not json_path.is_file():
        return False
    try:
        value = json.loads(json_path.read_text(encoding="utf-8"))
        return (
            value["segment_id"] == item.segment_id
            and int(value["frame_index"]) == item.frame_index
            and int(value["timestamp_micros"]) == item.timestamp_micros
            and value["task"] == "Waymo TOP LiDAR 32-to-64 range and intensity upsampling"
            and bool(value["exact_subset_verified"])
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False

def index_row(item, npz_path: Path, json_path: Path,
              metadata: dict, dataset_role: str) -> dict[str, str]:
    return {
        "dataset_role": dataset_role,
        "q_subset": item.subset,
        "q_seg": item.segment_id,
        "q_frame_index": str(item.frame_index),
        "timestamp_micros": str(item.timestamp_micros),
        "pair_npz": str(npz_path),
        "pair_json": str(json_path),
        "input_height": str(metadata["input_shape"][0]),
        "target_height": str(metadata["target_shape"][0]),
        "width": str(metadata["input_shape"][1]),
        "channels": str(metadata["input_shape"][2]),
        "observed_target_rows": "0:64:2",
        "generated_target_rows": "1:64:2",
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--index-output", required=True, type=Path)
    parser.add_argument("--failures", required=True, type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dataset-role", default="localization_evaluation",
        choices=("localization_evaluation", "train", "validation", "test"))
    return parser.parse_args()

def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    args = parse_args()
    items = load_export_items(args.manifest, args.max_frames)
    if args.dry_run:
        print(json.dumps({
            "selected_frames": len(items), "dataset_role": args.dataset_role,
            "would_build": sum(
                output_paths(args.input_root, item)[0].is_file() for item in items),
        }, ensure_ascii=False, indent=2))
        return
    rows = []
    failures = []
    built = 0
    skipped = 0
    for position, item in enumerate(items, 1):
        frame_npz, _ = output_paths(args.input_root, item)
        pair_npz, pair_json = pair_paths(frame_npz)
        try:
            if args.resume and pair_complete(pair_npz, pair_json, item):
                metadata = json.loads(pair_json.read_text(encoding="utf-8"))
                skipped += 1
            else:
                with np.load(frame_npz, allow_pickle=False) as data:
                    arrays, metadata = build_pair(data)
                metadata.update({
                    "dataset_role": args.dataset_role,
                    "subset": item.subset,
                    "segment_id": item.segment_id,
                    "frame_index": item.frame_index,
                    "timestamp_micros": item.timestamp_micros,
                    "source_frame_npz": str(frame_npz),
                })
                atomic_npz(pair_npz, arrays)
                atomic_json(pair_json, metadata)
                built += 1
            rows.append(index_row(
                item, pair_npz, pair_json, metadata, args.dataset_role))
            print(f"paired {position}/{len(items)} "
                  f"{item.segment_id}#{item.frame_index}", flush=True)
        except Exception as exc:
            failures.append({
                "q_subset": item.subset, "q_seg": item.segment_id,
                "q_frame_index": str(item.frame_index),
                "error": f"{type(exc).__name__}: {exc}",
            })
    write_csv(args.index_output, INDEX_FIELDS, rows)
    write_csv(args.failures,
              ("q_subset", "q_seg", "q_frame_index", "error"), failures)
    summary = {
        "selected_frames": len(items), "indexed_pairs": len(rows),
        "built_pairs": built, "resume_skipped": skipped,
        "failed_pairs": len(failures), "dataset_role": args.dataset_role,
        "index_output": str(args.index_output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
