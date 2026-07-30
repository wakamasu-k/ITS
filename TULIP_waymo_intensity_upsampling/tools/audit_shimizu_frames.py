#!/usr/bin/env python3
"""Audit 64/32 exports and optionally prepare TULIP 16-to-32 pairs."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from waymo_preprocess.export_shimizu_frames_64_32 import atomic_json, atomic_npz
from waymo_preprocess.shimizu_batch_export import load_export_items, output_paths
from waymo_preprocess.tulip_frame_quality import (
    audit_frame, build_tulip_16_32,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--prepare-16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()

def pair_paths(npz_path: Path) -> tuple[Path, Path]:
    return npz_path.with_name("tulip_16_32.npz"), npz_path.with_name(
        "tulip_16_32.json")

def pair_complete(npz_path: Path, metadata_path: Path, item) -> bool:
    if not npz_path.is_file() or not metadata_path.is_file():
        return False
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (value["segment_id"] == item.segment_id
                and int(value["frame_index"]) == item.frame_index
                and int(value["timestamp_micros"]) == item.timestamp_micros)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False

def main() -> None:
    args = parse_args()
    items = load_export_items(args.manifest, args.max_frames)
    per_frame = []
    failures = []
    prepared = 0
    pair_skipped = 0
    for position, item in enumerate(items, 1):
        npz_path, metadata_path = output_paths(args.input_root, item)
        try:
            stats = audit_frame(npz_path, metadata_path, item)
            per_frame.append({
                "q_subset": item.subset, "q_seg": item.segment_id,
                "q_frame_index": item.frame_index,
                "timestamp_micros": item.timestamp_micros,
                "width": stats["width"],
                "valid_count_64": stats["valid_count_64"],
                "valid_count_32": stats["valid_count_32"],
                "range_median_32": stats["range_32"]["median"],
                "range_p99_5_32": stats["range_32"]["p99_5"],
                "intensity_median_32": stats["intensity_32"]["median"],
                "intensity_p99_5_32": stats["intensity_32"]["p99_5"],
                "intensity_max_32": stats["intensity_32"]["max"],
            })
            if args.prepare_16:
                pair_npz, pair_json = pair_paths(npz_path)
                if args.resume and pair_complete(pair_npz, pair_json, item):
                    pair_skipped += 1
                else:
                    with np.load(npz_path, allow_pickle=False) as loaded:
                        arrays = {name: loaded[name] for name in loaded.files}
                    pair = build_tulip_16_32(arrays)
                    atomic_npz(pair_npz, pair)
                    atomic_json(pair_json, {
                        "segment_id": item.segment_id,
                        "subset": item.subset,
                        "frame_index": item.frame_index,
                        "timestamp_micros": item.timestamp_micros,
                        "input_shape": list(pair["range_16"].shape),
                        "target_shape": list(pair["range_32"].shape),
                        "ring_ids_16": pair["ring_ids_16"].tolist(),
                        "ring_ids_32": pair["ring_ids_32"].tolist(),
                        "range_normalization": "none",
                        "intensity_normalization": "none",
                    })
                    prepared += 1
            print(f"audited {position}/{len(items)} "
                  f"{item.segment_id}#{item.frame_index}", flush=True)
        except Exception as exc:
            failures.append({
                "q_subset": item.subset, "q_seg": item.segment_id,
                "q_frame_index": item.frame_index,
                "error": f"{type(exc).__name__}: {exc}",
            })
    args.report_dir.mkdir(parents=True, exist_ok=True)
    per_frame_path = args.report_dir / "per_frame.csv"
    fields = list(per_frame[0]) if per_frame else [
        "q_subset", "q_seg", "q_frame_index", "error"]
    with per_frame_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_frame)
    failure_path = args.report_dir / "failures.csv"
    with failure_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["q_subset", "q_seg", "q_frame_index", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)
    summary = {
        "selected_frames": len(items), "audited_frames": len(per_frame),
        "failed_frames": len(failures), "prepared_16_32": prepared,
        "prepare_resume_skipped": pair_skipped,
        "max_intensity_32": max(
            (row["intensity_max_32"] for row in per_frame), default=None),
        "max_frame_p99_5_intensity_32": max(
            (row["intensity_p99_5_32"] for row in per_frame), default=None),
    }
    atomic_json(args.report_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
