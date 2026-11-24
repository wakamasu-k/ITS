#!/usr/bin/env python3
"""Remap the fixed 507-query/9759-pair distance protocol to B anchors.

The reference manifest and anchors are read only. The original row order,
query data, distances, and GT flags are preserved; only anchor file paths change.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_rows(path, fields, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--b-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    fields, source = read_rows(args.source)
    required = {"pair_id", "db_seg", "q_seg", "db_subset", "anchor_id",
                "q_image_png", "anchor_png", "anchor_json", "anchor_index_npz",
                "dist_xy_m", "is_strict_gt", "is_gt_t10m"}
    if not required.issubset(fields or []):
        raise ValueError(f"Missing source columns: {sorted(required - set(fields or []))}")
    if len(source) != 9759:
        raise ValueError(f"Expected 9759 reference pairs, got {len(source)}")

    output = []
    queries = set()
    strict_queries = set()
    checked_anchors = set()
    for number, old in enumerate(source, 1):
        key = (old["pair_id"], old["q_image_png"])
        queries.add(key)
        if old["is_strict_gt"].lower() in ("true", "1"):
            strict_queries.add(key)
        if float(old["dist_xy_m"]) > 10.000001:
            raise ValueError(f"Row {number}: distance exceeds 10 m")
        folder = args.b_root / old["db_subset"] / old["db_seg"] / str(int(old["anchor_id"]))
        new_paths = {"anchor_png": folder / "anchor.png",
                     "anchor_json": folder / "anchor.json",
                     "anchor_index_npz": folder / "anchor_index.npz"}
        if folder not in checked_anchors:
            for label, path in new_paths.items():
                if not path.is_file():
                    raise FileNotFoundError(f"Row {number}: {label}: {path}")
            original_meta = json.loads(Path(old["anchor_json"]).read_text())
            b_meta = json.loads(new_paths["anchor_json"].read_text())
            for name, expected in (("segment_id", old["db_seg"]),
                                   ("submap_id", int(old["anchor_id"]))):
                if original_meta[name] != expected or b_meta[name] != expected:
                    raise ValueError(f"Row {number}: {name} mismatch")
            if original_meta.get("subset") != b_meta.get("subset"):
                raise ValueError(f"Row {number}: subset mismatch")
            if not np.allclose(original_meta["anchor_pose_T_wv"],
                               b_meta["anchor_pose_T_wv"], rtol=0, atol=1e-6):
                raise ValueError(f"Row {number}: anchor pose mismatch")
            if (original_meta["width"], original_meta["height"]) != (b_meta["width"], b_meta["height"]):
                raise ValueError(f"Row {number}: image dimensions mismatch")
            checked_anchors.add(folder)
        row = old.copy()
        row.update({name: str(path.resolve()) for name, path in new_paths.items()})
        output.append(row)

    if len(queries) != 507 or len(strict_queries) != 507:
        raise ValueError(f"Expected 507 queries and strict pairs, got {len(queries)}, {len(strict_queries)}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    strict = [row for row in output if row["is_strict_gt"].lower() in ("true", "1")]
    write_rows(args.output_dir / "B_all_t10m_9759.csv", fields, output)
    write_rows(args.output_dir / "B_strict_507.csv", fields, strict)
    summary = {"source": str(args.source.resolve()), "b_root": str(args.b_root.resolve()),
               "queries": len(queries), "all_pairs": len(output), "strict_pairs": len(strict),
               "verified_unique_anchors": len(checked_anchors),
               "changed_fields": list(new_paths), "verified": "paths, ID, subset, pose, dimensions"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
