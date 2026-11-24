#!/usr/bin/env python3
"""Create A/B/C retrieval manifests from one fixed Shimizu protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def remap_anchor_path(row: pd.Series, root: Path) -> str:
    return str(root / str(row["db_subset"]) / str(row["db_seg"]) / str(int(row["anchor_id"])) / "anchor.png")


def write_pair(root: Path, prefix: str, pairs: pd.DataFrame, database: pd.DataFrame) -> None:
    pairs.to_csv(root / f"{prefix}_pairs.csv", index=False)
    pairs.to_parquet(root / f"{prefix}_pairs.parquet", index=False)
    database.to_csv(root / f"{prefix}_db.csv", index=False)
    database.to_parquet(root / f"{prefix}_db.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pairs", required=True, type=Path)
    parser.add_argument("--source-db", required=True, type=Path)
    parser.add_argument("--a-root", required=True, type=Path)
    parser.add_argument("--b-root", required=True, type=Path)
    parser.add_argument("--c-root", type=Path, help="Optional C root; omit to retain reference C paths.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pairs_source = read_table(args.source_pairs)
    db_source = read_table(args.source_db)
    required = {"pair_id", "db_subset", "db_seg", "anchor_id", "anchor_png"}
    for label, frame in (("pairs", pairs_source), ("DB", db_source)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} missing columns: {missing}")

    variants: dict[str, Path | None] = {"A": args.a_root, "B": args.b_root, "C": args.c_root}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"queries": int(len(pairs_source)), "database": int(len(db_source)), "variants": {}}

    for variant, anchor_root in variants.items():
        pairs = pairs_source.copy()
        database = db_source.copy()
        if anchor_root is not None:
            anchor_root = anchor_root.resolve()
            pairs["anchor_png"] = pairs.apply(remap_anchor_path, axis=1, root=anchor_root)
            database["anchor_png"] = database.apply(remap_anchor_path, axis=1, root=anchor_root)
        missing_paths = [path for path in database["anchor_png"].astype(str) if not Path(path).is_file()]
        if missing_paths:
            raise FileNotFoundError(f"variant {variant}: missing {len(missing_paths)} DB images; first={missing_paths[0]}")
        output_paths = [
            args.output_dir / f"{variant}_pairs.csv",
            args.output_dir / f"{variant}_pairs.parquet",
            args.output_dir / f"{variant}_db.csv",
            args.output_dir / f"{variant}_db.parquet",
        ]
        existing = [str(path) for path in output_paths if path.exists()]
        if existing and not args.force:
            raise FileExistsError(f"Refusing to overwrite variant {variant}: {existing}")
        write_pair(args.output_dir, variant, pairs, database)
        summary["variants"][variant] = {
            "anchor_root": "reference paths" if anchor_root is None else str(anchor_root.resolve()),
            "query_rows": int(len(pairs)),
            "db_rows": int(len(database)),
            "unique_query_images": int(pairs["cam_png"].nunique()),
            "unique_db_images": int(database["anchor_png"].nunique()),
            "missing_db_images": 0,
        }

    summary_path = args.output_dir / "summary.json"
    if summary_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite: {summary_path}")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
