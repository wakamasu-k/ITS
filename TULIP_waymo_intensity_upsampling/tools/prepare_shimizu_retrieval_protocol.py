#!/usr/bin/env python3
"""Prepare a read-only copy of Shimizu's multi-drive retrieval protocol.

The source manifests and images are never modified.  This tool selects the
segment pairs that actually appear in a reference per-query result, validates
the expected query/database counts, and writes new manifests under an explicit
output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def ordered_unique(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value)
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing columns {missing}; available={list(df.columns)}")


def assert_writable_targets(paths: list[Path], force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        joined = "\n  ".join(existing)
        raise FileExistsError(
            "Refusing to overwrite existing output(s). Use --force only after review:\n  " + joined
        )


def path_audit(paths: Iterable[object]) -> dict[str, int]:
    unique = ordered_unique(paths)
    existing = sum(Path(path).is_file() for path in unique)
    return {"unique": len(unique), "existing": existing, "missing": len(unique) - existing}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and validate Shimizu's evaluated multi-drive retrieval protocol."
    )
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument(
        "--reference-per-query",
        type=Path,
        required=True,
        help="Evaluated per_query CSV used to identify the exact segment-pair set.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="shimizu_test4")
    parser.add_argument("--expected-queries", type=int, default=507)
    parser.add_argument("--expected-db", type=int, default=754)
    parser.add_argument("--skip-path-audit", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_pairs = read_table(args.source_pairs)
    source_db = read_table(args.source_db)
    reference = read_table(args.reference_per_query)
    require_columns(source_pairs, ["pair_id", "cam_png", "anchor_png"], "source pairs")
    require_columns(source_db, ["pair_id", "anchor_png"], "source DB")
    require_columns(reference, ["pair_id", "q_cam_png"], "reference per-query")

    selected_pair_ids = ordered_unique(reference["pair_id"].tolist())
    selected_set = set(selected_pair_ids)
    pairs = source_pairs[source_pairs["pair_id"].astype(str).isin(selected_set)].copy()
    database = source_db[source_db["pair_id"].astype(str).isin(selected_set)].copy()

    # Preserve source order, because it determines q_idx/db_idx in evaluation.
    pairs.reset_index(drop=True, inplace=True)
    database.reset_index(drop=True, inplace=True)

    reference_q = reference["q_cam_png"].astype(str).tolist()
    selected_q = pairs["cam_png"].astype(str).tolist()
    if selected_q != reference_q:
        missing = sorted(set(reference_q) - set(selected_q))
        extra = sorted(set(selected_q) - set(reference_q))
        raise ValueError(
            "Selected query order/content differs from the evaluated reference: "
            f"missing={len(missing)}, extra={len(extra)}, same_set={not missing and not extra}"
        )

    query_count = int(len(pairs))
    db_count = int(len(database))
    if query_count != args.expected_queries or db_count != args.expected_db:
        raise ValueError(
            f"Protocol count mismatch: queries={query_count} (expected {args.expected_queries}), "
            f"DB={db_count} (expected {args.expected_db})"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = args.out_dir / f"{args.prefix}_pairs.csv"
    pairs_parquet = args.out_dir / f"{args.prefix}_pairs.parquet"
    db_csv = args.out_dir / f"{args.prefix}_db.csv"
    db_parquet = args.out_dir / f"{args.prefix}_db.parquet"
    pair_ids_txt = args.out_dir / f"{args.prefix}_pair_ids.txt"
    summary_json = args.out_dir / f"{args.prefix}_summary.json"
    targets = [pairs_csv, pairs_parquet, db_csv, db_parquet, pair_ids_txt, summary_json]
    assert_writable_targets(targets, args.force)

    by_pair: list[dict[str, object]] = []
    for pair_id in selected_pair_ids:
        q_rows = pairs[pairs["pair_id"].astype(str) == pair_id]
        db_rows = database[database["pair_id"].astype(str) == pair_id]
        by_pair.append(
            {
                "pair_id": pair_id,
                "queries": int(len(q_rows)),
                "database": int(len(db_rows)),
                "db_seg": str(q_rows["db_seg"].iloc[0]) if "db_seg" in q_rows and len(q_rows) else "",
                "q_seg": str(q_rows["q_seg"].iloc[0]) if "q_seg" in q_rows and len(q_rows) else "",
            }
        )

    summary: dict[str, object] = {
        "protocol": args.prefix,
        "source_pairs": str(args.source_pairs.resolve()),
        "source_db": str(args.source_db.resolve()),
        "reference_per_query": str(args.reference_per_query.resolve()),
        "queries": query_count,
        "database": db_count,
        "segment_pairs": len(selected_pair_ids),
        "pair_ids": selected_pair_ids,
        "by_pair": by_pair,
    }
    if not args.skip_path_audit:
        summary["query_path_audit"] = path_audit(pairs["cam_png"])
        summary["database_path_audit"] = path_audit(database["anchor_png"])

    pairs.to_csv(pairs_csv, index=False)
    pairs.to_parquet(pairs_parquet, index=False)
    database.to_csv(db_csv, index=False)
    database.to_parquet(db_parquet, index=False)
    pair_ids_txt.write_text("\n".join(selected_pair_ids) + "\n", encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[OK] wrote protocol manifests under: {args.out_dir}")


if __name__ == "__main__":
    main()
