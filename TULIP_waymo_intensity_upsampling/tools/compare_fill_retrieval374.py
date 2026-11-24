"""Evaluate 374-query Retrieval with Closing and preserved measured brightness."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluate_retrieval_topk_pipeline import flag, read_csv, write_csv
from run_priority_validation374 import N_DB, N_QUERY, TOOLS, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, baseline, output = (path.resolve() for path in (args.root, args.baseline, args.output))
    db = pd.read_parquet(baseline / "A/db.parquet")
    pairs = pd.read_parquet(baseline / "A/pairs.parquet")
    priority = pd.read_parquet(root / "database/db.parquet")
    ids = ["pair_id", "db_seg", "anchor_id"]
    if (len(pairs), len(db)) != (N_QUERY, N_DB) or not db[ids].astype(str).equals(
        priority[ids].astype(str)
    ):
        raise ValueError("Validation anchor identities or counts differ")
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "A_inputs"
    inputs.mkdir(exist_ok=True)
    for frame, kind in ((db, "db"), (pairs, "pairs")):
        frame.to_parquet(inputs / f"A_{kind}.parquet", index=False)
    closing = output / "closing"
    execute([sys.executable, TOOLS / "prepare_observed16_bilinear_close3.py", inputs, closing])

    preserved = output / "preserve_A_png"
    paths = {}
    for i, (raw_a, raw_p) in enumerate(zip(db.anchor_png, priority.anchor_png)):
        a, p = Path(raw_a), Path(raw_p)
        with np.load(a.with_name("anchor_index.npz"), allow_pickle=False) as z:
            observed = z["point_index"] >= 0
        original = cv2.imread(str(a), cv2.IMREAD_GRAYSCALE)
        filled = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if original is None or filled is None or original.shape != observed.shape or filled.shape != observed.shape:
            raise ValueError(f"Invalid anchor images: {a}, {p}")
        filled[observed] = original[observed]
        dest = preserved / "images" / f"{i:04d}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dest), filled):
            raise OSError(dest)
        paths[raw_a] = str(dest)
    for frame, kind in ((db, "db"), (pairs, "pairs")):
        frame = frame.assign(anchor_png=frame.anchor_png.map(paths))
        if frame.anchor_png.isna().any():
            raise ValueError("A ground-truth anchor is missing")
        frame.to_parquet(preserved / f"{kind}.parquet", index=False)

    command = json.loads((root / "retrieval/command.json").read_text())
    folders = {"A": baseline / "A/retrieval"}
    for folder in (closing, preserved):
        result = folder / "retrieval"
        cmd = command.copy()
        for option, value in (("--pairs", folder / "pairs.parquet"), ("--db", folder / "db.parquet"),
                              ("--out_dir", result), ("--name", folder.name)):
            cmd[cmd.index(option) + 1] = str(value)
        execute(cmd)
        folders[folder.name] = result

    rows = []
    query_keys = ("q_idx", "pair_id", "q_cam_png", "gt_index_db")
    reference = None
    for name, folder in folders.items():
        queries = read_csv(folder / f"{name}_per_query.csv")
        if len(queries) != N_QUERY:
            raise ValueError(f"Unexpected number of queries: {name}")
        keys = [tuple(r[k] for k in query_keys) for r in queries]
        if reference is None:
            reference = keys
        if keys != reference:
            raise ValueError(f"Query identities or GT indices differ: {name}")
        groups = [("ALL", queries)] + [
            (pair, [r for r in queries if r["pair_id"] == pair])
            for pair in sorted({r["pair_id"] for r in queries})]
        for pair, group in groups:
            rows.append(dict(condition=name, pair_id=pair, queries=len(group),
                             **{field: sum(flag(r[field]) for r in group) for field in
                                ("strict_hit@10", "soft_t2m_hit@10", "soft_t5m_hit@10")}))
    write_csv(output / "retrieval_comparison.csv", rows)
    print(pd.DataFrame(rows).to_string(index=False))
    print(output / "retrieval_comparison.csv")


if __name__ == "__main__":
    main()
