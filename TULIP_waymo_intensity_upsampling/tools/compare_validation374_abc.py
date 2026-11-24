"""Compare measured A, original TULIP B, and observed-priority on the same split."""
import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_retrieval_topk_pipeline import PYTHON, candidate_manifest, flag, read_csv, write_csv
from run_priority_validation374 import N_DB, N_QUERY, SOURCE, TOOLS, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--include-c", action="store_true", help="Also evaluate the original measured32 DB")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    source = root / "retrieval_manifests"
    pairs = pd.read_csv(source / "B_pairs.csv")
    original = pd.read_parquet(root / "database/pairs.parquet")
    ids = ["pair_id", "db_seg", "anchor_id"]
    query_ids = ids + ["q_seg", "cam_png"]
    original_db = pd.read_parquet(root / "database/db.parquet")
    if len(pairs) != N_QUERY or not pairs[query_ids].astype(str).equals(original[query_ids].astype(str)):
        raise ValueError("The validation queries differ")
    retrieval = json.loads((root / "retrieval/command.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    evaluations = {"observed_priority": (root / "retrieval", root / "dense/all507/all")}
    conditions = ("A", "B", "C") if args.include_c else ("A", "B")
    for name in conditions:
        target = output / name
        target.mkdir(exist_ok=True)
        db = (pd.read_parquet(SOURCE / "val_db_source.parquet") if name == "C"
              else pd.read_csv(source / f"{name}_db.csv"))
        if len(db) != N_DB or not db[ids].astype(str).equals(original_db[ids].astype(str)):
            raise ValueError(f"{name} DB identities or order differ")
        if db.duplicated(ids).any():
            raise ValueError(f"{name} has duplicate anchors")
        q = pairs.drop(columns="anchor_png").merge(db[ids + ["anchor_png"]], on=ids,
                                                   how="left", validate="many_to_one")
        if q.anchor_png.isna().any():
            raise ValueError(f"{name} is missing GT anchors")
        for row_id, raw in enumerate(db.anchor_png):
            image = Path(raw)
            for path in (image, image.with_name("anchor.json"), image.with_name("anchor_index.npz")):
                if not path.is_file():
                    raise FileNotFoundError(path)
            if name == "C":
                original_image = Path(original_db.iloc[row_id]["anchor_png"])
                reference = json.loads(original_image.with_name("anchor.json").read_text())
                measured32 = json.loads(image.with_name("anchor.json").read_text())
                import numpy as np
                for field in ("intrinsic", "world_to_cam", "width", "height"):
                    if not np.allclose(reference[field], measured32[field], rtol=0, atol=1e-6):
                        raise ValueError(f"C camera geometry differs: {image}/{field}")
        for frame, filename in ((db, "db.parquet"), (q, "pairs.parquet")):
            path = target / filename
            if path.exists():
                if not pd.read_parquet(path).equals(frame):
                    raise ValueError(f"Prepared manifest differs: {path}")
            else:
                frame.to_parquet(path, index=False)
        if args.prepare_only:
            print(f"{name}: verified {len(q)} queries, {len(db)} anchors")
            continue
        retrieval_out = target / "retrieval"
        topk = retrieval_out / f"{name}_topk_top50.jsonl"
        metrics = retrieval_out / f"{name}_metrics.json"
        if not (metrics.is_file() and topk.is_file()):
            command = retrieval.copy()
            for option, value in (("--pairs", target / "pairs.parquet"),
                                  ("--db", target / "db.parquet"),
                                  ("--out_dir", retrieval_out), ("--name", name)):
                command[command.index(option) + 1] = str(value)
            execute(command)
        _, _, candidates = candidate_manifest(topk, 10, N_QUERY)
        manifest = target / "top10.csv"
        if manifest.exists():
            if read_csv(manifest) != [{k: str(v) for k, v in r.items()} for r in candidates]:
                raise ValueError(f"Candidate manifest differs: {manifest}")
        else:
            write_csv(manifest, candidates)
        dense = target / "dense"
        result = dense / "all507/all"
        if not (result / "all507_pnp/integrated_summary.json").is_file():
            execute([PYTHON, "-u", TOOLS / "audit_priority_pnp_sources.py",
                     "--manifest", manifest, "--output", dense, "--scope", "all507",
                     "--expected-queries", N_QUERY, "--all-only"])
        evaluations[name] = (retrieval_out, result)

    if args.prepare_only:
        return
    rows, by_query = [], {}
    for name in (*conditions, "observed_priority"):
        retrieval_out, dense = evaluations[name]
        r = json.loads((retrieval_out / f"{name}_metrics.json").read_text())
        d = json.loads((dense / "all507_pnp/integrated_summary.json").read_text())
        if (r["strict"]["Q"], r["strict"]["DB"], d["queries"], d["candidate_pairs"]) != (
                N_QUERY, N_DB, N_QUERY, N_QUERY * 10):
            raise ValueError(f"Unexpected run size: {name}")
        by_query[name] = {(x["pair_id"], x["q_seg"], x["q_frame_index"]): x
                          for x in read_csv(dense / "all507_pnp/integrated_per_query.csv")}
        rows.append(dict(condition=name, strict_r10=r["strict"]["R@"]["10"],
                         soft5_r10=r["soft"]["R_soft"]["5.0"]["R@"]["10"],
                         **{f"xy_{m:g}m_count": round(N_QUERY * d[f"success_xy_{m}m_all_queries"])
                            for m in (.5, 1, 2, 5)},
                         all_failed=d["all_candidates_failed"],
                         median_xy_m=d["xy_median_success_only"]))
    table = pd.DataFrame(rows)
    table.to_csv(output / "comparison.csv", index=False)
    print(table.to_string(index=False))
    transitions = []
    comparisons = [("A", "B"), ("B", "observed_priority"), ("A", "observed_priority")]
    if args.include_c:
        comparisons.extend([("A", "C"), ("observed_priority", "C")])
    for old, new in comparisons:
        a, b = by_query[old], by_query[new]
        if a.keys() != b.keys() or len(a) != N_QUERY:
            raise ValueError("Query identities differ")
        for m in (1, 2, 5):
            field = f"success_xy_{m}m"
            improved = sum(not flag(a[k][field]) and flag(b[k][field]) for k in a)
            worsened = sum(flag(a[k][field]) and not flag(b[k][field]) for k in a)
            transitions.append(dict(before=old, after=new, threshold_m=m,
                                    improved=improved, worsened=worsened, net=improved-worsened))
    table = pd.DataFrame(transitions)
    table.to_csv(output / "transitions.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
