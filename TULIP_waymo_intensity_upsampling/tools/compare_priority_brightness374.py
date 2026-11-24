"""Hold Top-10 and measured depth fixed while changing the LoFTR anchor image."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_retrieval_topk_pipeline import PYTHON, flag, read_csv, write_csv
from run_priority_validation374 import N_QUERY, TOOLS, execute


def prepare(root, output, count):
    rows = read_csv(root / "validation_top10.csv")
    queries = sorted({int(r["retrieval_q_idx"]) for r in rows})
    if len(queries) != N_QUERY or len(rows) != N_QUERY * 10:
        raise ValueError("Expected the validation Top-10 manifest")
    chosen = set(queries[:count])
    anchors = {(r["db_seg"], int(r["anchor_id"])): Path(r["anchor_png"])
               for r in read_csv(root / "retrieval_manifests/A_db.csv")}
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    variants = {name: [] for name in ("A", "priority", "preserve_A_png")}
    verified = {}
    for row in rows:
        if int(row["retrieval_q_idx"]) not in chosen:
            continue
        k = row["db_seg"], int(row["anchor_id"])
        a_path = anchors[k]
        priority_path = Path(row["anchor_png"])
        meta = a_path.with_name("anchor.json")
        depth_path = a_path.with_name("anchor_index.npz")
        if k not in verified:
            a_meta = json.loads(meta.read_text())
            p_meta = json.loads(Path(row["anchor_json"]).read_text())
            for field in ("intrinsic", "world_to_cam", "width", "height"):
                if a_meta[field] != p_meta[field]:
                    raise ValueError(f"Anchor calibration differs: {k}/{field}")
            with np.load(depth_path, allow_pickle=False) as a, np.load(
                row["anchor_index_npz"], allow_pickle=False
            ) as p:
                measured = a["point_index"] >= 0
                priority_valid = p["point_index"] >= 0
                if not np.array_equal(measured, priority_valid & ~p["source_is_generated"]):
                    raise ValueError(f"Measured support differs: {k}")
                if not np.array_equal(a["depth"][measured], p["depth"][measured]):
                    raise ValueError(f"Measured depth differs: {k}")
            original = cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE)
            filled = cv2.imread(str(priority_path), cv2.IMREAD_GRAYSCALE)
            if original is None or filled is None or original.shape != measured.shape or filled.shape != measured.shape:
                raise ValueError(f"Invalid anchor images: {k}")
            filled[measured] = original[measured]
            dest = images / k[0] / f"{k[1]}.png"
            dest.parent.mkdir(exist_ok=True)
            if not cv2.imwrite(str(dest), filled):
                raise OSError(dest)
            verified[k] = (priority_path, dest)
        prior, dest = verified[k]
        if prior != priority_path:
            raise ValueError(f"Ambiguous priority image: {k}")
        for name, path in zip(variants, (a_path, prior, dest)):
            variants[name].append(dict(row, anchor_png=str(path), anchor_json=str(meta),
                                       anchor_index_npz=str(depth_path)))
    for name, table in variants.items():
        if len(table) != count * 10:
            raise ValueError("Missing Top-10 rows")
        folder = output / name
        folder.mkdir(exist_ok=True)
        target = folder / "top10.csv"
        if target.exists():
            if read_csv(target) != table:
                raise ValueError(f"Saved manifest differs: {target}")
        else:
            write_csv(target, table)
    print(f"Prepared {count} queries, {count * 10} pairs per condition, {len(verified)} distinct anchors",
          flush=True)


def paired_transitions(output, count):
    conditions = ("A", "priority", "preserve_A_png")
    tables = {}
    for name in conditions:
        path = output / name / "dense/all507/all/all507_pnp/integrated_per_query.csv"
        rows = read_csv(path)
        tables[name] = {(r["pair_id"], r["q_seg"], r["q_frame_index"]): r for r in rows}
        if len(rows) != count or len(tables[name]) != count:
            raise ValueError(f"Unexpected query count: {path}")
    keys = tables["A"].keys()
    if any(table.keys() != keys for table in tables.values()):
        raise ValueError("Query identities differ between image variants")
    groups = [("ALL", list(keys))]
    groups.extend((pair, [key for key in keys if key[0] == pair])
                  for pair in sorted({key[0] for key in keys}))
    results = []
    after = tables["preserve_A_png"]
    for before_name in ("priority", "A"):
        before = tables[before_name]
        for pair, group in groups:
            for threshold in (.5, 1, 2):
                field = f"success_xy_{threshold}m"
                changes = [(flag(before[k][field]), flag(after[k][field])) for k in group]
                improved = sum(not a and b for a, b in changes)
                worsened = sum(a and not b for a, b in changes)
                results.append(dict(before=before_name, after="preserve_A_png", pair_id=pair,
                                    queries=len(group), threshold_m=threshold,
                                    improved=improved, worsened=worsened, net=improved-worsened))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=N_QUERY)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.queries <= N_QUERY:
        parser.error("queries out of range")
    prepare(args.root.resolve(), args.output.resolve(), args.queries)
    if args.prepare_only:
        return
    results = []
    for name in ("A", "priority", "preserve_A_png"):
        folder = args.output / name
        dense = folder / "dense"
        summary = dense / "all507/all/all507_pnp/integrated_summary.json"
        if not summary.is_file():
            execute([PYTHON, "-u", TOOLS / "audit_priority_pnp_sources.py",
                     "--manifest", folder / "top10.csv", "--output", dense,
                     "--scope", "all507", "--expected-queries", args.queries, "--all-only"])
        data = json.loads(summary.read_text())
        if (data["queries"], data["candidate_pairs"]) != (args.queries, args.queries * 10):
            raise ValueError(f"Unexpected evaluation size: {summary}")
        result = dict(condition=name, queries=args.queries, all_failed=data["all_candidates_failed"],
                      median_xy_m=data["xy_median_success_only"],
                      **{f"xy_{m:g}m_count": round(args.queries * data[f"success_xy_{m}m_all_queries"])
                         for m in (.5, 1, 2, 5)})
        results.append(result)
        print(result, flush=True)
    target = args.output / "comparison.csv"
    if not target.exists():
        write_csv(target, results)
    transitions = args.output / "transitions.csv"
    if not transitions.exists():
        write_csv(transitions, paired_transitions(args.output, args.queries))
    print(target)
    print(transitions)


if __name__ == "__main__":
    main()
