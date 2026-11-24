"""Compare saved generated-depth ablations on the same queries and candidates."""
import argparse
import math
from collections import Counter
from pathlib import Path

from evaluate_retrieval_topk_pipeline import flag, read_csv, write_csv


def query_key(row):
    return row["pair_id"], row["q_seg"], int(row["q_frame_index"])


def candidate_key(row):
    return (*query_key(row), row["db_seg"], int(row["anchor_id"]))


def load(root):
    queries = read_csv(root / "all507_pnp/integrated_per_query.csv")
    pairs = read_csv(root / "all507_pnp/per_pair.csv")
    manifest = read_csv(root / "all507.csv")
    q = {query_key(r): r for r in queries}
    p = {candidate_key(r): r for r in pairs}
    m = {candidate_key(r): r for r in manifest}
    if not (len(q) == len(queries) == 507 and len(pairs) == len(p) == len(m) == len(manifest) == 5070):
        raise ValueError(f"Unexpected evaluation size or duplicate keys: {root}")
    if p.keys() != m.keys() or Counter(k[:3] for k in p) != {k: 10 for k in q}:
        raise ValueError(f"Candidate coverage differs: {root}")
    return q, p, m


def hit(pair, manifest, threshold):
    if not (flag(pair["pnp_success"]) and flag(pair["comparable"])
            and flag(manifest["retrieval_is_same_pair"])):
        return False
    value = float(pair["t_err_xy"] or "nan")
    return math.isfinite(value) and value <= threshold


def selected(row):
    if not flag(row["pnp_returned"]):
        return None
    return (*query_key(row), row["selected_db_seg"], int(row["selected_anchor_id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filtered", type=Path, help="Compare all generated points against this filtered result root")
    args = parser.parse_args()
    before_name, after_name = ("all", "filtered") if args.filtered else ("measured", "all")
    full_root = args.run / "all507/all"
    before_root = full_root if args.filtered else args.run / "all507/measured_only"
    before_queries, before_pairs, manifest = load(before_root)
    after_queries, after_pairs, after_manifest = load(args.filtered or full_root)
    if before_queries.keys() != after_queries.keys() or manifest != after_manifest:
        raise ValueError("The two evaluations do not use the same queries and candidate manifest")
    arms = {before_name: before_pairs, after_name: after_pairs}
    print(f"Comparison: {before_name} -> {after_name}")

    for threshold in (0.5, 1, 2, 5, 10):
        field = f"success_xy_{threshold}m"
        transitions = Counter((flag(before_queries[k][field]), flag(after_queries[k][field]))
                              for k in before_queries)
        oracle = {name: len({k[:3] for k, pair in pairs.items()
                            if hit(pair, manifest[k], threshold)}) for name, pairs in arms.items()}
        print(f"{threshold:g} m: improved={transitions[False, True]}, "
              f"worsened={transitions[True, False]}, same_success={transitions[True, True]}, "
              f"same_failure={transitions[False, False]}, oracle={oracle}")

    records = []
    for k, before in before_queries.items():
        after = after_queries[k]
        old_hit, new_hit = flag(before["success_xy_1m"]), flag(after["success_xy_1m"])
        if old_hit == new_hit:
            continue
        old, new = selected(before), selected(after)
        if old is None:
            selection = "pnp_recovered"
        elif new is None:
            selection = "pnp_lost"
        else:
            selection = "same_anchor" if old == new else "changed_anchor"
        record = dict(query_index=before["query_index"], pair_id=k[0], q_seg=k[1],
                      q_frame_index=k[2], change="improved" if new_hit else "worsened",
                      selection=selection)
        for role, candidate in (("old", old), ("new", new)):
            record[f"{role}_db_seg"] = candidate[-2] if candidate else ""
            record[f"{role}_anchor_id"] = candidate[-1] if candidate else ""
            for arm, pairs in arms.items():
                pair = pairs[candidate] if candidate else {}
                prefix = f"{role}_{arm}"
                record[f"{prefix}_within_1m"] = hit(pair, manifest[candidate], 1) if candidate else ""
                for field in ("retrieval_rank", "t_err_xy", "n_3d", "n_inlier", "reproj_med",
                              "generated_valid_available", "generated_valid_used", "generated_inliers",
                              "restored_generated", "restored_inliers"):
                    record[f"{prefix}_{field}"] = pair.get(field, "")
        records.append(record)
    print("1 m groups:", dict(Counter((r["change"], r["selection"]) for r in records)))
    for change, field in (("improved", f"new_{before_name}_within_1m"),
                          ("worsened", f"old_{after_name}_within_1m")):
        print(change, field, dict(Counter(r[field] for r in records
              if r["change"] == change and r["selection"] == "changed_anchor")))
    for name, pairs in arms.items():
        count = sum(int(row["generated_inliers"]) > 0 for row in pairs.values())
        print(f"{name}: candidates with generated inliers: {count}/{len(pairs)}")
    if records:
        write_csv(args.output, records)
        print(args.output)


if __name__ == "__main__":
    main()
