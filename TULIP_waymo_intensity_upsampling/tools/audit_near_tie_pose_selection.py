"""Compare saved PnP selection with a same-map spatial vote for near ties."""
import argparse
import math
from collections import defaultdict
from pathlib import Path

from evaluate_retrieval_topk_pipeline import (
    flag, key, number, read_csv, selection_key, write_csv,
)


def position(row):
    return (number(row["pred_Cx"]), number(row["pred_Cy"]))


def select_near_tie(rows, baseline, margin, radius):
    peers = [r for r in rows if flag(r["pnp_success"])
             and r["db_seg"] == baseline["db_seg"]
             and all(math.isfinite(v) for v in position(r))]
    if not flag(baseline["pnp_success"]) or baseline not in peers:
        return baseline
    minimum = number(baseline["n_inlier"]) * (1 - margin)
    options = [r for r in peers if number(r["n_inlier"]) >= minimum]

    def vote(row):
        return sum(math.dist(position(row), position(peer)) <= radius for peer in peers)

    return min(options, key=lambda row: (-vote(row), selection_key(row)))


def hit(row, threshold):
    return (flag(row["pnp_success"]) and flag(row["comparable"])
            and flag(row["retrieval_is_same_pair"]) and number(row["t_err_xy"]) <= threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=.05)
    parser.add_argument("--radius-m", type=float, default=2.)
    args = parser.parse_args()
    if not (0 <= args.margin < 1 and 0 < args.radius_m < math.inf):
        parser.error("Invalid margin or radius")
    manifest = {key(r): r for r in read_csv(args.root / "all507.csv")}
    pairs = read_csv(args.root / "all507_pnp/per_pair.csv")
    if not pairs or not {"pred_Cx", "pred_Cy"} <= pairs[0].keys():
        raise ValueError("Expected candidate predictions with pred_Cx and pred_Cy")
    if len(pairs) != len(manifest) or {key(r) for r in pairs} != manifest.keys():
        raise ValueError("Candidate results and manifest differ")
    queries = defaultdict(list)
    for pair in pairs:
        row = manifest[key(pair)]
        pairs_rank = {field: row[field] for field in (
            "retrieval_rank", "retrieval_is_same_pair")}
        queries[int(row["retrieval_q_idx"])].append(dict(pair, **pairs_rank))
    details = []
    for qi, candidates in sorted(queries.items()):
        baseline = min(candidates, key=selection_key)
        voted = select_near_tie(candidates, baseline, args.margin, args.radius_m)
        row = dict(query_index=qi, pair_id=baseline["pair_id"],
                   changed=key(baseline) != key(voted))
        for name, pair in (("baseline", baseline), ("vote", voted)):
            for field in ("db_seg", "anchor_id", "retrieval_rank", "n_inlier", "reproj_med", "t_err_xy"):
                row[f"{name}_{field}"] = pair[field]
            for threshold in (.5, 1, 2, 5):
                row[f"{name}_hit_{threshold:g}m"] = hit(pair, threshold)
        details.append(row)
    write_csv(args.output, details)
    print(f"Queries={len(details)}; changed selections={sum(r['changed'] for r in details)}")
    for threshold in (.5, 1, 2, 5):
        before, after = (f"{name}_hit_{threshold:g}m" for name in ("baseline", "vote"))
        print(f"{threshold:g} m: {sum(r[before] for r in details)} -> "
              f"{sum(r[after] for r in details)}; "
              f"improved={sum(not r[before] and r[after] for r in details)}; "
              f"worsened={sum(r[before] and not r[after] for r in details)}")
    print(args.output)


if __name__ == "__main__":
    main()
