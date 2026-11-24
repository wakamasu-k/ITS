"""Compare both selected anchors under both dense inputs using saved CSVs."""
import argparse
import csv
import math
from collections import Counter
from pathlib import Path


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def query_key(row):
    return row["pair_id"], row["q_seg"], int(row["q_frame_index"])


def candidate_key(row):
    return (*query_key(row), row["db_seg"], int(row["anchor_id"]))


def load_candidates(root):
    manifest = read_csv(root.parent / "all507.csv")
    pairs = read_csv(root / "per_pair.csv")
    listed = {candidate_key(row): row for row in manifest}
    results = {candidate_key(row): row for row in pairs}
    if len(listed) != len(manifest) or len(results) != len(pairs) or listed.keys() != results.keys():
        raise ValueError(f"Candidate keys do not match the manifest: {root}")
    return {key: {**results[key], "retrieval_rank": row["retrieval_rank"],
                  "retrieval_is_same_pair": row["retrieval_is_same_pair"]}
            for key, row in listed.items()}


def flag(value):
    return value.strip().lower() in {"true", "1"}


def pose_status(row):
    if not flag(row["pnp_success"]):
        return "pnp_failed"
    if not (flag(row["comparable"]) and flag(row["retrieval_is_same_pair"])):
        return "not_comparable"
    try:
        error = float(row["t_err_xy"])
    except ValueError:
        return "not_comparable"
    if not math.isfinite(error):
        return "not_comparable"
    return "within_1m" if error <= 1 else "over_1m"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed", type=Path, required=True)
    parser.add_argument("--a", type=Path, required=True, help="A input all507_pnp directory")
    parser.add_argument("--priority", type=Path, required=True, help="Priority input all507_pnp directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, priority = map(load_candidates, (args.a, args.priority))
    if a.keys() != priority.keys() or any(
        any(a[key][field] != priority[key][field]
            for field in ("retrieval_rank", "retrieval_is_same_pair")) for key in a
    ):
        raise ValueError("Top-K identity, rank or scene membership differs")

    rows = []
    counts = Counter()
    fields = ("retrieval_rank", "t_err_xy", "n_match_in", "n_3d", "n_inlier",
              "inlier_ratio", "reproj_med")
    for changed in read_csv(args.changed):
        if changed["selection"] != "changed_anchor":
            continue
        row = {field: changed[field] for field in (
            "query_index", "pair_id", "q_seg", "q_frame_index", "change")}
        for role, source in (("old", "A"), ("new", "priority")):
            seg = changed[f"{source}_db_seg"]
            anchor = int(changed[f"{source}_anchor_id"])
            key = (*query_key(changed), seg, anchor)
            row.update({f"{role}_db_seg": seg, f"{role}_anchor_id": anchor})
            for name, candidates in (("A", a), ("priority", priority)):
                result = candidates[key]
                prefix = f"{role}_{name}"
                status = pose_status(result)
                row[f"{prefix}_status"] = status
                for field in fields:
                    row[f"{prefix}_{field}"] = (
                        "" if field == "t_err_xy" and status in {"not_comparable", "pnp_failed"}
                        else result[field]
                    )
        if changed["change"] == "improved":
            counts["improved_new_anchor_under_A", row["new_A_status"]] += 1
        else:
            counts["worsened_old_anchor_under_priority", row["old_priority_status"]] += 1
        rows.append(row)

    if not rows:
        raise ValueError("No changed-anchor cases")
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Changed-anchor cases: {len(rows)}")
    for (group, status), count in sorted(counts.items()):
        print(f"  {group} / {status}: {count}")
    print(args.output)


if __name__ == "__main__":
    main()
