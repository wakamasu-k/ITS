"""Compare fixed anchors across the saved region-pilot Retrieval rankings."""
import argparse
import json
from pathlib import Path

from evaluate_retrieval_topk_pipeline import read_csv, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", nargs="+", type=int, default=[49, 268, 146])
    args = parser.parse_args()
    names = ("closing", "preserve_A_png", "region")
    cases = {int(r["q_idx"]): r for r in read_csv(args.run / "selected_cases.csv")}
    rankings = {}
    for name in names:
        with (args.run / name / f"{name}_topk_top50.jsonl").open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        rankings[name] = {r["q_cam_png"]: r for r in rows}
        if len(rankings[name]) != len(rows):
            raise ValueError(f"Duplicate camera path: {name}")
    output = []
    for qi in args.queries:
        case = cases[qi]
        records = {name: rankings[name][case["camera"]] for name in names}
        ids = {}
        for name, q in records.items():
            if q["pair_id"] != case["pair_id"] or int(q["strict_gt"]["db_index"]) != int(case["gt_index"]):
                raise ValueError(f"Query identity mismatch: {qi}/{name}")
            near = next((r for r in q["topk"] if r["soft_gt"]["t2m"]), None)
            entries = [("strict_gt", q["strict_gt"]), ("rank10", q["topk"][9]),
                       ("top1", q["topk"][0])]
            if near is not None:
                entries.append(("best_2m", near))
            print(f"q{qi} {name}: best_2m rank={near['rank'] if near else '>50'}, "
                  f"rank10 score={q['topk'][9]['score']:.5f}")
            for role, row in entries:
                ids.setdefault(row["db_index"], []).append(f"{name}:{role}")
        for idx, roles in ids.items():
            for name, q in records.items():
                row = next((r for r in q["topk"] if r["db_index"] == idx), None)
                if row is None and q["strict_gt"]["db_index"] == idx:
                    row = q["strict_gt"]
                cutoff = q["topk"][9]["score"]
                output.append(dict(
                    query=qi, db_index=idx, roles=";".join(roles), condition=name,
                    rank=row.get("rank", row.get("strict_rank")) if row else ">50",
                    score=row["score"] if row else "", rank10_score=cutoff,
                    margin_vs_rank10=row["score"]-cutoff if row else "",
                    dist_m=row.get("dist_m") if row else "",
                    anchor_png=row["anchor_png"] if row else "", overlay=case["overlay"]))
    write_csv(args.output, output)
    print(args.output)


if __name__ == "__main__":
    main()
