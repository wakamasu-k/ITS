"""Read-only inference audit; writes new diagnostic tables, never selects using GT."""
import argparse
import json
from pathlib import Path
import pandas as pd


def boolean(s):
    return s.astype(str).str.lower().isin(["true", "1"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1] / "outputs"
    roots = {
        v: base / f"{v}_retrieval_top10_stage1_fp16_{date}"
        for v, date in [("A", "20261020"), ("B", "20261020"), ("C", "20261019")]
    }
    keys = ["q_seg", "q_frame_index", "db_seg", "anchor_id"]
    tables, reports = {}, {}
    for v, root in roots.items():
        manifest = pd.read_csv(root / "all507.csv")
        raw = pd.read_csv(root / "all507_pnp/per_pair.csv")
        selected = pd.read_csv(root / "all507_pnp/integrated_per_query.csv")
        assert len(raw) == len(manifest) == 5070
        assert not raw.duplicated(keys).any()
        pairs = raw.merge(
            manifest[keys + ["retrieval_q_idx", "retrieval_rank", "retrieval_is_same_pair"]],
            on=keys, validate="one_to_one",
        )
        assert len(pairs) == 5070
        assert len(selected) == 507 and selected.query_index.nunique() == 507
        pairs["valid"] = (
            boolean(pairs.pnp_success) & boolean(pairs.comparable)
            & boolean(pairs.retrieval_is_same_pair)
            & pd.to_numeric(pairs.t_err_xy, errors="coerce").notna()
            & (pd.to_numeric(pairs.t_err_xy, errors="coerce").abs() < float("inf"))
        )
        records = []
        for _, s in selected.iterrows():
            g = pairs[pairs.retrieval_q_idx == s.query_index]
            assert len(g) == 10
            same = g[boolean(g.retrieval_is_same_pair)]
            rec = dict(variant=v, query_index=int(s.query_index), pair_id=s.pair_id,
                       q_seg=s.q_seg, q_frame_index=int(s.q_frame_index),
                       selected_rank=s.selected_rank, selected_xy=s.xy_error_m,
                       wrong_pair=bool(boolean(pd.Series([s.selected_wrong_pair])).iloc[0]),
                       nearest_candidate_m=same.dist_xy_m.min(),
                       oracle_xy_m=g.loc[g.valid, "t_err_xy"].min())
            for t in (1, 2, 5):
                good = g.valid & (g.t_err_xy <= t)
                success = boolean(pd.Series([s[f"success_xy_{t}m"]])).iloc[0]
                near = (same.dist_xy_m <= t).any()
                # Oracle is diagnostic only, never used for actual selection.
                category = ("success" if success else
                            "selection_miss" if good.any() else
                            "no_good_pose_near_present" if near else
                            "no_good_pose_no_near")
                rec[f"category_{t}m"] = category
                rec[f"oracle_success_{t}m"] = bool(good.any())
                rec[f"near_present_{t}m"] = bool(near)
            records.append(rec)
        df = pd.DataFrame(records)
        tables[v] = df
        reports[v] = {
            str(t): {
                "categories": df[f"category_{t}m"].value_counts().to_dict(),
                "oracle_success_count": int(df[f"oracle_success_{t}m"].sum()),
                "near_candidate_count": int(df[f"near_present_{t}m"].sum()),
            } for t in (1, 2, 5)
        }
    paired = tables["A"].merge(tables["B"], on=["q_seg", "q_frame_index"],
                               suffixes=("_A", "_B"), validate="one_to_one")
    assert len(paired) == 507
    reports["A_to_B_1m"] = pd.crosstab(
        paired.category_1m_A, paired.category_1m_B
    ).to_dict()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pd.concat(tables.values()).to_csv(args.output_dir / "per_query.csv", index=False)
    tables["B"].groupby(["pair_id", "category_1m"]).size().unstack(fill_value=0).to_csv(
        args.output_dir / "B_by_segment_1m.csv")
    (args.output_dir / "summary.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
