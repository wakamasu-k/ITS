"""Prepare or run three fixed queries for each of nine saved representative anchors."""
import argparse
import csv
import json
import subprocess
from pathlib import Path
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
TOOLS=PROJECT/"tools"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source",type=Path,default=PROJECT/"outputs/AC_loftr_locations374_jOm2o4")
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    plan_path=args.output/"plan.json"
    if not plan_path.exists():
        original=json.loads((args.source/"protocol.json").read_text())
        base=Path("/mnt/w/its_outputs/priority_validation374_abc_gXMLfD")
        queries=pd.read_csv(base/"C/retrieval/C_per_query.csv")
        a_db=pd.read_parquet(base/"A/db.parquet")
        c_db=pd.read_parquet(base/"C/db.parquet")
        with (base/"C/top10.csv").open() as stream:
            templates={int(r["retrieval_q_idx"]):r for r in csv.DictReader(stream)}
        batches=[[] for _ in range(3)]
        for old in original["jobs"]:
            eligible=queries[queries.pair_id==old["pair_id"]].copy()
            target=int(old["anchor_id"])
            eligible["anchor_gap"]=[abs(int(c_db.iloc[int(i)].anchor_id)-target) for i in eligible.gt_index_db]
            # Known positions choose diagnostic examples only, not candidates at inference.
            eligible=eligible.sort_values(["anchor_gap","gt_dist_m","q_idx"])
            if len(eligible)<3:
                raise ValueError("Need three queries per scene")
            for rank,(_,query) in enumerate(eligible.head(3).iterrows()):
                job=dict(templates[int(query.q_idx)])
                idx=old["db_index"]
                image=Path(c_db.iloc[idx].anchor_png)
                job.update(db_index=idx,anchor_id=str(target),A_png=a_db.iloc[idx].anchor_png,
                           anchor_png=str(image),anchor_json=str(image.with_name("anchor.json")),
                           anchor_index_npz=str(image.with_name("anchor_index.npz")),
                           query_selection="nearest known GT anchor ID, then distance",
                           dist_xy_m="",retrieval_rank="diagnostic_fixed_anchor")
                batches[rank].append(job)
        plan=dict(source=str(args.source.resolve()),groups=[])
        for rank,jobs in enumerate(batches):
            group=args.output/f"query_group_{rank}"
            ac=group/"AC"
            ac.mkdir(parents=True,exist_ok=True)
            protocol=dict(original,jobs=jobs)
            (ac/"protocol.json").write_text(json.dumps(protocol,indent=2))
            recovery=group/"B"
            recovery.mkdir(parents=True,exist_ok=True)
            b_db=pd.read_parquet(base/"B/db.parquet")
            b_jobs=[dict(job,B_png=b_db.iloc[job["db_index"]].anchor_png) for job in jobs]
            (recovery/"jobs.json").write_text(json.dumps(b_jobs,indent=2))
            plan["groups"].append(dict(AC=str(ac),B=str(recovery)))
        plan_path.write_text(json.dumps(plan,indent=2))
    plan=json.loads(plan_path.read_text())
    if plan["source"]!=str(args.source.resolve()):
        raise ValueError("Output belongs to another source")
    print("Prepared 9 anchors x 3 queries; A/B/C use identical pairs",flush=True)
    if args.prepare_only:
        return
    python="/home/wakamatsu/ITS/.venv_tulip/bin/python"
    original=json.loads((args.source/"protocol.json").read_text())
    records=[]
    for rank,group in enumerate(plan["groups"]):
        subprocess.run([python,"-u",str(TOOLS/"compare_A_C_loftr_locations374.py"),
                        "--source",original["source"],"--output",group["AC"]],check=True)
        subprocess.run([python,"-u",str(TOOLS/"audit_B_loftr_recovery374.py"),
                        "--source",group["AC"],"--output",group["B"]],check=True)
        summary=pd.read_csv(Path(group["B"])/"summary.csv")
        summary["query_group"]=rank
        records.append(summary)
    summary=pd.concat(records,ignore_index=True)
    summary.to_csv(args.output/"summary27.csv",index=False)
    point_tables=[]
    for rank,group in enumerate(plan["groups"]):
        for path in Path(group["B"]).glob("db_*_points.csv"):
            table=pd.read_csv(path)
            for flag in ("recovered", "lost", "still_missing"):
                table[flag] = table[flag].astype(int)
            table["db_index"]=int(path.name.split("_")[1])
            table["query_group"]=rank
            # Spatial bins describe image location only, with no semantic labels.
            table["grid_x"]=(table.x//160).astype(int)
            table["grid_y"]=(table.y//160).astype(int)
            point_tables.append(table)
    points=pd.concat(point_tables,ignore_index=True)
    points.to_csv(args.output/"all_reference_points.csv",index=False)
    regions=points.groupby(["db_index","grid_x","grid_y"]).agg(
        queries_observed=("query_group","nunique"),references=("x","size"),
        B_recovered=("recovered","sum"),B_lost=("lost","sum"),still_missing=("still_missing","sum"))
    regions.to_csv(args.output/"spatial_counts.csv")
    print(summary.to_string(index=False))
    print(f"Done: {args.output}",flush=True)


if __name__=="__main__":
    main()
