"""Apply the existing depth-fill replay to all 27 cached fixed-pair comparisons."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    groups = json.loads((args.source/"plan.json").read_text())["groups"]
    args.output.mkdir(parents=True,exist_ok=True)
    config = args.output/"protocol.json"
    settings = dict(source=str(args.source.resolve()),groups=groups,seeds=1)
    if config.exists() and json.loads(config.read_text()) != settings:
        raise ValueError("Output belongs to another comparison")
    config.write_text(json.dumps(settings,indent=2))
    tables = []
    for number,group in enumerate(groups):
        out = args.output/f"query_group_{number}"
        complete = out/"selected.csv"
        if not complete.exists():
            command = ["/home/wakamatsu/ITS/.venv_tulip/bin/python","-u",
                       str(PROJECT/"tools/replay_ACB_pnp_hypotheses374.py"),
                       "--run",group["B"],"--output",str(out),"--seeds","1"]
            subprocess.run(command,check=True,env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4"))
        table = pd.read_csv(complete)
        table = table[table.method=="seed0"].copy()
        table["query_group"] = number
        tables.append(table)
    full = pd.concat(tables,ignore_index=True)
    full["xy_error"] = pd.to_numeric(full.xy_error,errors="coerce")
    keys = ["query_group","db_index","query","line","depth_mode"]
    if len(full)!=162 or full.duplicated(keys).any():
        raise ValueError("Expected 27 pairs x 3 line types x 2 depth modes")
    full.to_csv(args.output/"all27.csv",index=False)
    summaries,changes = [],[]
    for line,table in full.groupby("line"):
        for mode,g in table.groupby("depth_mode"):
            finite = np.isfinite(g.xy_error)
            summaries.append(dict(line=line,mode=mode,pairs=len(g),pnp_success=int(g.success.sum()),
                                  comparable_finite=int(finite.sum()),
                                  xy_median_success_only=float(g.loc[finite,"xy_error"].median()),
                                  within_05m=int((g.xy_error<=.5).sum()),
                                  within_1m=int((g.xy_error<=1).sum()),
                                  within_2m=int((g.xy_error<=2).sum()),
                                  within_5m=int((g.xy_error<=5).sum()),
                                  newly_valid_correspondences=int(g.newly_valid.sum())))
        paired=table.pivot(index=["query_group","db_index","query"],columns="depth_mode",values="xy_error")
        if len(paired)!=27:
            raise ValueError("Expected identical sets of 27 pairs")
        for threshold in (.5,1.,2.,5.):
            before=paired.original<=threshold
            after=paired.depth_fill<=threshold
            changes.append(dict(line=line,threshold_m=threshold,before=int(before.sum()),
                                after=int(after.sum()),improved=int((~before&after).sum()),
                                worsened=int((before&~after).sum())))
    pd.DataFrame(summaries).to_csv(args.output/"comparison.csv",index=False)
    pd.DataFrame(changes).to_csv(args.output/"transitions.csv",index=False)
    print(pd.DataFrame(summaries).round(4).to_string(index=False))
    print(pd.DataFrame(changes).to_string(index=False))
    print(f"Done: {args.output}",flush=True)


if __name__=="__main__":
    main()
