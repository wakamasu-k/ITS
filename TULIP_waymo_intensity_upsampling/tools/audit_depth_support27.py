"""Report support and depth spread around every missing-depth correspondence."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for group_id,group in enumerate(json.loads((args.source/"plan.json").read_text())["groups"]):
        jobs=json.loads((Path(group["B"])/"jobs.json").read_text())
        for job in jobs:
            stem=f"db_{job['db_index']:04d}"
            with np.load(Path(group["AC"])/(stem+".npz")) as data:
                ac={k:data[k].copy() for k in ("A_anchor_xy","C_anchor_xy")}
            with np.load(Path(group["B"])/(stem+"_B_matches.npz")) as data:
                bm=data["anchor_xy"].copy()
            for name,image,points in (("A",job["A_png"],ac["A_anchor_xy"]),
                                      ("B",job["B_png"],bm),("C",job["anchor_png"],ac["C_anchor_xy"])):
                with np.load(Path(image).with_name("anchor_index.npz")) as data:
                    depth=data["depth"].copy()
                    valid=(data["point_index"]>=0)&np.isfinite(depth)&(depth>=.5)&(depth<=200)
                h,w=depth.shape
                u=np.clip(np.round(points[:,0]).astype(int),0,w-1)
                v=np.clip(np.round(points[:,1]).astype(int),0,h-1)
                for j in np.flatnonzero(~valid[v,u]):
                    y,x=v[j],u[j]
                    y0,y1=max(0,y-2),min(h,y+3)
                    x0,x1=max(0,x-2),min(w,x+3)
                    mask=valid[y0:y1,x0:x1]
                    values=depth[y0:y1,x0:x1][mask]
                    support=len(values)
                    spread=float(np.ptp(values)) if support else None
                    accepted=support>=3 and spread<=1
                    yy,xx=np.nonzero(mask)
                    distance=float(np.sqrt(((yy+y0-y)**2+(xx+x0-x)**2).min())) if support else None
                    records.append(dict(query_group=group_id,db_index=job["db_index"],
                                        query=job["retrieval_q_idx"],line=name,match_index=int(j),
                                        x=int(x),y=int(y),valid_support5x5=support,
                                        support_fraction=support/mask.size,depth_spread_m=spread,
                                        nearest_valid_distance_px=distance,accepted=accepted,
                                        reason="accepted" if accepted else "low_support" if support<3 else "wide_depth_range"))
    table=pd.DataFrame(records)
    table.to_csv(args.output/"per_correspondence.csv",index=False)
    counts=table.groupby(["line","reason"]).size().rename("correspondences").reset_index()
    counts.to_csv(args.output/"summary.csv",index=False)
    print(counts.to_string(index=False))
    print(args.output)


if __name__=="__main__":
    main()
