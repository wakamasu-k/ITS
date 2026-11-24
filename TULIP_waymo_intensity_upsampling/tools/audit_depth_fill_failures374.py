"""Inspect depth-fill geometry for worsened and improved integrated queries."""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from replay_ACB_pnp_hypotheses374 import fill_sampled_depth, gt_errors
from audit_loftr_generated_contribution import load_evaluator
from audit_priority_pnp_sources import solve
from evaluate_retrieval_topk_pipeline import ICP, read_csv
from run_tulip_pixel_ablation374 import BASE, VALID


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    settings=json.loads((args.run/"protocol.json").read_text())
    source=Path(settings["source"])
    manifest=read_csv(source/"top10.csv")
    changes=pd.read_csv(args.run/"changed_queries_1m.csv")
    c_db=pd.read_parquet(BASE/"C/db.parquet")
    c_lookup={(r.db_seg,int(r.anchor_id)):r.anchor_png for r in c_db.itertuples()}
    wanted={}
    for change in changes.itertuples():
        for side in ("old","new"):
            lookup=(int(change.query_index),getattr(change,"selected_db_seg_"+side),int(getattr(change,"selected_anchor_id_"+side)))
            candidates=[(i,row) for i,row in enumerate(manifest) if
                        (int(row["retrieval_q_idx"]),row["db_seg"],int(row["anchor_id"]))==lookup]
            if len(candidates)!=1:
                raise ValueError("Selected candidate missing or duplicated")
            i,row=candidates[0]
            wanted.setdefault(i,dict(row=row,query=int(change.query_index),change=change.change,
                                     same_anchor=bool(change.same_anchor),sides=[]))["sides"].append(side)
    args.output.mkdir(parents=True,exist_ok=True)
    ev=load_evaluator()
    ev.cv2.setNumThreads(1)
    icp=ev.load_icp_db_to_q(ICP,db_col=None,q_col=None,mat_col=None,mat_cols=None)
    @lru_cache(maxsize=2)
    def points(subset,segment):
        with np.load(VALID/"maps/A_observed16"/subset/segment/"map_static.npz") as z:
            return z["xyz"]
    records=[]
    for number,(candidate,item) in enumerate(sorted(wanted.items()),1):
        row=item["row"]
        with np.load(source/"dense/matches"/f"{candidate:05d}.npz") as z:
            mk0,mk1=z["mkpts0"],z["mkpts1"]
        folder=Path(row["anchor_png"]).parent
        anchor=json.loads((folder/"anchor.json").read_text())
        query=json.loads(Path(row["q_meta_json"]).read_text())
        original=ev.load_anchor_index_npz(str(folder/"anchor_index.npz"))
        with np.load(folder/"anchor_index.npz") as z:
            point_index=z["point_index"].copy()
        filled,_=fill_sampled_depth(ev,original,mk0)
        h,w=original.depth_x.shape
        u=np.clip(np.round(mk0[:,0]).astype(int),0,w-1)
        v=np.clip(np.round(mk0[:,1]).astype(int),0,h-1)
        base_valid=original.valid_mask&np.isfinite(original.depth_x)&(original.depth_x>=.5)&(original.depth_x<=200)
        new_valid=filled.valid_mask[v,u]&np.isfinite(filled.depth_x[v,u])&(filled.depth_x[v,u]>=.5)&(filled.depth_x[v,u]<=200)
        added=new_valid&~base_valid[v,u]
        transform=icp.get(row["db_seg"],row["q_seg"])
        pose,inliers,error=solve(ev,mk0,mk1,np.ones(len(mk0),bool),new_valid,filled,anchor,query,transform)
        inlier=np.zeros(len(mk0),bool)
        inlier[inliers]=True
        check=gt_errors(ev,mk0,mk1,added,filled.depth_x[v,u],anchor,query,transform) if transform is not None else np.full(len(mk0),np.nan)
        c_image=c_lookup.get((row["db_seg"],int(row["anchor_id"])))
        cd,cv=None,None
        if c_image is not None:
            cm=json.loads(Path(c_image).with_name("anchor.json").read_text())
            for field in ("intrinsic","world_to_cam","width","height"):
                if not np.allclose(anchor[field],cm[field],atol=1e-8,rtol=0):
                    raise ValueError("C geometry differs")
            with np.load(Path(c_image).with_name("anchor_index.npz")) as z:
                cd=z["depth"].copy()
                cv=(z["point_index"]>=0)&np.isfinite(cd)&(cd>0)
        xyz=points("training",row["db_seg"])
        image=cv2.imread(row["anchor_png"])
        for j in np.flatnonzero(added):
            y,x=v[j],u[j]
            y0,y1=max(0,y-2),min(h,y+3)
            x0,x1=max(0,x-2),min(w,x+3)
            local=base_valid[y0:y1,x0:x1]
            ids=np.unique(point_index[y0:y1,x0:x1][local])
            if ids.min()<0 or ids.max()>=len(xyz):
                raise ValueError("Invalid map point IDs")
            support=xyz[ids].astype(np.float64)
            center=support.mean(axis=0)
            residual,rms=None,None
            noncollinear=False
            if len(ids)>=3:
                _,singular,vectors=np.linalg.svd(support-center,full_matrices=False)
                noncollinear=bool(singular[1]>1e-6)
                if noncollinear:
                    normal=vectors[-1]
                    rms=float(np.sqrt(np.mean(((support-center)@normal)**2)))
                    fx,fy,cx,cy=ev._parse_intrinsic(anchor)
                    proposed=ev.backproject_xforward(np.array([mk0[j,0]]),np.array([mk0[j,1]]),
                              np.array([filled.depth_x[y,x]]),fx,fy,cx,cy)
                    proposed=ev._transform_points(ev._invert_T(ev._parse_Twc(anchor)),proposed)[0]
                    residual=float(abs((proposed-center)@normal))
            c_delta=float(filled.depth_x[y,x]-cd[y,x]) if cd is not None and cv[y,x] else None
            local_depth=original.depth_x[y0:y1,x0:x1][local]
            records.append(dict(candidate_index=candidate,query=item["query"],change=item["change"],
                                same_anchor=item["same_anchor"],selected_sides=",".join(item["sides"]),
                                db_seg=row["db_seg"],anchor_id=row["anchor_id"],match_index=int(j),
                                x=int(x),y=int(y),support_pixels=int(local.sum()),
                                support_unique_points=len(ids),support_depth_spread_m=float(np.ptp(local_depth)),
                                filled_depth_m=float(filled.depth_x[y,x]),C_depth_delta_m=c_delta,
                                noncollinear=noncollinear,plane_rms_m=rms,point_to_plane_m=residual,
                                filled_pnp_inlier=bool(inlier[j]),
                                gt_reprojection_px=float(check[j]) if np.isfinite(check[j]) else None))
            cv2.circle(image,(int(x),int(y)),4,(0,220,0) if inlier[j] else (0,0,255),-1)
        if image is None or not cv2.imwrite(str(args.output/f"candidate_{candidate:05d}.png"),image):
            raise OSError("Could not save overlay")
        print(f"Audited {number}/{len(wanted)} selected candidates",flush=True)
    table=pd.DataFrame(records)
    table.to_csv(args.output/"added_correspondences.csv",index=False)
    summary=table.groupby("change").agg(added=("x","size"),
        median_support_pixels=("support_pixels","median"),
        median_unique_points=("support_unique_points","median"),
        pnp_inliers=("filled_pnp_inlier","sum"),
        median_plane_residual_m=("point_to_plane_m","median"))
    summary.to_csv(args.output/"summary.csv")
    print(summary.to_string())
    print(args.output)


if __name__=="__main__":
    main()
