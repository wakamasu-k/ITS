"""Use fixed cached matches to compare depth-hole filling and RANSAC restarts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def gt_errors(ev, mk0, mk1, valid, depth, anchor, query, transform):
    errors = np.full(len(mk0), np.inf)
    if not valid.any():
        return errors
    fx, fy, cx, cy = ev._parse_intrinsic(anchor)
    xyz = ev.backproject_xforward(mk0[valid,0], mk0[valid,1], depth[valid], fx, fy, cx, cy)
    world = ev._transform_points(ev._invert_T(ev._parse_Twc(anchor)), xyz)
    world = ev._transform_points(transform, world)
    camera = ev._transform_points(ev._parse_Twc(query), world)
    front = camera[:,0] > .1
    fx, fy, cx, cy = ev._parse_intrinsic(query)
    projected = np.full((len(camera),2), np.inf)
    projected[front,0] = cx-fx*camera[front,1]/camera[front,0]
    projected[front,1] = cy-fy*camera[front,2]/camera[front,0]
    errors[valid] = np.linalg.norm(projected-mk1[valid],axis=1)
    return errors



def fill_sampled_depth(ev, original, mk0):
    """Copy nearby original depth only at matched pixels that lack usable depth."""
    h,w=original.depth_x.shape
    u=np.clip(np.round(mk0[:,0]).astype(int),0,w-1)
    v=np.clip(np.round(mk0[:,1]).astype(int),0,h-1)
    base_valid=original.valid_mask&np.isfinite(original.depth_x)&(original.depth_x>=.5)&(original.depth_x<=200)
    depth=original.depth_x.copy()
    mask=original.valid_mask.copy()
    counts=dict(missing_pixels=0,low_support=0,wide_depth_range=0,filled_pixels=0)
    for y,x in sorted(set(zip(v[~base_valid[v,u]],u[~base_valid[v,u]]))):
        counts["missing_pixels"]+=1
        y0,y1=max(0,y-2),min(h,y+3)
        x0,x1=max(0,x-2),min(w,x+3)
        valid=base_valid[y0:y1,x0:x1]
        values=original.depth_x[y0:y1,x0:x1][valid]
        if len(values)<3:
            counts["low_support"]+=1
            continue
        if np.ptp(values)>1:
            counts["wide_depth_range"]+=1
            continue
        yy,xx=np.nonzero(valid)
        nearest=np.argmin((yy+y0-y)**2+(xx+x0-x)**2)
        depth[y,x]=values[nearest]
        mask[y,x]=True
        counts["filled_pixels"]+=1
    return ev.AnchorIndex(depth,mask),counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds",type=int,default=5)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    from audit_loftr_generated_contribution import load_evaluator
    from audit_priority_pnp_sources import solve
    from evaluate_retrieval_topk_pipeline import ICP
    ev = load_evaluator()
    icp = ev.load_icp_db_to_q(ICP,db_col=None,q_col=None,mat_col=None,mat_cols=None)
    jobs = json.loads((args.run/"jobs.json").read_text())
    source = Path(json.loads((args.run/"protocol.json").read_text())["source"])
    args.output.mkdir(parents=True,exist_ok=True)
    settings = dict(run=str(args.run.resolve()),seeds=args.seeds,radius_px=2,max_spread_m=1.,min_support=3)
    config = args.output/"protocol.json"
    if config.exists() and json.loads(config.read_text()) != settings:
        raise ValueError("Output uses other settings")
    config.write_text(json.dumps(settings,indent=2))
    all_rows, selections = [], []
    for job in jobs:
        stem=f"db_{job['db_index']:04d}"
        with np.load(source/(stem+".npz")) as z:
            ac={k:z[k].copy() for k in z.files}
        with np.load(args.run/(stem+"_B_matches.npz")) as z:
            bc={k:z[k].copy() for k in z.files}
        query=json.loads(Path(job["q_meta_json"]).read_text())
        transform=icp.get(job["db_seg"],job["q_seg"])
        if transform is None:
            raise ValueError("Missing ICP transform")
        for name, image in (("A",job["A_png"]),("B",job["B_png"]),("C",job["anchor_png"])):
            cache=bc if name=="B" else ac
            mk0=cache["anchor_xy" if name=="B" else name+"_anchor_xy"]
            mk1=cache["query_xy" if name=="B" else name+"_query_xy"]
            folder=Path(image).parent
            anchor=json.loads((folder/"anchor.json").read_text())
            original=ev.load_anchor_index_npz(str(folder/"anchor_index.npz"))
            h,w=original.depth_x.shape
            u=np.clip(np.round(mk0[:,0]).astype(int),0,w-1)
            v=np.clip(np.round(mk0[:,1]).astype(int),0,h-1)
            base_valid=original.valid_mask&np.isfinite(original.depth_x)&(original.depth_x>=.5)&(original.depth_x<=200)
            filled, _ = fill_sampled_depth(ev, original, mk0)
            for mode,index in (("original",original),("depth_fill",filled)):
                sampled=index.depth_x[v,u]
                valid=index.valid_mask[v,u]&np.isfinite(sampled)&(sampled>=.5)&(sampled<=200)
                good=gt_errors(ev,mk0,mk1,valid,sampled,anchor,query,transform)<=4
                rows=[]
                for seed in range(args.seeds):
                    pose,inliers,error=solve(ev,mk0,mk1,np.ones(len(mk0),bool),valid,index,anchor,query,transform,seed=seed)
                    selected=np.zeros(len(mk0),bool)
                    selected[inliers]=True
                    row=dict(db_index=job["db_index"],query=job["retrieval_q_idx"],line=name,depth_mode=mode,seed=seed,
                             success=bool(pose.success),n_3d=int(pose.n_3d),inliers=int(pose.n_inlier),
                             reproj_median=float(pose.reproj_med),xy_error=error,
                             newly_valid=int((valid&~base_valid[v,u]).sum()),
                             gt_good_available=int(good.sum()),gt_good_inliers=int((good&selected).sum()),
                             gt_good_rejected=int((good&~selected).sum()),
                             gt_bad_inliers=int((~good&selected).sum()))
                    rows.append(row)
                    all_rows.append(row)
                winner=min(rows,key=lambda r:(not r["success"],-r["inliers"],r["reproj_median"],r["seed"]))
                for method,row in (("seed0",rows[0]),("max_inlier_restart",winner)):
                    selections.append(dict(row,method=method))
        print(f"Replayed {stem}",flush=True)
    pd.DataFrame(all_rows).to_csv(args.output/"trials.csv",index=False)
    pd.DataFrame(selections).to_csv(args.output/"selected.csv",index=False)
    print(pd.DataFrame(selections).query("db_index == 345")[["line","depth_mode","method","seed","xy_error","newly_valid","gt_good_available","gt_good_inliers","gt_bad_inliers"]].to_string(index=False))
    print(args.output)


if __name__=="__main__":
    main()
