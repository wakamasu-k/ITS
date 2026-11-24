"""Replay cached A Top-10 matches with local depth filling and reselect candidates."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("/mnt/w/its_outputs/priority_validation374_abc_gXMLfD/A"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--limit",type=int,default=0,help="0 processes all 3740 pairs")
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--shard-index",type=int,default=0,help=argparse.SUPPRESS)
    parser.add_argument("--shard-count",type=int,default=1,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.workers<1 or args.shard_count<1 or not 0<=args.shard_index<args.shard_count:
        parser.error("Invalid worker settings")
    if args.limit and args.workers>1:
        parser.error("--limit cannot be combined with --workers")
    if args.limit<0:
        parser.error("--limit must be nonnegative")
    from audit_loftr_generated_contribution import load_evaluator
    from audit_priority_pnp_sources import solve
    from replay_ACB_pnp_hypotheses374 import fill_sampled_depth,gt_errors
    from evaluate_retrieval_topk_pipeline import ICP,key,read_csv,summarize,write_csv
    manifest=args.source/"top10.csv"
    rows=read_csv(manifest)
    if len(rows)!=3740 or len({key(r) for r in rows})!=3740 or len({r["retrieval_q_idx"] for r in rows})!=374:
        raise ValueError("Expected 374 queries x 10 candidates")
    args.output.mkdir(parents=True,exist_ok=True)
    settings=dict(source=str(args.source.resolve()),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  cached_protocol=json.loads((args.source/"dense/protocol.json").read_text()),
                  radius_px=2,max_spread_m=1.,min_support=3,seed=0)
    protocol=args.output/"protocol.json"
    if protocol.exists() and json.loads(protocol.read_text())!=settings:
        raise ValueError("Output belongs to another protocol")
    protocol.write_text(json.dumps(settings,indent=2))
    if args.workers>1:
        processes=[]
        for shard in range(args.workers):
            command=[sys.executable,"-u",str(Path(__file__).resolve()),"--source",str(args.source),
                     "--output",str(args.output),"--shard-index",str(shard),"--shard-count",str(args.workers)]
            processes.append(subprocess.Popen(command,env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
                                                              OPENBLAS_NUM_THREADS="1")))
        codes=[process.wait() for process in processes]
        if any(code!=0 for code in codes):
            raise RuntimeError(f"PnP workers failed: {codes}; saved pairs can be resumed")
    cache=args.output/"pairs"
    cache.mkdir(exist_ok=True)
    ev=load_evaluator()
    ev.cv2.setNumThreads(1)
    icp=ev.load_icp_db_to_q(ICP,db_col=None,q_col=None,mat_col=None,mat_cols=None)
    @lru_cache(maxsize=4)
    def index_for(path):
        return ev.load_anchor_index_npz(path)
    @lru_cache(maxsize=512)
    def meta(path):
        return json.loads(Path(path).read_text())
    results={"original":[],"depth_fill":[]}
    diagnostics=[]
    jobs=rows[:args.limit] if args.limit else rows
    order=sorted(range(len(jobs)),key=lambda i:(jobs[i]["db_seg"],int(jobs[i]["anchor_id"]),int(jobs[i]["retrieval_q_idx"])))
    order=[i for i in order if i%args.shard_count==args.shard_index]
    for number,i in enumerate(order,1):
        row=jobs[i]
        target=cache/f"{i:05d}.json"
        if target.exists():
            saved=json.loads(target.read_text())
            if tuple(saved["identity"])!=key(row):
                raise ValueError("Saved pair identity differs")
        else:
            path=args.source/"dense/matches"/f"{i:05d}.npz"
            with np.load(path) as data:
                mk0,mk1=data["mkpts0"],data["mkpts1"]
                original=json.loads(data["records"].item())["all"]
                old_valid=data["valid_3d"]
            if key(original)!=key(row) or int(original["retrieval_rank"])!=int(row["retrieval_rank"]):
                raise ValueError(f"Match cache identity differs: {path}")
            index=index_for(row["anchor_index_npz"])
            h,w=index.depth_x.shape
            u=np.clip(np.round(mk0[:,0]).astype(int),0,w-1)
            v=np.clip(np.round(mk0[:,1]).astype(int),0,h-1)
            sampled=index.depth_x[v,u]
            valid=index.valid_mask[v,u]&np.isfinite(sampled)&(sampled>=.5)&(sampled<=200)
            if not np.array_equal(valid,old_valid):
                raise ValueError("Original depth raster differs from cache")
            filled,counts=fill_sampled_depth(ev,index,mk0)
            values=filled.depth_x[v,u]
            valid2=filled.valid_mask[v,u]&np.isfinite(values)&(values>=.5)&(values<=200)
            added=valid2&~valid
            counts["newly_valid_correspondences"]=int(added.sum())
            counts["matches"]=len(mk0)
            if not added.any():
                changed=dict(original)
                counts["newly_valid_gt_within4px"]=0
            else:
                anchor,query=meta(row["anchor_json"]),meta(row["q_meta_json"])
                transform=icp.get(row["db_seg"],row["q_seg"])
                pose,inliers,error=solve(ev,mk0,mk1,np.ones(len(mk0),bool),valid2,filled,
                                         anchor,query,transform,seed=0)
                changed=dict(original,pnp_success=bool(pose.success),comparable=transform is not None,
                             t_err_xy=error,n_3d=int(pose.n_3d),n_inlier=int(pose.n_inlier),
                             reproj_med=float(pose.reproj_med),
                             pred_Cx=float(pose.C_pred[0]) if pose.success else "",
                             pred_Cy=float(pose.C_pred[1]) if pose.success else "")
                if transform is None:
                    counts["newly_valid_gt_within4px"]=""
                else:
                    check=gt_errors(ev,mk0,mk1,added,values,anchor,query,transform)
                    counts["newly_valid_gt_within4px"]=int((check[added]<=4).sum())
                counts["newly_valid_inliers"]=int(added[inliers].sum())
            counts.setdefault("newly_valid_inliers",0)
            saved=dict(identity=key(row),original=original,depth_fill=changed,diagnostics=dict(
                candidate_index=i,retrieval_q_idx=row["retrieval_q_idx"],pair_id=row["pair_id"],**counts))
            temporary=target.with_suffix(".tmp")
            temporary.write_text(json.dumps(saved))
            temporary.replace(target)
        for name in results:
            results[name].append(saved[name])
        diagnostics.append(saved["diagnostics"])
        if number==1 or number%100==0 or number==len(order):
            print(f"PnP replay worker {args.shard_index}: {number}/{len(order)}",flush=True)
    if args.shard_count>1:
        print(f"Worker {args.shard_index} finished",flush=True)
        return
    if len(jobs)!=len(rows):
        print("Partial replay saved. Run again without --limit to finish.",flush=True)
        return
    for name,records in results.items():
        root=args.output/name
        output=root/"all507_pnp"
        output.mkdir(parents=True,exist_ok=True)
        report=output/"integrated_summary.json"
        if not report.exists():
            for file,content in ((root/"all507.csv",rows),(output/"per_pair.csv",records)):
                if not file.exists():
                    write_csv(file,content)
                elif read_csv(file)!=[{k:str(v) for k,v in r.items()} for r in content]:
                    raise ValueError(f"Existing export differs: {file}")
            # Finish an interrupted summary export without overwriting candidate results.
            details=output/"integrated_per_query.csv"
            if details.exists():
                details.unlink()
            summarize(root,"all507")
        print(name,json.loads(report.read_text()),flush=True)
    import pandas as pd
    pd.DataFrame(diagnostics).to_csv(args.output/"depth_support.csv",index=False)
    old=read_csv(args.output/"original/all507_pnp/integrated_per_query.csv")
    new=read_csv(args.output/"depth_fill/all507_pnp/integrated_per_query.csv")
    if [r["query_index"] for r in old]!=[r["query_index"] for r in new]:
        raise ValueError("Selected query identities differ")
    from evaluate_retrieval_topk_pipeline import flag
    changes=[]
    for threshold in (.5,1,2,5,10):
        field=f"success_xy_{threshold}m"
        before=[flag(r[field]) for r in old]
        after=[flag(r[field]) for r in new]
        changes.append(dict(threshold_m=threshold,before=sum(before),after=sum(after),
                            improved=sum(not a and b for a,b in zip(before,after)),
                            worsened=sum(a and not b for a,b in zip(before,after))))
    pd.DataFrame(changes).to_csv(args.output/"transitions.csv",index=False)
    print(pd.DataFrame(changes).to_string(index=False))
    print(f"Done: {args.output}",flush=True)


if __name__=="__main__":
    main()
