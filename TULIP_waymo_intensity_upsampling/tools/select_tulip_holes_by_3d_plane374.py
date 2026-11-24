"""Select actual TULIP points at A raster holes using nearby measured 3D points."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from run_tulip_pixel_ablation374 import BASE, VALID, B_MAPS
from audit_A_C_visible_pixels374 import verify_map
from run_fill_followup374 import FILL
from run_priority_validation374 import execute


def decide(points, measured, tree, radius, min_support, rms_limit, distance_limit):
    accepted = np.zeros(len(points),np.uint8)
    for start in range(0,len(points),4096):
        query=points[start:start+4096]
        distance,idx=tree.query(query,k=16,distance_upper_bound=radius,workers=1)
        valid=np.isfinite(distance)
        support=valid.sum(axis=1)
        reason=np.full(len(query),2,np.uint8)
        eligible=support>=min_support
        if eligible.any():
            x=measured[np.minimum(idx[eligible],len(measured)-1)]
            weight=valid[eligible]
            center=(x*weight[:,:,None]).sum(axis=1)/support[eligible,None]
            delta=(x-center[:,None,:])*weight[:,:,None]
            covariance=np.einsum("nki,nkj->nij",delta,delta)/support[eligible,None,None]
            eigenvalues,vectors=np.linalg.eigh(covariance)
            normal=vectors[:,:,0]
            rms=np.sqrt(np.maximum(eigenvalues[:,0],0))
            separation=np.abs(np.einsum("ni,ni->n",query[eligible]-center,normal))
            noncollinear=eigenvalues[:,1]>1e-6
            local=np.full(eligible.sum(),3,np.uint8)
            local[noncollinear & (rms>rms_limit)]=4
            planar=noncollinear & (rms<=rms_limit)
            local[planar]=5
            local[planar & (separation<=distance_limit)]=1
            reason[eligible]=local
        accepted[start:start+len(query)]=reason
    return accepted


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--limit",type=int,default=3,help="anchors per segment; 0 selects all 443")
    parser.add_argument("--radius-m",type=float,default=1.)
    parser.add_argument("--min-support",type=int,default=6)
    parser.add_argument("--plane-rms-m",type=float,default=.05)
    parser.add_argument("--point-plane-m",type=float,default=.1)
    parser.add_argument("--evaluate",action="store_true")
    args=parser.parse_args()
    if args.limit<0 or not 3<=args.min_support<=16 or any(
        not np.isfinite(x) or x<=0 for x in (args.radius_m,args.plane_rms_m,args.point_plane_m)):
        parser.error("Invalid plane settings")
    if args.evaluate and args.limit:
        parser.error("--evaluate requires --limit 0 for the complete DB")
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    config=dict(radius_m=args.radius_m,min_support=args.min_support,plane_rms_m=args.plane_rms_m,
                point_plane_m=args.point_plane_m,neighbors=16,B_maps=str(B_MAPS),
                A_DB=str(BASE/"A/db.parquet"),brightness_reference=str(FILL/"preserve_A_png/db.parquet"))
    protocol=out/"protocol.json"
    if protocol.exists() and json.loads(protocol.read_text())!=config:
        raise ValueError("Output belongs to another selection protocol")
    protocol.write_text(json.dumps(config,indent=2))
    a=pd.read_parquet(BASE/"A/db.parquet")
    b=pd.read_parquet(BASE/"B/db.parquet")
    priority=pd.read_parquet(VALID/"database/db.parquet")
    fields=["pair_id","db_subset","db_seg","anchor_id"]
    if len(a)!=443 or not all(a[fields].astype(str).equals(t[fields].astype(str)) for t in (b,priority)):
        raise ValueError("A/B/priority DB identities differ")
    stats=[]
    paths={}
    for (subset,segment),group in a.groupby(["db_subset","db_seg"],sort=False):
        if args.limit:
            positions=np.unique(np.linspace(0,len(group)-1,min(args.limit,len(group))).astype(int))
            group=group.iloc[positions]
        pending=[i for i in group.index if not (out/"anchors"/f"{i:04d}"/"counts.json").exists()]
        tree=None
        xyz=None
        generated=None
        decision=None
        if pending:
            with np.load(B_MAPS/subset/segment/"map_static.npz") as z:
                xyz=z["xyz"]
                generated=z["source_is_generated"].reshape(-1).astype(bool)
            measured=np.asarray(xyz[~generated],dtype=np.float64)
            tree=cKDTree(measured)
            cache=out/"decisions"/(segment+".npy")
            cache.parent.mkdir(exist_ok=True)
            if cache.exists():
                decision=np.lib.format.open_memmap(cache,mode="r+")
                if decision.shape!=(len(xyz),):
                    raise ValueError("Point count changed")
            else:
                decision=np.lib.format.open_memmap(cache,mode="w+",dtype=np.uint8,shape=(len(xyz),))
                decision[:]=0
            a_to_b=np.flatnonzero(~generated)
        for i,row in group.iterrows():
            folder=out/"anchors"/f"{i:04d}"
            saved=folder/"counts.json"
            if saved.exists():
                stats.append(json.loads(saved.read_text()))
                paths[row.anchor_png]=str(folder/"anchor.png")
                continue
            folder.mkdir(parents=True,exist_ok=True)
            ap,bp,pp=map(Path,(row.anchor_png,b.iloc[i].anchor_png,priority.iloc[i].anchor_png))
            am=json.loads(ap.with_name("anchor.json").read_text())
            bm=json.loads(bp.with_name("anchor.json").read_text())
            for field in ("intrinsic","world_to_cam","width","height"):
                if not np.allclose(am[field],bm[field],atol=1e-8,rtol=0):
                    raise ValueError(f"Geometry differs: {i}")
            with np.load(ap.with_name("anchor_index.npz")) as z:
                ai,ad=z["point_index"].copy(),z["depth"].copy()
            with np.load(bp.with_name("anchor_index.npz")) as z:
                bi,bd=z["point_index"].copy(),z["depth"].copy()
            av,bv=ai>=0,bi>=0
            holes=~av&bv
            ids=np.unique(bi[holes])
            if np.any(~generated[ids]):
                raise ValueError("New hole coverage includes unexpected measured points")
            unknown=ids[decision[ids]==0]
            if len(unknown):
                decision[unknown]=decide(np.asarray(xyz[unknown],dtype=np.float64),measured,tree,
                                         args.radius_m,args.min_support,args.plane_rms_m,args.point_plane_m)
                decision.flush()
            keep=np.zeros(ai.shape,bool)
            keep[holes]=decision[bi[holes]]==1
            original=cv2.imread(str(ap),0)
            supplement=cv2.imread(str(pp),0)
            if original is None or supplement is None or original.shape!=ai.shape or supplement.shape!=ai.shape:
                raise ValueError("Missing or incompatible PNG")
            image=original.copy()
            image[keep]=supplement[keep]
            index=np.full(ai.shape,-1,dtype=np.int64)
            index[av]=a_to_b[ai[av]]
            index[keep]=bi[keep]
            depth=np.full(ai.shape,np.nan,dtype=np.float32)
            depth[av]=ad[av]
            depth[keep]=bd[keep]
            verify_map(xyz,index,depth,np.asarray(am["world_to_cam"]))
            np.savez_compressed(folder/"anchor_index.npz",point_index=index,depth=depth,source_is_generated=keep)
            if not cv2.imwrite(str(folder/"anchor.png"),image):
                raise OSError(folder)
            # Save the exact brightness-preserved all-fill control for visual comparison.
            control=original.copy()
            control[holes]=supplement[holes]
            overlay=cv2.cvtColor(original,cv2.COLOR_GRAY2BGR)
            overlay[holes & ~keep]=(0,0,255)
            overlay[keep]=(0,220,0)
            if not cv2.imwrite(str(folder/"comparison.png"),np.hstack([
                cv2.cvtColor(original,cv2.COLOR_GRAY2BGR),cv2.cvtColor(control,cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),overlay])):
                raise OSError(folder)
            am["image_path"]=str(folder/"anchor.png")
            am["coverage"]=float((av|keep).mean())
            am["point_map_path"]=str(B_MAPS/subset/segment/"map_static.npz")
            am["partial_completion"]=config
            (folder/"anchor.json").write_text(json.dumps(am,indent=2))
            record=dict(db_index=i,segment=segment,anchor_id=row.anchor_id,A_pixels=int(av.sum()),
                        candidate_pixels=int(holes.sum()),accepted_pixels=int(keep.sum()),
                        candidate_unique_generated=len(ids),accepted_unique_generated=int((decision[ids]==1).sum()))
            for code,name in ((2,"insufficient_support"),(3,"degenerate"),(4,"nonplanar"),(5,"far_from_plane")):
                record[name+"_pixels"]=int((decision[bi[holes]]==code).sum())
            saved.write_text(json.dumps(record,indent=2))
            stats.append(record)
            paths[row.anchor_png]=str(folder/"anchor.png")
            print(f"Prepared DB {i}: kept {int(keep.sum())}/{int(holes.sum())} hole pixels",flush=True)
        del tree,xyz,generated,decision
    counts=pd.DataFrame(stats)
    counts.to_csv(out/"pixel_counts.csv",index=False)
    print(counts[["candidate_pixels","accepted_pixels","candidate_unique_generated","accepted_unique_generated"]].sum().to_string())
    if len(paths)==443:
        for table,name in ((a,"db"),(pd.read_parquet(BASE/"A/pairs.parquet"),"pairs")):
            mapped=table.anchor_png.map(paths)
            if mapped.isna().any():
                raise ValueError("Incomplete anchor mapping")
            table.assign(anchor_png=mapped).to_parquet(out/f"{name}.parquet",index=False)
    if args.evaluate:
        command=json.loads((VALID/"retrieval/command.json").read_text())
        dest=out/"retrieval"
        dest.mkdir(exist_ok=True)
        for flag,value in (("--pairs",out/"pairs.parquet"),("--db",out/"db.parquet"),
                           ("--out_dir",dest),("--name","plane_selected")):
            command[command.index(flag)+1]=str(value)
        if not (dest/"plane_selected_per_query.csv").exists():
            execute(command)
        from evaluate_retrieval_topk_pipeline import read_csv,flag
        sources=dict(A=BASE/"A/retrieval/A_per_query.csv",
                     all_holes_preserve_A=FILL/"preserve_A_png/retrieval/preserve_A_png_per_query.csv",
                     plane_selected=dest/"plane_selected_per_query.csv")
        frames={n:pd.DataFrame(read_csv(p)) for n,p in sources.items()}
        keys=["pair_id","q_cam_png","gt_index_db"]
        ref=frames["A"][keys].astype(str)
        summaries=[]
        changes=[]
        for name,table in frames.items():
            if len(table)!=374 or not table[keys].astype(str).equals(ref):
                raise ValueError("Query identities/order differ")
            for metric in ("strict_hit@10","soft_t2m_hit@10","soft_t5m_hit@10"):
                values=table[metric].map(flag)
                summaries.append(dict(condition=name,metric=metric,queries=374,hits=int(values.sum())))
                if name!="plane_selected":
                    other=frames["plane_selected"][metric].map(flag)
                    changes.append(dict(before=name,after="plane_selected",metric=metric,
                                        improved=int((~values&other).sum()),worsened=int((values&~other).sum())))
        pd.DataFrame(summaries).to_csv(out/"comparison.csv",index=False)
        pd.DataFrame(changes).to_csv(out/"transitions.csv",index=False)
        print(pd.DataFrame(summaries).to_string(index=False))
    print(f"Output: {out}",flush=True)


if __name__=="__main__":
    main()
