"""Inspect cached fixed-pair A/B/C correspondences, depth samples, and local imagery."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from run_priority_validation374 import B_MAPS


def sample(array, point):
    x = int(np.clip(np.round(point[0]), 0, array.shape[1] - 1))
    y = int(np.clip(np.round(point[1]), 0, array.shape[0] - 1))
    return array[y, x]


def neighborhood(depth, index, point):
    x, y = np.round(point).astype(int)
    x, y = np.clip(x, 0, depth.shape[1]-1), np.clip(y, 0, depth.shape[0]-1)
    values = depth[max(0,y-2):y+3, max(0,x-2):x+3]
    valid = (index[max(0,y-2):y+3, max(0,x-2):x+3] >= 0) & np.isfinite(values)
    valid &= (values >= .5) & (values <= 200)
    return float(valid.mean())


def tile(image, point, title):
    x, y = np.round(point).astype(int)
    padded = cv2.copyMakeBorder(image, 80, 80, 80, 80, cv2.BORDER_CONSTANT)
    crop = padded[y:y+161, x:x+161].copy()
    crop = cv2.resize(crop, (322, 322), interpolation=cv2.INTER_NEAREST)
    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    cv2.circle(crop, (160,160), 8, (0,0,255), 1)
    label = np.zeros((45,322,3), np.uint8)
    cv2.putText(label, title, (7,28), cv2.FONT_HERSHEY_SIMPLEX, .52, (255,255,255), 1)
    return np.vstack([label,crop])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads((args.run / "jobs.json").read_text())
    source = Path(json.loads((args.run / "protocol.json").read_text())["source"])
    visible = Path(json.loads((source / "protocol.json").read_text())["source"])
    maps = pd.read_csv(visible / "summary.csv").set_index("db_index")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for segment in dict.fromkeys(job["db_seg"] for job in jobs):
        group = [job for job in jobs if job["db_seg"] == segment]
        example = maps.loc[group[0]["db_index"]]
        intensities = {}
        for name, path in (("A", example.map_A), ("C", example.map_C),
                           ("B", B_MAPS / "training" / segment / "map_static.npz")):
            with np.load(path) as z:
                intensities[name] = z["intensity"].reshape(-1)
        for job in group:
            stem = f"db_{job['db_index']:04d}"
            with np.load(source / (stem + ".npz")) as z:
                cache = {key:z[key].copy() for key in z.files}
            with np.load(args.run / (stem + "_B_matches.npz")) as z:
                bcache = {key:z[key].copy() for key in z.files}
            details = pd.read_csv(args.run / (stem + "_points.csv"))
            rasters = {}
            for name, path in (("A", job["A_png"]), ("B", job["B_png"]), ("C", job["anchor_png"])):
                with np.load(Path(path).with_name("anchor_index.npz")) as z:
                    index, depth = z["point_index"].copy(), z["depth"].copy()
                image = cv2.imread(path, 0)
                if image is None or image.shape != depth.shape:
                    raise ValueError("Missing or incompatible image")
                rasters[name] = (index,depth,image)
            count = {}
            for number, detail in enumerate(details.itertuples()):
                reference = np.array([detail.x,detail.y])
                query_reference = np.array([detail.query_x,detail.query_y])
                record = dict(db_index=job["db_index"], query=job["retrieval_q_idx"], point=number,
                              x=detail.x, y=detail.y, AC_region=detail.AC_region,
                              A_status=detail.A_status, B_status=detail.B_status)
                for name in "ABC":
                    index,depth,image = rasters[name]
                    point_index = int(sample(index, reference))
                    record[name+"_ref_depth_m"] = float(sample(depth,reference)) if point_index >= 0 else None
                    record[name+"_ref_intensity"] = float(intensities[name][point_index]) if point_index >= 0 else None
                    record[name+"_ref_png"] = int(sample(image,reference))
                    record[name+"_ref_valid5x5"] = neighborhood(depth,index,reference)
                for name in ("A","B"):
                    if name == "A":
                        j = int(cache["A_nearest_indices"][number])
                        mk0, mk1 = cache["A_anchor_xy"], cache["A_query_xy"]
                        inliers, errors = cache["A_inlier_indices"], cache["A_gt_reprojection_px"]
                    else:
                        j = int(bcache["chosen_indices"][number])
                        mk0, mk1 = bcache["anchor_xy"], bcache["query_xy"]
                        inliers, errors = bcache["inlier_indices"], bcache["gt_reprojection_px"]
                    record[name+"_match_found"] = j >= 0
                    if j < 0:
                        record[name+"_reason"] = "no_near_anchor_match"
                        continue
                    index,depth,image = rasters[name]
                    d = float(sample(depth,mk0[j]))
                    valid = sample(index,mk0[j]) >= 0 and np.isfinite(d) and .5 <= d <= 200
                    gap = float(np.linalg.norm(mk1[j]-query_reference))
                    good_gt = errors[j] <= 4
                    inlier = bool(np.any(inliers == j))
                    record.update({name+"_matched_x":float(mk0[j,0]), name+"_matched_y":float(mk0[j,1]),
                                   name+"_matched_depth_m":d if np.isfinite(d) else None,
                                   name+"_depth_valid":bool(valid),name+"_query_gap_px":gap,
                                   name+"_gt_reprojection_px":float(errors[j]) if np.isfinite(errors[j]) else None,
                                   name+"_pnp_inlier":inlier, name+"_gt_pass":bool(good_gt)})
                    reason = ("different_query_match" if gap>3 else "invalid_depth" if not valid
                              else "gt_and_pnp_rejected" if not good_gt and not inlier
                              else "gt_rejected" if not good_gt else "pnp_rejected" if not inlier else "retained")
                    record[name+"_reason"] = reason
                key = (record["A_reason"],record["B_reason"])
                # All anchor-96 reference points plus up to 3 of each reason pair elsewhere.
                if job["db_index"] == 345 or count.get(key,0)<3:
                    count[key] = count.get(key,0)+1
                    target = args.output / f"{stem}_point_{number:03d}.png"
                    panels = [tile(rasters[name][2],reference,f"{name} ref x={reference[0]:.0f} y={reference[1]:.0f}") for name in "ABC"]
                    if not cv2.imwrite(str(target),np.hstack(panels)):
                        raise OSError(target)
                    record["patch"] = str(target)
                records.append(record)
            print(f"Inspected {stem}: {len(details)} reference points",flush=True)
    table = pd.DataFrame(records)
    table.to_csv(args.output / "points.csv",index=False)
    counts = table.groupby(["db_index","A_reason","B_reason"]).size().rename("count").reset_index()
    counts.to_csv(args.output / "reason_counts.csv",index=False)
    print(counts[counts.db_index==345].to_string(index=False))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
