"""Compare fixed A/C pairs at C inliers validated by known query geometry."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


BASE = Path("/mnt/w/its_outputs/priority_validation374_abc_gXMLfD")


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    protocol_file = args.output / "protocol.json"
    if not args.prepare_only and protocol_file.exists():
        protocol = json.loads(protocol_file.read_text())
        if protocol["source"] != str(args.source.resolve()):
            raise ValueError("Source differs from prepared protocol")
        jobs = protocol["jobs"]
    else:
        cases = read_csv(args.source / "summary.csv")
        candidates = read_csv(BASE / "C/top10.csv")
        import pandas as pd
        a_db = pd.read_parquet(BASE / "A/db.parquet")
        c_db = pd.read_parquet(BASE / "C/db.parquet")
        jobs = []
        for case in cases:
            eligible = [r for r in candidates if r["db_seg"] == case["segment"]
                        and int(r["anchor_id"]) == int(case["anchor_id"]) and r["retrieval_is_same_pair"] == "1"]
            fallback = not eligible
            if fallback:
                eligible = [r for r in candidates if r["db_seg"] == case["segment"]
                            and r["retrieval_is_same_pair"] == "1"]
            if not eligible:
                raise ValueError(f"No same-scene query for {case['segment']}")
            row = min(eligible, key=lambda r: (abs(int(r["anchor_id"]) - int(case["anchor_id"])),
                                              float(r["dist_xy_m"]), int(r["retrieval_q_idx"])))
            idx = int(case["db_index"])
            cp = Path(c_db.iloc[idx].anchor_png)
            job = dict(row, db_index=idx, anchor_id=case["anchor_id"],
                       A_png=a_db.iloc[idx].anchor_png, anchor_png=str(cp),
                       anchor_json=str(cp.with_name("anchor.json")),
                       anchor_index_npz=str(cp.with_name("anchor_index.npz")),
                       query_chosen_via_nearby_anchor=fallback)
            for field in ("q_image_png", "q_meta_json", "anchor_png", "anchor_json", "anchor_index_npz", "A_png"):
                if not Path(job[field]).is_file():
                    raise FileNotFoundError(job[field])
            jobs.append(job)
        args.output.mkdir(parents=True, exist_ok=True)
        protocol = dict(source=str(args.source.resolve()), jobs=jobs,
                        near_anchor_px=3., near_query_px=3., gt_reprojection_px=4.,
                        resize_long=840, seed=0, checkpoint="/home/wakamatsu/ITS/waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/stage1_final_gstep715958.pth")
        protocol_file = args.output / "protocol.json"
        if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
            raise ValueError("Output belongs to another experiment")
        protocol_file.write_text(json.dumps(protocol, indent=2))
        print(f"Prepared {len(jobs)} fixed A/C pairs; {args.output}", flush=True)
    if args.prepare_only:
        return
    import torch
    from audit_loftr_generated_contribution import load_evaluator
    from audit_priority_pnp_sources import solve
    from evaluate_retrieval_topk_pipeline import ICP
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with .venv_tulip Python")
    ev = load_evaluator()
    icp = ev.load_icp_db_to_q(ICP, db_col=None, q_col=None, mat_col=None, mat_cols=None)
    device = torch.device("cuda")
    matcher = ev.load_matcher(protocol["checkpoint"], device=device, enable_fine=True, enable_repeatability=False)
    summaries = []
    for job in jobs:
        prefix = args.output / f"db_{job['db_index']:04d}"
        result_json = prefix.with_suffix(".json")
        if result_json.exists():
            summaries.append(json.loads(result_json.read_text()))
            continue
        query = json.loads(Path(job["q_meta_json"]).read_text())
        transform = icp.get(job["db_seg"], job["q_seg"])
        if transform is None:
            raise ValueError("No DB-to-query transform")
        data, saved = {}, {}
        for name, image_path in (("A", job["A_png"]), ("C", job["anchor_png"])):
            folder = Path(image_path).parent
            anchor = json.loads((folder / "anchor.json").read_text())
            index = ev.load_anchor_index_npz(str(folder / "anchor_index.npz"))
            matches = ev.infer_matches(matcher, ev._read_gray_u8(image_path),
                                      ev._read_gray_u8(job["q_image_png"]),
                                      resize_long=840, stride=8, cell_point="center", device=device,
                                      amp_enabled=True, amp_dtype=torch.float16, match_source="auto",
                                      conf_th=0., max_matches=4096)
            mk0, mk1 = matches["mkpts0"], matches["mkpts1"]
            h, w = index.depth_x.shape
            u = np.clip(np.round(mk0[:, 0]).astype(int), 0, w - 1)
            v = np.clip(np.round(mk0[:, 1]).astype(int), 0, h - 1)
            depth = index.depth_x[v, u]
            valid = index.valid_mask[v, u] & np.isfinite(depth) & (depth >= .5) & (depth <= 200)
            pose, inliers, error = solve(ev, mk0, mk1, np.ones(len(mk0), bool), valid,
                                         index, anchor, query, transform)
            # Validate each correspondence with the known camera pose and ICP.
            gt_error = np.full(len(mk0), np.inf)
            if valid.any():
                fx, fy, cx, cy = ev._parse_intrinsic(anchor)
                xyz = ev.backproject_xforward(mk0[valid, 0], mk0[valid, 1], depth[valid], fx, fy, cx, cy)
                world = ev._transform_points(ev._invert_T(ev._parse_Twc(anchor)), xyz)
                world = ev._transform_points(transform, world)
                cam = ev._transform_points(ev._parse_Twc(query), world)
                fx, fy, cx, cy = ev._parse_intrinsic(query)
                front = cam[:, 0] > .1
                projected = np.full((len(cam), 2), np.inf)
                projected[front, 0] = cx - fx * cam[front, 1] / cam[front, 0]
                projected[front, 1] = cy - fy * cam[front, 2] / cam[front, 0]
                gt_error[valid] = np.linalg.norm(projected - mk1[valid], axis=1)
            data[name] = dict(mk0=mk0, mk1=mk1, inliers=inliers, valid=valid,
                              good=gt_error <= 4., error=error, pose=pose)
            saved.update({name + "_anchor_xy": mk0, name + "_query_xy": mk1,
                          name + "_inlier_indices": inliers, name + "_gt_reprojection_px": gt_error})
        for field in ("intrinsic", "world_to_cam"):
            am = json.loads(Path(job["A_png"]).with_name("anchor.json").read_text())
            cm = json.loads(Path(job["anchor_json"]).read_text())
            if not np.allclose(am[field], cm[field], atol=1e-8, rtol=0):
                raise ValueError("A/C anchor geometry differs")
        c, a = data["C"], data["A"]
        reference = c["inliers"][c["good"][c["inliers"]]]
        statuses, nearest = [], []
        a_inlier = np.zeros(len(a["mk0"]), bool)
        a_inlier[a["inliers"]] = True
        for j in reference:
            spatial = np.linalg.norm(a["mk0"] - c["mk0"][j], axis=1) <= 3.
            if not spatial.any():
                statuses.append(0)
                nearest.append(-1)
                continue
            ids = np.flatnonzero(spatial)
            candidate = ids[np.argmin(np.linalg.norm(a["mk1"][ids] - c["mk1"][j], axis=1))]
            corresponding = np.linalg.norm(a["mk1"][candidate] - c["mk1"][j]) <= 3.
            nearest.append(int(candidate))
            if not corresponding:
                statuses.append(1)
            elif not a["valid"][candidate]:
                statuses.append(2)
            elif not a["good"][candidate] or not a_inlier[candidate]:
                statuses.append(3)
            else:
                statuses.append(4)
        statuses = np.asarray(statuses, np.uint8)
        saved.update(C_reference_indices=reference, A_nearest_indices=np.array(nearest), status=statuses)
        np.savez_compressed(prefix.with_suffix(".npz"), **saved)
        with np.load(args.source / f"db_{job['db_index']:04d}_pixels.npz") as z:
            category = z["category"]
        positions = c["mk0"][reference]
        h, w = category.shape
        u = np.clip(np.round(positions[:, 0]).astype(int), 0, w-1)
        v = np.clip(np.round(positions[:, 1]).astype(int), 0, h-1)
        regions = category[v, u]
        detail = [dict(C_match_index=int(j), x=float(c["mk0"][j, 0]), y=float(c["mk0"][j, 1]),
                       query_x=float(c["mk1"][j, 0]), query_y=float(c["mk1"][j, 1]),
                       status=int(status), AC_region=int(region))
                  for j, status, region in zip(reference, statuses, regions)]
        with prefix.with_suffix(".csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["C_match_index","x","y","query_x","query_y","status","AC_region"])
            writer.writeheader()
            writer.writerows(detail)
        image = cv2.imread(job["anchor_png"])
        colors = [(0, 0, 255), (0, 150, 255), (255, 100, 0), (0, 220, 255), (0, 220, 0)]
        for point, status in zip(positions, statuses):
            cv2.circle(image, tuple(np.round(point).astype(int)), 5, colors[int(status)], -1)
        if not cv2.imwrite(str(prefix) + "_locations.png", image):
            raise OSError(prefix)
        summary = dict(db_index=job["db_index"], query=job["retrieval_q_idx"],
                       C_reference=len(reference), A_matches=len(a["mk0"]), C_matches=len(c["mk0"]),
                       A_xy_error=a["error"], C_xy_error=c["error"],
                       **{f"status_{i}": int((statuses == i).sum()) for i in range(5)},
                       missing_on_C_new=int(((statuses != 4) & (regions == 2)).sum()),
                       missing_on_C_changed=int(((statuses != 4) & (regions == 3)).sum()))
        result_json.write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
        print(summary, flush=True)
    import pandas as pd
    pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
    print(f"Done: {args.output}", flush=True)


if __name__ == "__main__":
    main()
