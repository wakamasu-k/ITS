"""Check B recovery of fixed C-reference correspondences saved in the A/C audit."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((args.source / "protocol.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(source=str(args.source.resolve()), checkpoint=protocol["checkpoint"],
                  anchor_tolerance_px=3., query_tolerance_px=3., gt_threshold_px=4., seed=0)
    config_path = args.output / "protocol.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output belongs to a different run")
    config_path.write_text(json.dumps(config, indent=2))
    jobs_file = args.output / "jobs.json"
    if not jobs_file.exists():
        import pandas as pd
        from run_tulip_pixel_ablation374 import BASE
        db = pd.read_parquet(BASE / "B/db.parquet")
        jobs = []
        for source_job in protocol["jobs"]:
            job = dict(source_job)
            row = db.iloc[job["db_index"]]
            if row.db_seg != job["db_seg"] or int(row.anchor_id) != int(job["anchor_id"]):
                raise ValueError("B identity mismatch")
            job["B_png"] = row.anchor_png
            if not (args.source / f"db_{job['db_index']:04d}.npz").is_file():
                raise FileNotFoundError("Missing A/C correspondence cache")
            jobs.append(job)
        jobs_file.write_text(json.dumps(jobs, indent=2))
    jobs = json.loads(jobs_file.read_text())
    if args.prepare_only:
        print(f"Prepared {len(jobs)} jobs: {args.output}")
        return
    import torch
    import pandas as pd
    from audit_loftr_generated_contribution import load_evaluator
    from audit_priority_pnp_sources import solve
    from evaluate_retrieval_topk_pipeline import ICP
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    ev = load_evaluator()
    device = torch.device("cuda")
    matcher = ev.load_matcher(config["checkpoint"], device=device, enable_fine=True, enable_repeatability=False)
    icp = ev.load_icp_db_to_q(ICP, db_col=None, q_col=None, mat_col=None, mat_cols=None)
    summaries = []
    generated_segment = None
    generated_points = None
    for job in jobs:
        stem = f"db_{job['db_index']:04d}"
        result_json = args.output / (stem + ".json")
        if result_json.exists():
            summaries.append(json.loads(result_json.read_text()))
            continue
        folder = Path(job["B_png"]).parent
        anchor = json.loads((folder / "anchor.json").read_text())
        reference_anchor = json.loads(Path(job["anchor_json"]).read_text())
        for field in ("width", "height", "intrinsic", "world_to_cam"):
            if not np.allclose(anchor[field], reference_anchor[field], atol=1e-8, rtol=0):
                raise ValueError(f"B/C geometry differs: {field}")
        query = json.loads(Path(job["q_meta_json"]).read_text())
        transform = icp.get(job["db_seg"], job["q_seg"])
        if transform is None:
            raise ValueError("Missing ICP")
        index = ev.load_anchor_index_npz(str(folder / "anchor_index.npz"))
        matches = ev.infer_matches(matcher, ev._read_gray_u8(job["B_png"]),
                                  ev._read_gray_u8(job["q_image_png"]),
                                  resize_long=840, stride=8, cell_point="center", device=device,
                                  amp_enabled=True, amp_dtype=torch.float16, match_source="auto",
                                  conf_th=0., max_matches=4096)
        mk0, mk1 = matches["mkpts0"], matches["mkpts1"]
        h, w = index.depth_x.shape
        u = np.clip(np.round(mk0[:, 0]).astype(int), 0, w-1)
        v = np.clip(np.round(mk0[:, 1]).astype(int), 0, h-1)
        depth = index.depth_x[v, u]
        valid = index.valid_mask[v, u] & np.isfinite(depth) & (depth >= .5) & (depth <= 200)
        pose, inliers, error = solve(ev, mk0, mk1, np.ones(len(mk0), bool), valid,
                                    index, anchor, query, transform)
        gt_error = np.full(len(mk0), np.inf)
        if valid.any():
            fx, fy, cx, cy = ev._parse_intrinsic(anchor)
            xyz = ev.backproject_xforward(mk0[valid, 0], mk0[valid, 1], depth[valid], fx, fy, cx, cy)
            world = ev._transform_points(ev._invert_T(ev._parse_Twc(anchor)), xyz)
            world = ev._transform_points(transform, world)
            cam = ev._transform_points(ev._parse_Twc(query), world)
            fx, fy, cx, cy = ev._parse_intrinsic(query)
            front = cam[:, 0] > .1
            projection = np.full((len(cam), 2), np.inf)
            projection[front, 0] = cx - fx * cam[front, 1] / cam[front, 0]
            projection[front, 1] = cy - fy * cam[front, 2] / cam[front, 0]
            gt_error[valid] = np.linalg.norm(projection - mk1[valid], axis=1)
        inlier_mask = np.zeros(len(mk0), bool)
        inlier_mask[inliers] = True
        with np.load(args.source / (stem + ".npz")) as z:
            refs = z["C_reference_indices"]
            reference0, reference1 = z["C_anchor_xy"][refs], z["C_query_xy"][refs]
            a_status = z["status"]
        with np.load(Path(protocol["source"]) / (stem + "_pixels.npz")) as z:
            region_map = z["category"]
        with np.load(folder / "anchor_index.npz") as z:
            if "source_is_generated" in z:
                provenance = z["source_is_generated"].astype(bool)
            else:
                from run_priority_validation374 import B_MAPS
                if generated_segment != job["db_seg"]:
                    with np.load(B_MAPS / "training" / job["db_seg"] / "map_static.npz") as data:
                        generated_points = data["source_is_generated"].reshape(-1).astype(bool)
                    generated_segment = job["db_seg"]
                point_index = z["point_index"]
                provenance = np.zeros(point_index.shape, bool)
                covered = point_index >= 0
                provenance[covered] = generated_points[point_index[covered]]
        statuses, chosen = [], []
        for point0, point1 in zip(reference0, reference1):
            nearby = np.flatnonzero(np.linalg.norm(mk0 - point0, axis=1) <= 3)
            if not len(nearby):
                statuses.append(0)
                chosen.append(-1)
                continue
            j = nearby[np.argmin(np.linalg.norm(mk1[nearby] - point1, axis=1))]
            chosen.append(int(j))
            if np.linalg.norm(mk1[j] - point1) > 3:
                statuses.append(1)
            elif not valid[j]:
                statuses.append(2)
            elif gt_error[j] > 4 or not inlier_mask[j]:
                statuses.append(3)
            else:
                statuses.append(4)
        b_status = np.array(statuses, np.uint8)
        chosen = np.array(chosen, int)
        recovered = (a_status != 4) & (b_status == 4)
        lost = (a_status == 4) & (b_status != 4)
        kept = (a_status == 4) & (b_status == 4)
        missing = (a_status != 4) & (b_status != 4)
        ref_u = np.clip(np.round(reference0[:, 0]).astype(int), 0, w-1)
        ref_v = np.clip(np.round(reference0[:, 1]).astype(int), 0, h-1)
        regions = region_map[ref_v, ref_u]
        generated = np.full(len(chosen), -1, int)
        if provenance is not None:
            present = chosen >= 0
            generated[present] = provenance[v[chosen[present]], u[chosen[present]]]
        detail = pd.DataFrame(dict(x=reference0[:, 0], y=reference0[:, 1],
                                   query_x=reference1[:, 0], query_y=reference1[:, 1],
                                   A_status=a_status, B_status=b_status, AC_region=regions,
                                   B_generated=generated, recovered=recovered, lost=lost,
                                   still_missing=missing))
        detail.to_csv(args.output / (stem + "_points.csv"), index=False)
        np.savez_compressed(args.output / (stem + "_B_matches.npz"),
                            anchor_xy=mk0, query_xy=mk1, inlier_indices=inliers,
                            gt_reprojection_px=gt_error, chosen_indices=chosen)
        preview = cv2.imread(job["B_png"])
        for point, rec, loss, keep in zip(reference0, recovered, lost, kept):
            color = (0, 220, 0) if rec else (0, 0, 255) if loss else (150, 150, 150) if keep else (0, 220, 255)
            cv2.circle(preview, tuple(np.round(point).astype(int)), 5, color, -1)
        target = args.output / (stem + "_recovery.png")
        if not cv2.imwrite(str(target), preview):
            raise OSError(target)
        ac = json.loads((args.source / (stem + ".json")).read_text())
        record = dict(db_index=job["db_index"], query=job["retrieval_q_idx"],
                      C_reference=len(refs), A_retained=int((a_status == 4).sum()),
                      B_retained=int((b_status == 4).sum()), recovered=int(recovered.sum()),
                      lost=int(lost.sum()), still_missing=int(missing.sum()),
                      recovered_generated=int((recovered & (generated == 1)).sum()),
                      recovered_C_new=int((recovered & (regions == 2)).sum()),
                      missing_C_new=int((missing & (regions == 2)).sum()),
                      A_xy_error=ac["A_xy_error"], B_xy_error=error, C_xy_error=ac["C_xy_error"])
        result_json.write_text(json.dumps(record, indent=2))
        summaries.append(record)
        print(record, flush=True)
    pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
    print(f"Done: {args.output}", flush=True)


if __name__ == "__main__":
    main()
